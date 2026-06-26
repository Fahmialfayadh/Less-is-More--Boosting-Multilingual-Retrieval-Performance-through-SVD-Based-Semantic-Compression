import os
import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}", flush=True)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

run_cmd("pip install 'datasets<3.0' mteb ranx scikit-learn accelerate sentence-transformers")

os.makedirs("src", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("runs_external", exist_ok=True)

with open('src/qwen_ablation_filter.py', 'w') as f: f.write(r'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os

class QwenEmbedFilterPipeline:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B", vh_path: str = "models/vh_matrix.pt"):
        print(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=self.device)
        self.model.eval()
        
        self.vh_matrix = None
        if vh_path and os.path.exists(vh_path):
            self.vh_matrix = torch.load(vh_path, map_location=self.device, weights_only=True)

    def get_raw_embeddings(self, texts: list[str]) -> torch.Tensor:
        from tqdm import tqdm
        batch_size = 4
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Embeddings"):
            batch_texts = texts[i:i+batch_size]
            prompts = [f'Summarize the sentence: "{text}" in one word:"' for text in batch_texts]
            inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1] 
                last_token_hidden_states = hidden_states[:, -1, :]
                all_embeddings.append(last_token_hidden_states.cpu().float()) 
                del outputs
                del hidden_states
                del inputs
            torch.cuda.empty_cache()
        return torch.cat(all_embeddings, dim=0)

    def encode_baseline(self, precomputed_raw_embs: torch.Tensor) -> np.ndarray:
        return precomputed_raw_embs.cpu().numpy()

    def encode_filtered(self, precomputed_raw_embs: torch.Tensor, window: tuple = None) -> np.ndarray:
        raw_embs = precomputed_raw_embs.to(self.device)
        if self.vh_matrix is not None and window is not None:
            start_idx, end_idx = window
            vh_sub = self.vh_matrix[start_idx:end_idx, :]
            filtered_embs = torch.matmul(raw_embs, vh_sub.T)
            return filtered_embs.cpu().numpy()
        return raw_embs.cpu().numpy()
''')

with open('src/extract_vh_matrix.py', 'w') as f: f.write(r'''
import torch
from transformers import AutoModelForCausalLM
import os
import time

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    pt_path = "models/vh_matrix.pt"
    if os.path.exists(pt_path):
        print(f"Vh matrix already exists at {pt_path}. Skipping extraction.")
        return
    print(f"Loading model {model_name} to extract lm_head.weight...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    W = model.lm_head.weight.detach().cpu().float()
    start = time.time()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    os.makedirs("models", exist_ok=True)
    torch.save(Vh, pt_path)
    print(f"Saved Vh matrix to {pt_path}")

if __name__ == "__main__":
    main()
''')

with open('src/evaluate_external_rag.py', 'w') as f: f.write(r'''
import os
import json
import random
from qwen_ablation_filter import QwenEmbedFilterPipeline
from datasets import load_dataset
from ranx import Qrels, Run, evaluate
from sklearn.metrics.pairwise import cosine_similarity
import torch
import gc

def evaluate_on_miracl():
    print("\n--- Loading MTEB MIRACL ---")
    qrels_ds = load_dataset('mteb/MIRACLRetrieval', 'id-qrels', split='dev')
    queries_ds = load_dataset('mteb/MIRACLRetrieval', 'id-queries', split='dev')
    corpus_ds = load_dataset('mteb/MIRACLRetrieval', 'id-corpus', split='dev', streaming=True)
    
    random.seed(42)
    all_q_ids = list(set([r['query-id'] for r in qrels_ds]))
    sampled_q_ids = random.sample(all_q_ids, min(500, len(all_q_ids)))
    
    pos_corpus_ids = set()
    qrels_dict = {}
    for r in qrels_ds:
        qid = r['query-id']
        cid = r['corpus-id']
        if qid in sampled_q_ids:
            pos_corpus_ids.add(cid)
            if qid not in qrels_dict: qrels_dict[qid] = {}
            qrels_dict[qid][cid] = r['score']
            
    pos_docs = {}
    neg_docs = []
    neg_doc_ids = []
    
    for c in corpus_ds:
        cid = c['_id']
        if cid in pos_corpus_ids:
            if cid not in pos_docs: pos_docs[cid] = c['text']
        else:
            if len(neg_docs) < 10000:
                neg_docs.append(c['text'])
                neg_doc_ids.append(cid)
        if len(pos_docs) == len(pos_corpus_ids) and len(neg_docs) >= 10000:
            break
            
    rag_docs = list(pos_docs.values()) + neg_docs
    rag_doc_ids = list(pos_docs.keys()) + neg_doc_ids
    rag_queries = queries_ds.filter(lambda x: x['_id'] in sampled_q_ids)['text']
    rag_query_ids = queries_ds.filter(lambda x: x['_id'] in sampled_q_ids)['_id']
    return rag_queries, rag_query_ids, rag_docs, rag_doc_ids, qrels_dict

def evaluate_on_mrtydi():
    print("\n--- Loading Mr.TyDi Indonesian ---")
    ds = load_dataset('castorini/mr-tydi', 'indonesian', split='test', trust_remote_code=True)
    
    random.seed(42)
    sampled_indices = random.sample(range(len(ds)), min(500, len(ds)))
    sampled_ds = ds.select(sampled_indices)
    
    qrels_dict = {}
    pos_corpus_ids = set()
    rag_queries = []
    rag_query_ids = []
    
    for item in sampled_ds:
        qid = item['query_id']
        rag_query_ids.append(qid)
        rag_queries.append(item['query'])
        qrels_dict[qid] = {}
        for pos in item['positive_passages']:
            cid = pos['docid']
            pos_corpus_ids.add(cid)
            qrels_dict[qid][cid] = 1
            
    pos_docs = {}
    neg_docs = []
    neg_doc_ids = []
    
    corpus_stream = load_dataset('castorini/mr-tydi-corpus', 'indonesian', split='train', streaming=True, trust_remote_code=True)
    for c in corpus_stream:
        cid = c['docid']
        if cid in pos_corpus_ids:
            if cid not in pos_docs: pos_docs[cid] = c['text']
        else:
            if len(neg_docs) < 10000:
                neg_docs.append(c['text'])
                neg_doc_ids.append(cid)
        if len(pos_docs) == len(pos_corpus_ids) and len(neg_docs) >= 10000:
            break
            
    rag_docs = list(pos_docs.values()) + neg_docs
    rag_doc_ids = list(pos_docs.keys()) + neg_doc_ids
    return rag_queries, rag_query_ids, rag_docs, rag_doc_ids, qrels_dict

def main():
    mq_text, mq_id, md_text, md_id, m_qrels = evaluate_on_miracl()
    tq_text, tq_id, td_text, td_id, t_qrels = evaluate_on_mrtydi()
    
    results_miracl = []
    results_mrtydi = []
    
    print("\n=== Model 1: Qwen2.5-1.5B ===")
    pipeline = QwenEmbedFilterPipeline(model_name="Qwen/Qwen2.5-1.5B", vh_path="models/vh_matrix.pt")
    
    # Qwen MIRACL
    q_mq_embs = pipeline.get_raw_embeddings(mq_text)
    q_md_embs = pipeline.get_raw_embeddings(md_text)
    b_q_mq, b_q_md = pipeline.encode_baseline(q_mq_embs), pipeline.encode_baseline(q_md_embs)
    sims = cosine_similarity(b_q_mq, b_q_md)
    run_dict = {qid: {md_id[idx]: float(sims[i][idx]) for idx in sims[i].argsort()[-100:][::-1]} for i, qid in enumerate(mq_id)}
    results_miracl.append({"model": "Qwen Baseline (1536D)", **evaluate(Qrels(m_qrels), Run(run_dict), ["ndcg@10", "recall@100"])})
    
    f_q_mq, f_q_md = pipeline.encode_filtered(q_mq_embs, window=(0,768)), pipeline.encode_filtered(q_md_embs, window=(0,768))
    sims = cosine_similarity(f_q_mq, f_q_md)
    run_dict = {qid: {md_id[idx]: float(sims[i][idx]) for idx in sims[i].argsort()[-100:][::-1]} for i, qid in enumerate(mq_id)}
    results_miracl.append({"model": "Qwen Indonesian-Retention (768D)", **evaluate(Qrels(m_qrels), Run(run_dict), ["ndcg@10", "recall@100"])})
    
    # Qwen MrTyDi
    q_tq_embs = pipeline.get_raw_embeddings(tq_text)
    q_td_embs = pipeline.get_raw_embeddings(td_text)
    b_q_tq, b_q_td = pipeline.encode_baseline(q_tq_embs), pipeline.encode_baseline(q_td_embs)
    sims = cosine_similarity(b_q_tq, b_q_td)
    run_dict = {qid: {td_id[idx]: float(sims[i][idx]) for idx in sims[i].argsort()[-100:][::-1]} for i, qid in enumerate(tq_id)}
    results_mrtydi.append({"model": "Qwen Baseline (1536D)", **evaluate(Qrels(t_qrels), Run(run_dict), ["ndcg@10", "recall@100"])})
    
    f_q_tq, f_q_td = pipeline.encode_filtered(q_tq_embs, window=(0,768)), pipeline.encode_filtered(q_td_embs, window=(0,768))
    sims = cosine_similarity(f_q_tq, f_q_td)
    run_dict = {qid: {td_id[idx]: float(sims[i][idx]) for idx in sims[i].argsort()[-100:][::-1]} for i, qid in enumerate(tq_id)}
    results_mrtydi.append({"model": "Qwen Indonesian-Retention (768D)", **evaluate(Qrels(t_qrels), Run(run_dict), ["ndcg@10", "recall@100"])})
    
    del pipeline; gc.collect(); torch.cuda.empty_cache()
    
    print("\n=== Model 2: BAAI/bge-m3 ===")
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer('BAAI/bge-m3')
    bge.max_seq_length = 512
    
    bge_mq, bge_md = bge.encode(mq_text, batch_size=16), bge.encode(md_text, batch_size=16)
    sims = cosine_similarity(bge_mq, bge_md)
    run_dict = {qid: {md_id[idx]: float(sims[i][idx]) for idx in sims[i].argsort()[-100:][::-1]} for i, qid in enumerate(mq_id)}
    results_miracl.append({"model": "BAAI/bge-m3 (1024D)", **evaluate(Qrels(m_qrels), Run(run_dict), ["ndcg@10", "recall@100"])})
    
    bge_tq, bge_td = bge.encode(tq_text, batch_size=16), bge.encode(td_text, batch_size=16)
    sims = cosine_similarity(bge_tq, bge_td)
    run_dict = {qid: {td_id[idx]: float(sims[i][idx]) for idx in sims[i].argsort()[-100:][::-1]} for i, qid in enumerate(tq_id)}
    results_mrtydi.append({"model": "BAAI/bge-m3 (1024D)", **evaluate(Qrels(t_qrels), Run(run_dict), ["ndcg@10", "recall@100"])})
    
    del bge; gc.collect(); torch.cuda.empty_cache()
    
    print("\n=== Model 3: Multilingual-E5-large ===")
    e5 = SentenceTransformer('intfloat/multilingual-e5-large')
    e5.max_seq_length = 512
    e5_mq_text = [f"query: {t}" for t in mq_text]
    e5_md_text = [f"passage: {t}" for t in md_text]
    e5_tq_text = [f"query: {t}" for t in tq_text]
    e5_td_text = [f"passage: {t}" for t in td_text]
    
    e5_mq, e5_md = e5.encode(e5_mq_text, batch_size=16), e5.encode(e5_md_text, batch_size=16)
    sims = cosine_similarity(e5_mq, e5_md)
    run_dict = {qid: {md_id[idx]: float(sims[i][idx]) for idx in sims[i].argsort()[-100:][::-1]} for i, qid in enumerate(mq_id)}
    results_miracl.append({"model": "Multilingual-E5-large (1024D)", **evaluate(Qrels(m_qrels), Run(run_dict), ["ndcg@10", "recall@100"])})
    
    e5_tq, e5_td = e5.encode(e5_tq_text, batch_size=16), e5.encode(e5_td_text, batch_size=16)
    sims = cosine_similarity(e5_tq, e5_td)
    run_dict = {qid: {td_id[idx]: float(sims[i][idx]) for idx in sims[i].argsort()[-100:][::-1]} for i, qid in enumerate(tq_id)}
    results_mrtydi.append({"model": "Multilingual-E5-large (1024D)", **evaluate(Qrels(t_qrels), Run(run_dict), ["ndcg@10", "recall@100"])})
    
    print("\n" + "="*50)
    print("RESULTS MIRACL INDONESIAN")
    print("="*50)
    for r in results_miracl: print(r)
        
    print("\n" + "="*50)
    print("RESULTS MR.TYDI INDONESIAN")
    print("="*50)
    for r in results_mrtydi: print(r)
    
if __name__ == "__main__":
    main()
''')

print("=== 1. Extract Vh Matrix ===")
run_cmd("python -u src/extract_vh_matrix.py")

print("=== 2. Running External Baselines Evaluation ===")
run_cmd("python -u src/evaluate_external_rag.py")
