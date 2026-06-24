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

run_cmd("pip install mteb datasets ranx scikit-learn accelerate")

os.makedirs("src", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("runs", exist_ok=True)

with open('src/qwen_embed_filter.py', 'w') as f: f.write(r'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os

class QwenEmbedFilterPipeline:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B", vh_path: str = "models/vh_matrix.pt"):
        """
        Initializes the model, tokenizer, and loads the Vh Matrix.
        """
        print(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()
        
        self.vh_matrix = None
        if vh_path and os.path.exists(vh_path):
            print(f"Loading Vh Matrix from: {vh_path}")
            self.vh_matrix = torch.load(vh_path, map_location=self.device, weights_only=True)

    def get_raw_embeddings(self, texts: list[str]) -> torch.Tensor:
        from tqdm import tqdm
        batch_size = 8
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
                all_embeddings.append(last_token_hidden_states.cpu().float()) # Move to CPU immediately to prevent GPU OOM
                
        return torch.cat(all_embeddings, dim=0)

    def encode_baseline(self, precomputed_raw_embs: torch.Tensor) -> np.ndarray:
        return precomputed_raw_embs.cpu().numpy()

    def encode_filtered(self, precomputed_raw_embs: torch.Tensor, filter_type: str = "edge", filter_ratio: int = 2) -> np.ndarray:
        """
        Applies linear projection post-hoc.
        filter_type: "edge", "head", or "tail"
        """
        raw_embs = precomputed_raw_embs.to(self.device)
        
        if self.vh_matrix is not None:
            hidden_size = self.vh_matrix.shape[0]
            filter_num = hidden_size // filter_ratio
            
            if filter_type == "edge":
                start_idx = (hidden_size - filter_num) // 2
                end_idx = start_idx + filter_num
                vh_sub = self.vh_matrix[start_idx:end_idx, :]
            elif filter_type == "head":
                vh_sub = self.vh_matrix[0:filter_num, :]  # Head = largest singular values = index 0 to k
            elif filter_type == "tail":
                vh_sub = self.vh_matrix[-filter_num:, :]  # Tail = smallest singular values = index -k to end
            else:
                raise ValueError("filter_type must be 'edge', 'head', or 'tail'")
                
            filtered_embs = torch.matmul(raw_embs, vh_sub.T)
            return filtered_embs.cpu().numpy()
        else:
            print("Warning: No Vh matrix loaded. Returning baseline embeddings.")
            return raw_embs.cpu().numpy()
''')

with open('src/extract_crosslingual_filters.py', 'w') as f: f.write(r'''
import torch
from transformers import AutoModelForCausalLM
import os

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    
    print(f"Loading model {model_name} to extract lm_head.weight...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Extracting lm_head.weight and moving to CPU as float32...")
    W = model.lm_head.weight.detach().cpu().float() # Shape: (Vocab, Hidden)
    
    print("Computing Full SVD on the raw unembedding matrix...")
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # Vh is Shape: (Hidden, Hidden) -> (1536, 1536)
    print(f"Vh shape: {Vh.shape}")
    
    os.makedirs("models", exist_ok=True)
    pt_path = "models/vh_matrix.pt"
    torch.save(Vh, pt_path)
    print(f"Saved Vh matrix to {pt_path}")

if __name__ == "__main__":
    main()
''')

with open('src/evaluate_crosslingual_rag.py', 'w') as f: f.write(r'''
import os
import json
import random
from qwen_embed_filter import QwenEmbedFilterPipeline
from datasets import load_dataset
from ranx import Qrels

def main():
    print("Loading EmbedFilter Pipeline for Qwen2.5-1.5B...")
    
    pipeline = QwenEmbedFilterPipeline(
        model_name="Qwen/Qwen2.5-1.5B",
        vh_path="models/vh_matrix.pt"
    )
    
    # 1. Precompute STS Embeddings
    print("Precomputing STS Embeddings...")
    sts_ds = load_dataset("LazarusNLP/stsb_mt_id", split="test")
    if "text_1" in sts_ds.column_names:
        s1_data = sts_ds["text_1"]
        s2_data = sts_ds["text_2"]
    elif "text1" in sts_ds.column_names:
        s1_data = sts_ds["text1"]
        s2_data = sts_ds["text2"]
    else:
        s1_data = sts_ds["sentence1"]
        s2_data = sts_ds["sentence2"]
        
    limit = min(500, len(s1_data))
    s1 = s1_data[:limit]
    s2 = s2_data[:limit]
    pre_sts1 = pipeline.get_raw_embeddings(s1)
    pre_sts2 = pipeline.get_raw_embeddings(s2)
    
    # 2. Precompute RAG Embeddings (MIRACL)
    print("Loading MTEB MIRACL for RAG...")
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
            if qid not in qrels_dict:
                qrels_dict[qid] = {}
            qrels_dict[qid][cid] = r['score']
            
    print(f"Streaming corpus dataset to collect all {len(pos_corpus_ids)} positives and 20,000 random negatives...")
    pos_docs = {}
    neg_docs = []
    neg_doc_ids = []
    
    for c in corpus_ds:
        cid = c['_id']
        if cid in pos_corpus_ids:
            if cid not in pos_docs:
                pos_docs[cid] = c['text']
        else:
            if len(neg_docs) < 5000:
                neg_docs.append(c['text'])
                neg_doc_ids.append(cid)
                
        if len(pos_docs) == len(pos_corpus_ids) and len(neg_docs) >= 5000:
            break
            
    rag_docs = list(pos_docs.values()) + neg_docs
    rag_doc_ids = list(pos_docs.keys()) + neg_doc_ids
    print(f"Corpus streaming complete. Retained {len(rag_docs)} documents.")
    
    rag_queries = queries_ds.filter(lambda x: x['_id'] in sampled_q_ids)['text']
    rag_query_ids = queries_ds.filter(lambda x: x['_id'] in sampled_q_ids)['_id']
    
    print(f"Precomputing RAG embeddings for {len(rag_queries)} queries and {len(rag_docs)} documents...")
    pre_rag_docs = pipeline.get_raw_embeddings(rag_docs)
    pre_rag_queries = pipeline.get_raw_embeddings(rag_queries)
    
    os.makedirs("runs", exist_ok=True)
    Qrels(qrels_dict).save("runs/qrels.json")
    
    from scipy.stats import spearmanr
    from sklearn.metrics.pairwise import cosine_similarity
    from ranx import Run, evaluate
    
    def eval_sts_custom(filter_type=None):
        if filter_type is None:
            embs1 = pipeline.encode_baseline(pre_sts1)
            embs2 = pipeline.encode_baseline(pre_sts2)
        else:
            embs1 = pipeline.encode_filtered(pre_sts1, filter_type=filter_type)
            embs2 = pipeline.encode_filtered(pre_sts2, filter_type=filter_type)
            
        sims = []
        for e1, e2 in zip(embs1, embs2):
            sim = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0][0]
            sims.append(sim)
            
        if "label" in sts_ds.column_names:
            labels = sts_ds["label"][:limit]
        elif "score" in sts_ds.column_names:
            labels = sts_ds["score"][:limit]
        elif "similarity_score" in sts_ds.column_names:
            labels = sts_ds["similarity_score"][:limit]
        corr, _ = spearmanr(sims, labels)
        return corr
        
    def eval_rag_custom(run_name, filter_type=None):
        if filter_type is None:
            doc_embs = pipeline.encode_baseline(pre_rag_docs)
            query_embs = pipeline.encode_baseline(pre_rag_queries)
        else:
            doc_embs = pipeline.encode_filtered(pre_rag_docs, filter_type=filter_type)
            query_embs = pipeline.encode_filtered(pre_rag_queries, filter_type=filter_type)
            
        print(f"Computing cosine similarity for RAG ({run_name})...")
        sim_matrix = cosine_similarity(query_embs, doc_embs)
        
        run_dict = {}
        for i, q_id in enumerate(rag_query_ids):
            run_dict[q_id] = {}
            scores = sim_matrix[i]
            top_indices = scores.argsort()[-100:][::-1]
            for idx in top_indices:
                run_dict[q_id][rag_doc_ids[idx]] = float(scores[idx])
                
        run = Run(run_dict)
        run.name = run_name
        run.save(f"runs/{run_name}.json")
        
        qrels = Qrels(qrels_dict)
        return evaluate(qrels, run, ["ndcg@10", "recall@100"])

    results = []

    # 1. Baseline
    print("Evaluating Baseline...")
    sts_base = eval_sts_custom(filter_type=None)
    rag_base = eval_rag_custom("run_baseline", filter_type=None)
    results.append({"config": "Baseline", "sts_spearman": sts_base, **rag_base})

    # 2. Edge Filter
    print("Evaluating Edge Filter...")
    sts_edge = eval_sts_custom(filter_type="edge")
    rag_edge = eval_rag_custom("run_edge_filter", filter_type="edge")
    results.append({"config": "Edge-Filter", "sts_spearman": sts_edge, **rag_edge})
    
    # 3. Head Filter
    print("Evaluating Head Filter...")
    sts_head = eval_sts_custom(filter_type="head")
    rag_head = eval_rag_custom("run_head_filter", filter_type="head")
    results.append({"config": "Head-Filter", "sts_spearman": sts_head, **rag_head})

    # 4. Tail Filter
    print("Evaluating Tail Filter...")
    sts_tail = eval_sts_custom(filter_type="tail")
    rag_tail = eval_rag_custom("run_tail_filter", filter_type="tail")
    results.append({"config": "Tail-Filter", "sts_spearman": sts_tail, **rag_tail})
    
    with open("runs/metrics_crosslingual.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nResults:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
''')

with open('src/hypothesis_test_crosslingual.py', 'w') as f: f.write(r'''
import os
from ranx import Qrels, Run, compare

def main():
    print("Loading qrels and exported runs...")
    
    runs_dir = "runs"
        
    qrels = Qrels.from_file(os.path.join(runs_dir, "qrels.json"))
    
    runs = []
    for filename in ["run_baseline.json", "run_edge_filter.json", "run_head_filter.json", "run_tail_filter.json"]:
        run_path = os.path.join(runs_dir, filename)
        if os.path.exists(run_path):
            run = Run.from_file(run_path)
            run.name = filename.replace(".json", "")
            runs.append(run)
            
    if len(runs) < 4:
        print("Not all runs found!")
        return
        
    print(f"Loaded {len(runs)} runs. Performing Paired Student's t-test with alpha=0.01...")
    
    report = compare(
        qrels=qrels,
        runs=runs, # Baseline is first
        metrics=["ndcg@10", "recall@100"],
        stat_test="student",
        max_p=0.01
    )
    
    print("\n" + "="*50)
    print("      STATISTICAL SIGNIFICANCE REPORT (p < 0.01)")
    print("="*50)
    print(report)
    print("\nNote: Superscripts denote significant differences.")

if __name__ == "__main__":
    main()
''')

print("=== Running Extraction ===")
run_cmd("python -u src/extract_crosslingual_filters.py")

print("=== Running Evaluation ===")
run_cmd("python -u src/evaluate_crosslingual_rag.py")

print("=== Running Hypothesis Test ===")
run_cmd("python -u src/hypothesis_test_crosslingual.py > runs/significance_report.txt")
run_cmd("cat runs/significance_report.txt")

# Print the contents of metrics to console
run_cmd("cat runs/metrics_crosslingual.json")

print("\nALL DONE.")
