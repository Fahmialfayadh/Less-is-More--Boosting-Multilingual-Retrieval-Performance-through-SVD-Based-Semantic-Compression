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
os.makedirs("runs_ablation", exist_ok=True)

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
                all_embeddings.append(last_token_hidden_states.cpu().float()) 
                
        return torch.cat(all_embeddings, dim=0)

    def encode_baseline(self, precomputed_raw_embs: torch.Tensor) -> np.ndarray:
        return precomputed_raw_embs.cpu().numpy()

    def encode_filtered(self, precomputed_raw_embs: torch.Tensor, window: tuple = None) -> np.ndarray:
        """
        Applies linear projection post-hoc based on a custom window (start_idx, end_idx).
        """
        raw_embs = precomputed_raw_embs.to(self.device)
        
        if self.vh_matrix is not None:
            if window is not None:
                start_idx, end_idx = window
                vh_sub = self.vh_matrix[start_idx:end_idx, :]
                filtered_embs = torch.matmul(raw_embs, vh_sub.T)
                return filtered_embs.cpu().numpy()
            else:
                return raw_embs.cpu().numpy()
        else:
            print("Warning: No Vh matrix loaded. Returning baseline embeddings.")
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Extracting lm_head.weight and moving to CPU as float32...")
    W = model.lm_head.weight.detach().cpu().float()
    
    print("Computing Full SVD on the raw unembedding matrix...")
    start = time.time()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    print(f"SVD Computed in {time.time()-start:.2f} seconds.")
    
    os.makedirs("models", exist_ok=True)
    torch.save(Vh, pt_path)
    print(f"Saved Vh matrix to {pt_path}")

if __name__ == "__main__":
    main()
''')

with open('src/evaluate_ablation_rag.py', 'w') as f: f.write(r'''
import os
import json
import random
from qwen_ablation_filter import QwenEmbedFilterPipeline
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
            if len(neg_docs) < 10000:
                neg_docs.append(c['text'])
                neg_doc_ids.append(cid)
                
        if len(pos_docs) == len(pos_corpus_ids) and len(neg_docs) >= 10000:
            break
            
    rag_docs = list(pos_docs.values()) + neg_docs
    rag_doc_ids = list(pos_docs.keys()) + neg_doc_ids
    print(f"Corpus streaming complete. Retained {len(rag_docs)} documents.")
    
    rag_queries = queries_ds.filter(lambda x: x['_id'] in sampled_q_ids)['text']
    rag_query_ids = queries_ds.filter(lambda x: x['_id'] in sampled_q_ids)['_id']
    
    print(f"Precomputing RAG embeddings for {len(rag_queries)} queries and {len(rag_docs)} documents...")
    pre_rag_docs = pipeline.get_raw_embeddings(rag_docs)
    pre_rag_queries = pipeline.get_raw_embeddings(rag_queries)
    
    os.makedirs("runs_ablation", exist_ok=True)
    Qrels(qrels_dict).save("runs_ablation/qrels.json")
    
    from scipy.stats import spearmanr
    from sklearn.metrics.pairwise import cosine_similarity
    from ranx import Run, evaluate
    
    def eval_sts_custom(window=None):
        if window is None:
            embs1 = pipeline.encode_baseline(pre_sts1)
            embs2 = pipeline.encode_baseline(pre_sts2)
        else:
            embs1 = pipeline.encode_filtered(pre_sts1, window=window)
            embs2 = pipeline.encode_filtered(pre_sts2, window=window)
            
        sims = []
        for e1, e2 in zip(embs1, embs2):
            sim = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0][0]
            sims.append(sim)
            
        if "label" in sts_ds.column_names:
            labels = sts_ds["label"][:limit]
        elif "score" in sts_ds.column_names:
            labels = sts_ds["score"][:limit]
        corr, _ = spearmanr(sims, labels)
        return corr
        
    def eval_rag_custom(run_name, window=None):
        if window is None:
            doc_embs = pipeline.encode_baseline(pre_rag_docs)
            query_embs = pipeline.encode_baseline(pre_rag_queries)
        else:
            doc_embs = pipeline.encode_filtered(pre_rag_docs, window=window)
            query_embs = pipeline.encode_filtered(pre_rag_queries, window=window)
            
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
        run.save(f"runs_ablation/{run_name}.json")
        
        qrels = Qrels(qrels_dict)
        return evaluate(qrels, run, ["ndcg@10", "recall@100"])

    results = []

    ablation_windows = [
        ("Baseline (1536D)", None),
        ("Head 0:256", (0, 256)),
        ("Head 0:512", (0, 512)),
        ("Head 0:640", (0, 640)),
        ("Indonesian Retention 0:768", (0, 768)),
        ("Head 0:896", (0, 896)),
        ("Middle 128:896", (128, 896)),
        ("Middle 256:1024", (256, 1024)),
        ("English Middle 384:1152", (384, 1152)),
        ("Tail 768:1536", (768, 1536)),
    ]

    for config_name, window in ablation_windows:
        print(f"Evaluating {config_name}...")
        
        run_name = f"run_{config_name.replace(' ', '_').replace(':', '_').lower()}"
        if window is None:
            run_name = "run_baseline"
            
        sts_res = eval_sts_custom(window=window)
        rag_res = eval_rag_custom(run_name, window=window)
        results.append({"config": config_name, "sts_spearman": sts_res, **rag_res})

    with open("runs_ablation/metrics_ablation.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nResults:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
''')

with open('src/hypothesis_test_ablation.py', 'w') as f: f.write(r'''
import os
import glob
from ranx import Qrels, Run, compare

def main():
    print("Loading qrels and exported runs for ablation study...")
    
    runs_dir = "runs_ablation"
    qrels = Qrels.from_file(os.path.join(runs_dir, "qrels.json"))
    
    run_files = sorted(glob.glob(os.path.join(runs_dir, "run_*.json")))
    
    runs = []
    # Make sure baseline is first
    baseline_path = os.path.join(runs_dir, "run_baseline.json")
    if baseline_path in run_files:
        run_files.remove(baseline_path)
        run_files.insert(0, baseline_path)
        
    for run_path in run_files:
        filename = os.path.basename(run_path)
        run = Run.from_file(run_path)
        run.name = filename.replace(".json", "")
        runs.append(run)
            
    if len(runs) < 2:
        print("Not enough runs found for statistical comparison!")
        return
        
    print(f"Loaded {len(runs)} runs. Performing Paired Student's t-test with alpha=0.01...")
    
    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=["ndcg@10", "recall@100"],
        stat_test="student",
        max_p=0.01
    )
    
    print("\n" + "="*50)
    print("      ABLATION STATISTICAL SIGNIFICANCE REPORT")
    print("="*50)
    print(report)

if __name__ == "__main__":
    main()
''')

print("=== 1. Extract Vh Matrix ===")
run_cmd("python -u src/extract_vh_matrix.py")

print("=== 2. Running Ablation Evaluation ===")
run_cmd("python -u src/evaluate_ablation_rag.py")

print("=== 3. Hypothesis Test ===")
run_cmd("python -u src/hypothesis_test_ablation.py > runs_ablation/significance_ablation.txt")
run_cmd("cat runs_ablation/significance_ablation.txt")

run_cmd("cat runs_ablation/metrics_ablation.json")
print("\nALL DONE.")
