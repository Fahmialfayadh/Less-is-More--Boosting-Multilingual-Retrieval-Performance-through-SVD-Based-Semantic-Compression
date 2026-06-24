import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from embed_filter_model import EmbedFilterPipeline
from evaluate_sts import evaluate_sts
from evaluate_rag import evaluate_rag
import mlflow

def main():
    mlflow.set_tracking_uri("file:///content/mlruns" if os.path.exists("/content") else "file:../logs/mlruns")
    mlflow.set_experiment("embedfilter-ablation")
    
    matrix_path = "/content/v_noise.pt" if os.path.exists("/content") else "../models/v_noise.pt"
    pipeline = EmbedFilterPipeline(v_noise_path=matrix_path)
    
    k_values = [0, 1, 5, 10, 20]
    tau_values = [128, 256, 512, 768]
    
    results = []
    
    print("Precomputing raw embeddings (this takes a moment, but speeds up the loop)...")
    # Precompute STS
    from datasets import load_dataset
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
    
    # Precompute RAG (MTEB MIRACL)
    print("Loading MTEB MIRACL for RAG...")
    qrels_ds = load_dataset('mteb/MIRACLRetrieval', 'id-qrels', split='dev')
    queries_ds = load_dataset('mteb/MIRACLRetrieval', 'id-queries', split='dev')
    corpus_ds = load_dataset('mteb/MIRACLRetrieval', 'id-corpus', split='dev')
    
    import random
    random.seed(42)
    # 1. Sample queries (up to 500 queries)
    all_q_ids = list(set([r['query-id'] for r in qrels_ds]))
    sampled_q_ids = random.sample(all_q_ids, min(500, len(all_q_ids)))
    
    # 2. Extract positive corpus ids and build qrels
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
            
    # 3. Sample corpus to include all positives + random negatives (20,000 total)
    print("Sampling corpus...")
    all_corpus_ids = [c['_id'] for c in corpus_ds]
    negative_candidates = list(set(all_corpus_ids) - pos_corpus_ids)
    sampled_negatives = random.sample(negative_candidates, min(20000, len(negative_candidates)))
    sampled_corpus_ids = list(pos_corpus_ids) + sampled_negatives
    
    rag_queries = [q['text'] for q in queries_ds if q['_id'] in sampled_q_ids]
    rag_query_ids = [q['_id'] for q in queries_ds if q['_id'] in sampled_q_ids]
    
    sampled_corpus_ids_set = set(sampled_corpus_ids)
    rag_docs = []
    rag_doc_ids = []
    for c in corpus_ds:
        if c['_id'] in sampled_corpus_ids_set:
            rag_docs.append(c['text'])
            rag_doc_ids.append(c['_id'])
    
    print(f"Precomputing RAG embeddings for {len(rag_queries)} queries and {len(rag_docs)} documents...")
    pre_rag_docs = pipeline.get_raw_embeddings(rag_docs)
    pre_rag_queries = pipeline.get_raw_embeddings(rag_queries)
    
    # Save Qrels for Hypothesis Testing
    from ranx import Qrels
    os.makedirs("/content/runs", exist_ok=True)
    Qrels(qrels_dict).save("/content/runs/qrels.json")
    
    print("Starting Ablation Study (k vs tau)...")
    
    for k, tau in itertools.product(k_values, tau_values):
        if k == 0 and tau != 768:
            continue
            
        tau_val = None if tau == 768 else tau
        
        with mlflow.start_run():
            mlflow.log_params({"k_noise": k, "tau_compression": tau})
            print(f"Evaluating k={k}, tau={tau}...")
            
            # STS Evaluation
            sts_score = evaluate_sts(pipeline, k=k, tau=tau_val, split="dev", pre_embs1=pre_sts1, pre_embs2=pre_sts2)
            
            # RAG Evaluation
            run_name = f"run_k{k}_tau{tau_val}"
            rag_metrics = evaluate_rag(
                pipeline, k_noise=k, tau=tau_val, 
                pre_docs=pre_rag_docs, pre_queries=pre_rag_queries,
                doc_ids=rag_doc_ids, query_ids=rag_query_ids, qrels_dict=qrels_dict,
                run_save_path=f"/content/runs/{run_name}.json"
            )
            
            metrics = {
                "sts_spearman": sts_score,
                **rag_metrics
            }
            mlflow.log_metrics(metrics)
            
            results.append({
                "k": k,
                "tau": tau,
                **metrics
            })
            
    # Visualize Results
    df = pd.DataFrame(results)
    
    os.makedirs("/content/models" if os.path.exists("/content") else "../models", exist_ok=True)
    
    # 1. Plot STS across k for different taus
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df[df['k'] > 0], x='k', y='sts_spearman', hue='tau', marker='o')
    # Add baseline line
    baseline_sts = df[df['k'] == 0]['sts_spearman'].values[0]
    plt.axhline(baseline_sts, color='red', linestyle='--', label='Baseline (k=0, tau=768)')
    plt.title("Isometry Test: STS Spearman Correlation vs Edge Spectrum Removed (k)")
    plt.legend()
    plt.grid(True)
    plt.savefig("/content/models/ablation_sts.png" if os.path.exists("/content") else "../models/ablation_sts.png")
    
    # 2. Plot NDCG_10
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df[df['k'] > 0], x='k', y='NDCG_10', hue='tau', marker='o')
    baseline_ndcg = df[df['k'] == 0]['NDCG_10'].values[0]
    plt.axhline(baseline_ndcg, color='red', linestyle='--', label='Baseline (k=0, tau=768)')
    plt.title("RAG Quality: NDCG_10 vs Edge Spectrum Removed (k)")
    plt.legend()
    plt.grid(True)
    plt.savefig("/content/models/ablation_rag.png" if os.path.exists("/content") else "../models/ablation_rag.png")
    
    import json
    with open("/content/metrics.json" if os.path.exists("/content") else "../logs/metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Compressing MLruns and evaluation metrics...")
    os.system("tar -czf /content/mlruns_eval.tar.gz -C /content mlruns metrics.json runs")
    print("Metrics generation and packaging complete.")
    print("Ablation Study Complete. Plots saved.")

if __name__ == "__main__":
    main()
