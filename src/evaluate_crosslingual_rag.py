import os
import json
import random
from qwen_embed_filter import QwenEmbedFilterPipeline
from datasets import load_dataset
from ranx import Qrels

def main():
    print("Loading EmbedFilter Pipeline for Qwen2.5-1.5B...")
    
    matrix_id = "models/v_noise_id.pt"
    matrix_en = "models/v_noise_en.pt"
    
    pipeline = QwenEmbedFilterPipeline(
        model_name="Qwen/Qwen2.5-1.5B",
        v_noise_path_id=matrix_id,
        v_noise_path_en=matrix_en
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
    # Sample queries (up to 500 queries)
    all_q_ids = list(set([r['query-id'] for r in qrels_ds]))
    sampled_q_ids = random.sample(all_q_ids, min(500, len(all_q_ids)))
    
    # Extract positive corpus ids and build qrels
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
            if len(neg_docs) < 20000:
                neg_docs.append(c['text'])
                neg_doc_ids.append(cid)
                
        if len(pos_docs) == len(pos_corpus_ids) and len(neg_docs) >= 20000:
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
    
    # Custom evaluation functions replacing the ones that use `pipeline.encode_filtered`
    from scipy.stats import spearmanr
    from sklearn.metrics.pairwise import cosine_similarity
    from ranx import Run, evaluate
    
    def eval_sts_custom(filter_type=None):
        if filter_type is None:
            embs1 = pre_sts1.numpy()
            embs2 = pre_sts2.numpy()
        elif filter_type == "retention_0_768":
            embs1 = pipeline.encode_retention_window(pre_sts1, 0, 768)
            embs2 = pipeline.encode_retention_window(pre_sts2, 0, 768)
        else:
            embs1 = pipeline.encode_filtered(pre_sts1, filter_type=filter_type, k=10)
            embs2 = pipeline.encode_filtered(pre_sts2, filter_type=filter_type, k=10)
            
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
            doc_embs = pre_rag_docs.numpy()
            query_embs = pre_rag_queries.numpy()
        elif filter_type == "retention_0_768":
            doc_embs = pipeline.encode_retention_window(pre_rag_docs, 0, 768)
            query_embs = pipeline.encode_retention_window(pre_rag_queries, 0, 768)
        else:
            doc_embs = pipeline.encode_filtered(pre_rag_docs, filter_type=filter_type, k=10)
            query_embs = pipeline.encode_filtered(pre_rag_queries, filter_type=filter_type, k=10)
            
        print("Computing cosine similarity for RAG...")
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

    # 2. English Filter
    print("Evaluating English Filter...")
    sts_en = eval_sts_custom(filter_type="en")
    rag_en = eval_rag_custom("run_english_filter", filter_type="en")
    results.append({"config": "English-Filter", "sts_spearman": sts_en, **rag_en})
    
    # 3. Indonesian Filter
    print("Evaluating Indonesian Filter...")
    sts_id = eval_sts_custom(filter_type="id")
    rag_id = eval_rag_custom("run_indonesian_filter", filter_type="id")
    results.append({"config": "Indonesian-Filter", "sts_spearman": sts_id, **rag_id})
    
    # 4. Indonesian Retention Window (0:768)
    print("Evaluating Indonesian Retention Window (0:768)...")
    sts_ret = eval_sts_custom(filter_type="retention_0_768")
    rag_ret = eval_rag_custom("run_indonesian_retention_768", filter_type="retention_0_768")
    results.append({"config": "Indonesian-Retention-0-768", "sts_spearman": sts_ret, **rag_ret})
    
    with open("runs/metrics_crosslingual.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nResults:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
