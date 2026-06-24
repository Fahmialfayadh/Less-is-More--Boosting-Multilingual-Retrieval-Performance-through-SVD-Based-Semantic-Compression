import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from embed_filter_model import EmbedFilterPipeline
from ranx import Qrels, Run, evaluate

def evaluate_rag(pipeline: EmbedFilterPipeline, k_noise: int = 0, tau: int = None, 
                 pre_docs=None, pre_queries=None, doc_ids=None, query_ids=None, qrels_dict=None,
                 run_save_path: str = None) -> dict:
    if any(x is None for x in [pre_docs, pre_queries, doc_ids, query_ids, qrels_dict]):
        raise ValueError("Must provide all precomputed data and IDs for RAG evaluation.")
        
    if k_noise == 0 and tau == 768:
        doc_embs = pre_docs.numpy()
        query_embs = pre_queries.numpy()
    else:
        combined_embs = torch.cat([pre_docs, pre_queries], dim=0)
        filtered_combined = pipeline.encode_filtered(texts=None, k=k_noise, tau=tau, precomputed_raw_embs=combined_embs)
        doc_embs = filtered_combined[:len(pre_docs)]
        query_embs = filtered_combined[len(pre_docs):]
        
    # Calculate similarities
    sim_matrix = cosine_similarity(query_embs, doc_embs)
    
    # Build Run dict for ranx
    run_dict = {}
    for i, q_id in enumerate(query_ids):
        # We take top 100 to make ranx fast and avoid memory bloat
        k = min(100, len(doc_ids))
        top_k_idx = np.argpartition(sim_matrix[i], -k)[-k:]
        # Sort them descending
        top_k_idx = top_k_idx[np.argsort(sim_matrix[i][top_k_idx])[::-1]]
        
        run_dict[q_id] = {doc_ids[idx]: float(sim_matrix[i][idx]) for idx in top_k_idx}
        
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)
    
    if run_save_path is not None:
        run.save(run_save_path)
    
    # Evaluate with ranx
    results = evaluate(qrels, run, ["ndcg@10", "recall@100"])
    
    # return with sanitized names for MLFlow
    return {
        "NDCG_10": results["ndcg@10"],
        "Recall_100": results["recall@100"]
    }
