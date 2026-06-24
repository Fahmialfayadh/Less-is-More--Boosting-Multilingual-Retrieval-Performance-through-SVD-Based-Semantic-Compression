import numpy as np
import scipy.stats
from datasets import load_dataset
from embed_filter_model import EmbedFilterPipeline
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_sts(pipeline: EmbedFilterPipeline, k: int = 0, tau: int = None, split: str = "dev", pre_embs1=None, pre_embs2=None) -> float:
    """
    Evaluates the pipeline on the STS-B dataset using Spearman Rank Correlation.
    Returns the correlation score (0 to 1).
    """
    try:
        # Using LazarusNLP/stsb_mt_id, a standard Indonesian translation of STS-B
        dataset = load_dataset("LazarusNLP/stsb_mt_id", split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 0.0
        
    if "text_1" in dataset.column_names:
        sentence1 = dataset["text_1"]
        sentence2 = dataset["text_2"]
    elif "text1" in dataset.column_names:
        sentence1 = dataset["text1"]
        sentence2 = dataset["text2"]
    else:
        sentence1 = dataset["sentence1"]
        sentence2 = dataset["sentence2"]
        
    if "correlation" in dataset.column_names:
        human_scores = np.array([float(s) for s in dataset["correlation"]])
    elif "score" in dataset.column_names:
        human_scores = np.array(dataset["score"])
    else:
        human_scores = np.array(dataset["similarity_score"])
    
    limit = min(500, len(sentence1))
    sentence1 = sentence1[:limit]
    sentence2 = sentence2[:limit]
    human_scores = human_scores[:limit]
    
    # Encode
    if k == 0 and tau is None:
        if pre_embs1 is not None and pre_embs2 is not None:
            embs1 = pre_embs1.numpy()
            embs2 = pre_embs2.numpy()
        else:
            embs1 = pipeline.encode_baseline(sentence1)
            embs2 = pipeline.encode_baseline(sentence2)
    else:
        if k == 0 and tau == 768:
            embs1 = pre_embs1.numpy() if pre_embs1 is not None else pipeline.encode_baseline(sentence1)
            embs2 = pre_embs2.numpy() if pre_embs2 is not None else pipeline.encode_baseline(sentence2)
        else:
            if pre_embs1 is not None and pre_embs2 is not None:
                import torch
                combined_embs = torch.cat([pre_embs1, pre_embs2], dim=0)
                filtered_combined = pipeline.encode_filtered(texts=None, k=k, tau=tau, precomputed_raw_embs=combined_embs)
            else:
                combined_texts = sentence1 + sentence2
                filtered_combined = pipeline.encode_filtered(texts=combined_texts, k=k, tau=tau)
                
            embs1 = filtered_combined[:len(sentence1)]
            embs2 = filtered_combined[len(sentence1):]
        
    # Cosine Similarity
    # np.sum(embs1 * embs2, axis=1) works if vectors are normalized. 
    # To be safe, we use pairwise cosine_similarity but only take the diagonal
    sim_scores = []
    for e1, e2 in zip(embs1, embs2):
        score = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))[0][0]
        sim_scores.append(score)
        
    # Spearman Correlation
    correlation, p_value = scipy.stats.spearmanr(sim_scores, human_scores)
    return correlation

if __name__ == "__main__":
    import os
    matrix_path = "/content/v_noise.pt" if os.path.exists("/content") else "../models/v_noise.pt"
    pipeline = EmbedFilterPipeline(v_noise_path=matrix_path)
    
    print("Evaluating Baseline STS...")
    base_corr = evaluate_sts(pipeline, k=0, tau=None)
    print(f"Baseline Spearman Correlation: {base_corr:.4f}")
    
    print("Evaluating Filtered STS (k=10, tau=128)...")
    filt_corr = evaluate_sts(pipeline, k=10, tau=128)
    print(f"Filtered Spearman Correlation: {filt_corr:.4f}")
