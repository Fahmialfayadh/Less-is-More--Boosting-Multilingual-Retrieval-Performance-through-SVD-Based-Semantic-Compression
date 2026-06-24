import os
import json
import numpy as np
import mlflow
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from embed_filter_model import EmbedFilterPipeline

def main():
    print("Downloading IndoNLU Shopee Reviews dataset...")
    # MTEB/NusaX-senti is the modern Parquet-native version of the dataset
    dataset = load_dataset("mteb/NusaX-senti", "ind", split="train[:500]")
    
    raw_texts = dataset['text']
    raw_labels = dataset['label']
    
    # Safely filter out any None or non-string values that crash the tokenizer
    texts, labels = [], []
    for t, l in zip(raw_texts, raw_labels):
        if isinstance(t, str) and t.strip():
            texts.append(t)
            labels.append(l)
    
    # Mapping in NusaX-senti: 0=negative, 1=neutral, 2=positive
    # According to plan0.md, we must use the literal label words for zero-shot classification
    anchors = [
        "positif", # Positive Anchor (0)
        "negatif", # Negative Anchor (2)
    ]
    
    print("Extracting embeddings for 500 real reviews...")
    matrix_path = "/content/projection_matrix.pt" if os.path.exists("/content/projection_matrix.pt") else "../models/projection_matrix.pt"
    pipeline = EmbedFilterPipeline(
        model_name="intfloat/multilingual-e5-base",
        projection_matrix_path=matrix_path
    )
    
    print("Running Baseline...")
    anchor_embs_baseline = pipeline.encode_baseline(anchors)
    review_embs_baseline = pipeline.encode_baseline(texts)
    
    sim_baseline = cosine_similarity(review_embs_baseline, anchor_embs_baseline)
    
    # Predict based on closest semantic anchor
    correct_base = 0
    total_valid = 0
    
    for i, label in enumerate(labels):
        if label == 1: continue # skip neutral to force binary pos/neg task
        pred_idx = np.argmax(sim_baseline[i])
        # If closest to "positif" (idx 0), predict 2. If closest to "negatif" (idx 1), predict 0.
        pred_label = 2 if pred_idx == 0 else 0
        if pred_label == label:
            correct_base += 1
        total_valid += 1
        
    baseline_acc = correct_base / total_valid if total_valid > 0 else 0
    
    print("Running EmbedFilter...")
    anchor_embs_filt = pipeline.encode_filtered(anchors)
    review_embs_filt = pipeline.encode_filtered(texts)
    
    sim_filt = cosine_similarity(review_embs_filt, anchor_embs_filt)
    
    correct_filt = 0
    for i, label in enumerate(labels):
        if label == 1: continue
        pred_idx = np.argmax(sim_filt[i])
        pred_label = 2 if pred_idx == 0 else 0
        if pred_label == label:
            correct_filt += 1
            
    filtered_acc = correct_filt / total_valid if total_valid > 0 else 0
    
    metrics = {
        "baseline_zero_shot_acc": baseline_acc,
        "filtered_zero_shot_acc": filtered_acc,
        "absolute_improvement": filtered_acc - baseline_acc,
        "dataset_size": total_valid
    }
    
    print("\n--- ZERO-SHOT CLASSIFICATION RESULTS ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    with open("/content/metrics.json" if os.path.exists("/content") else "../logs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # MLflow tracking
    mlruns_path = "file:/content/mlruns" if os.path.exists("/content") else "file:../logs/mlruns"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment("embedfilter-real-data-eval")
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        print("Metrics successfully logged to MLflow.")

if __name__ == "__main__":
    main()
