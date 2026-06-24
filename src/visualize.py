import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from embed_filter_model import EmbedFilterPipeline

def main():
    from datasets import load_dataset
    print("Loading IndoNLU Shopee Reviews for visualization...")
    dataset = load_dataset("mteb/NusaX-senti", "ind", split="train[:500]")
    raw_texts = dataset['text']
    raw_labels = dataset['label']
    
    texts, labels = [], []
    for t, l in zip(raw_texts, raw_labels):
        if isinstance(t, str) and t.strip():
            texts.append(t)
            labels.append(l)
    
    # NusaX-senti mapping: 0=negative, 1=neutral, 2=positive
    color_map = {0: 'red', 1: 'gray', 2: 'green'}
    colors = [color_map[l] for l in labels]
    
    import os
    matrix_path = "/content/projection_matrix.pt" if os.path.exists("/content/projection_matrix.pt") else "../models/projection_matrix.pt"
    pipeline = EmbedFilterPipeline(
        model_name="intfloat/multilingual-e5-base",
        projection_matrix_path=matrix_path
    )
    
    print("Extracting embeddings...")
    baseline_embs = pipeline.encode_baseline(texts)
    filtered_embs = pipeline.encode_filtered(texts)
    
    # Use PCA to reduce to 2D
    pca_base = PCA(n_components=2)
    base_2d = pca_base.fit_transform(baseline_embs)
    
    pca_filt = PCA(n_components=2)
    filt_2d = pca_filt.fit_transform(filtered_embs)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline Plot
    ax1.scatter(base_2d[:, 0], base_2d[:, 1], c=colors, s=100)
    ax1.set_title("Baseline Embeddings (Clustered)")
    ax1.set_xlabel("PCA Component 1")
    ax1.set_ylabel("PCA Component 2")
    
    # Filtered Plot
    ax2.scatter(filt_2d[:, 0], filt_2d[:, 1], c=colors, s=100)
    ax2.set_title("EmbedFilter Embeddings (Clustered)")
    ax2.set_xlabel("PCA Component 1")
    ax2.set_ylabel("PCA Component 2")
    
    plt.suptitle("Semantic Clustering: Positive (Green) vs Negative (Red) Reviews")
    save_path = "/content/baseline_vs_filtered.png" if os.path.exists("/content") else "../models/baseline_vs_filtered.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    
    import subprocess
    print("Packaging mlruns_eval...")
    subprocess.run("tar -czf /content/mlruns_eval.tar.gz -C /content mlruns", shell=True)

if __name__ == "__main__":
    main()
