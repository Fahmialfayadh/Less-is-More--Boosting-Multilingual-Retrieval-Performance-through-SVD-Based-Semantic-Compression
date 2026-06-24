import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import mlflow
import os
import subprocess

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate data adhering to pandas-pro and ml-pipeline best practices."""
    assert 'word' in df.columns, "DataFrame must contain 'word' column"
    # Drop completely empty rows using vectorized operations
    df = df.dropna(subset=['word']).copy()
    df['word'] = df['word'].str.strip()
    df = df[df['word'] != '']
    assert df.isna().sum().sum() == 0, "Nulls found after cleaning"
    return df

def main():
    model_name = "intfloat/multilingual-e5-base"
    svd_components = 10
    
    mlflow.set_tracking_uri("file:///content/mlruns")
    mlflow.set_experiment("embedfilter-extraction")
    with mlflow.start_run():
        mlflow.log_params({
            "model_name": model_name,
            "svd_components": svd_components,
            "seed": SEED,
            "method": "unsupervised_frequency_weighted_svd"
        })
        
        print(f"Loading tokenizer and model {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print("Loading corpus for Unsupervised Unigram Frequency calculation...")
        from datasets import load_dataset
        # We use a real corpus to get the unigram distribution
        dataset = load_dataset("mteb/NusaX-senti", "ind", split="train")
        
        vocab_size = tokenizer.vocab_size
        print(f"Vocabulary Size: {vocab_size}")
        
        # 1. Compute Unigram Frequencies p(w)
        token_counts = np.zeros(vocab_size, dtype=np.float64)
        total_tokens = 0
        
        print("Tokenizing corpus to build frequency distribution...")
        for text in dataset['text']:
            if not isinstance(text, str) or not text.strip():
                continue
            # Tokenize and count
            inputs = tokenizer(text, add_special_tokens=False)
            ids = inputs["input_ids"]
            for tid in ids:
                if tid < vocab_size:
                    token_counts[tid] += 1
                    total_tokens += 1
                    
        # Probabilities
        p_w = token_counts / max(total_tokens, 1)
        print(f"Total tokens processed: {total_tokens}")
        print(f"Non-zero probability tokens: {np.count_nonzero(p_w)}")
        
        # 2. Extract Unembedding Matrix W
        W = model.get_input_embeddings().weight.detach() # Shape: (Vocab, Hidden)
        hidden_dim = W.shape[1]
        print(f"Unembedding Matrix Shape: {W.shape}")
        
        # 3. Frequency-Weighted Unembedding: W_weighted = W * p(w)
        p_w_tensor = torch.tensor(p_w, dtype=W.dtype, device=W.device).unsqueeze(1) # (Vocab, 1)
        W_weighted = W * p_w_tensor # Broadcast multiply
        
        # 4. Compute SVD on W_weighted to find Edge Spectrum (V_noise)
        print(f"Running SVD on weighted unembedding matrix {W_weighted.shape}...")
        mean_W = W_weighted.mean(dim=0, keepdim=True)
        centered_W = W_weighted - mean_W
        
        # Extract top k singular vectors
        U, S, V = torch.pca_lowrank(centered_W, q=svd_components, center=False, niter=3)
        V_noise = V  # Shape: (Hidden_Dim, svd_components)
        
        # 5. Save Edge Spectrum (We save V_noise instead of P, so we can do ablation later)
        os.makedirs("/content/models", exist_ok=True)
        pt_path = "/content/models/v_noise.pt"
        torch.save(V_noise, pt_path)
        
        mlflow.log_artifact(pt_path)
        mlflow.log_metric("vocab_size", vocab_size)
        mlflow.log_metric("hidden_dim", hidden_dim)
        mlflow.log_metric("total_corpus_tokens", total_tokens)
        
        print(f"V_noise (Edge Spectrum) of shape {V_noise.shape} saved to {pt_path}.")
        
    print("Packaging mlruns...")
    subprocess.run("tar -czf /content/mlruns.tar.gz -C /content mlruns", shell=True, check=True)
    print("mlruns packaged successfully.")

if __name__ == "__main__":
    main()
