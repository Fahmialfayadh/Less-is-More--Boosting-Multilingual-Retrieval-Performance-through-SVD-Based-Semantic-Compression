import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import subprocess
from datasets import load_dataset
from sklearn.decomposition import TruncatedSVD

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def get_unigram_frequencies(dataset_texts, tokenizer, vocab_size):
    token_counts = np.zeros(vocab_size, dtype=np.float64)
    total_tokens = 0
    
    print(f"Tokenizing {len(dataset_texts)} documents to build frequency distribution...")
    batch_size = 1000
    for i in range(0, len(dataset_texts), batch_size):
        batch = dataset_texts[i:i+batch_size]
        batch = [t for t in batch if isinstance(t, str) and t.strip()]
        if not batch:
            continue
        inputs = tokenizer(batch, add_special_tokens=False)
        for ids in inputs["input_ids"]:
            for tid in ids:
                if tid < vocab_size:
                    token_counts[tid] += 1
                    total_tokens += 1
                
    p_w = token_counts / max(total_tokens, 1)
    print(f"Total tokens processed: {total_tokens}")
    print(f"Non-zero probability tokens: {np.count_nonzero(p_w)}")
    return p_w

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    svd_components = 10
    
    print(f"Loading tokenizer and model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # We only need the unembedding matrix, we don't even need the full model for this script.
    # However, for Qwen, the embedding is tied or separate.
    # We can just load the embeddings from safetensors or load the model.
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary Size: {vocab_size}")
    
    # Extract Unembedding Matrix W (lm_head)
    # Move to CPU for TruncatedSVD
    print("Extracting lm_head.weight and moving to CPU...")
    W = model.lm_head.weight.detach().cpu().float() # Shape: (Vocab, Hidden)
    # Just in case some vocabulary are not in lm_head, clip vocab_size
    vocab_size = min(vocab_size, W.shape[0])
    
    print("Computing TruncatedSVD on the raw unembedding matrix...")
    svd_components_search = 200
    svd = TruncatedSVD(n_components=svd_components_search, n_iter=7, random_state=SEED)
    W_numpy = W[:vocab_size, :].numpy()
    U_Sigma = svd.fit_transform(W_numpy)
    V_T = torch.tensor(svd.components_) # Shape: (200, Hidden)
    
    print("=== Processing Indonesian Edge Spectrum ===")
    print("Loading Indonesian Corpus: ZakyF/PRDECT-ID...")
    id_dataset = load_dataset("ZakyF/PRDECT-ID", split="train")
    id_texts = id_dataset['Customer Review']
    # Subsample to speed up if it's too large, say 200k
    id_texts = id_texts[:min(200000, len(id_texts))]
    
    p_w_id = get_unigram_frequencies(id_texts, tokenizer, vocab_size)
    
    print("Calculating Pearson correlation to identify ID Edge Spectrum...")
    log_p_w_id = np.log(p_w_id + 1e-9)
    correlations_id = []
    for i in range(svd_components_search):
        proj = U_Sigma[:, i]
        corr = np.corrcoef(proj, log_p_w_id)[0, 1]
        correlations_id.append(abs(corr))
        
    top_k_indices_id = np.argsort(correlations_id)[::-1][:svd_components].copy()
    V_noise_id = V_T[top_k_indices_id].T # Shape: (Hidden_Dim, svd_components)
    
    os.makedirs("models", exist_ok=True)
    pt_path_id = "models/v_noise_id.pt"
    torch.save(V_noise_id, pt_path_id)
    print(f"Saved V_noise_id to {pt_path_id}")


    print("=== Processing English Edge Spectrum ===")
    print("Loading English Corpus: stanfordnlp/imdb...")
    en_dataset = load_dataset("stanfordnlp/imdb", split="train")
    en_texts = en_dataset['text']
    # Subsample to match scale
    en_texts = en_texts[:min(200000, len(en_texts))]
    
    p_w_en = get_unigram_frequencies(en_texts, tokenizer, vocab_size)
    
    print("Calculating Pearson correlation to identify EN Edge Spectrum...")
    log_p_w_en = np.log(p_w_en + 1e-9)
    correlations_en = []
    for i in range(svd_components_search):
        proj = U_Sigma[:, i]
        corr = np.corrcoef(proj, log_p_w_en)[0, 1]
        correlations_en.append(abs(corr))
        
    top_k_indices_en = np.argsort(correlations_en)[::-1][:svd_components].copy()
    V_noise_en = V_T[top_k_indices_en].T # Shape: (Hidden_Dim, svd_components)
    
    pt_path_en = "models/v_noise_en.pt"
    torch.save(V_noise_en, pt_path_en)
    print(f"Saved V_noise_en to {pt_path_en}")

if __name__ == "__main__":
    main()
