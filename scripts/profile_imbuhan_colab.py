import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time

def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16, device_map="auto")
    
    W = model.lm_head.weight.detach().float()
    
    print("Computing SVD to get Vh...")
    start_time = time.time()
    # Use torch.linalg.svd which is highly optimized in Colab and doesn't OOM
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    print(f"SVD computed in {time.time() - start_time:.2f} seconds.")
    
    print("Projecting W...")
    projected_W = torch.matmul(W, Vh.T) 
    
    imbuhans = [
        "nya", "lah", "kan", "pun", "kah", "ku", "mu",
        "nya", " kan", " lah",
        " me", " di", " ter", " pe", " ber", " se", " ke",
        " meng", " peng", " meny", " peny", " memper",
        " mem", " pem", " bel"
    ]
    
    # Map imbuhans to token ids
    target_ids = []
    for imbuhan in imbuhans:
        ids = tokenizer.encode(imbuhan)
        if len(ids) == 1:
            target_ids.append(ids[0])
            
    # Remove duplicates
    target_ids = list(set(target_ids))
    print(f"Tracking {len(target_ids)} unique affix tokens.")
    
    # 1536 dimensions into 24 bins of 64
    num_bins = 24
    bin_size = 64
    
    bin_densities = np.zeros(num_bins)
    
    for idx in target_ids:
        token_vec = projected_W[idx] # Shape: [1536]
        
        # Calculate L2 norm in each bin
        for b in range(num_bins):
            start = b * bin_size
            end = start + bin_size
            bin_norm = torch.norm(token_vec[start:end]).item()
            bin_densities[b] += bin_norm
            
    print("\n" + "="*50)
    print("IMBUHAN ENERGY DENSITY (per 64 dimensions)")
    print("="*50)
    
    for b in range(num_bins):
        start = b * bin_size
        end = start + bin_size
        density = bin_densities[b]
        # Create a simple bar chart
        bar = "#" * int((density / np.max(bin_densities)) * 30)
        print(f"Bin {b:02d} [{start:>4}:{end:>4}]: {density:>8.2f} | {bar}")
        
    print("\n" + "="*50)
    print("FINDING OPTIMAL 768-DIM WINDOW (12 Bins)")
    print("="*50)
    
    max_energy = 0
    best_start_bin = 0
    window_size = 12 # 12 bins = 768 dims
    
    for b in range(num_bins - window_size + 1):
        window_energy = np.sum(bin_densities[b:b+window_size])
        start_dim = b * bin_size
        end_dim = start_dim + (window_size * bin_size)
        print(f"Window [{start_dim:>4}:{end_dim:>4}]: Energy = {window_energy:>8.2f}")
        
        if window_energy > max_energy:
            max_energy = window_energy
            best_start_bin = b
            
    best_start_dim = best_start_bin * bin_size
    best_end_dim = best_start_dim + 768
    print(f"\n=> OPTIMAL SWEET SPOT: {best_start_dim}:{best_end_dim} (Energy: {max_energy:.2f})")

if __name__ == "__main__":
    main()
