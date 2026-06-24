import json
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

def analyze_7b():
    model_id = "Qwen/Qwen2.5-7B"
    print(f"Fetching {model_id} safetensors index...")
    
    # 1. Download index to find where lm_head is
    index_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
    with open(index_path, 'r') as f:
        index = json.load(f)
        
    lm_head_file = index['weight_map'].get('lm_head.weight')
    if not lm_head_file:
        print("lm_head.weight not found in index. Falling back to default...")
        # Sometimes it's just model.safetensors if not sharded, but 7B is definitely sharded
        lm_head_file = "model-00004-of-00004.safetensors" # Guessing, but the index should have it
        
    print(f"lm_head.weight is located in: {lm_head_file}. Downloading only this shard...")
    weight_path = hf_hub_download(repo_id=model_id, filename=lm_head_file)
    
    print("Loading safetensors shard...")
    state_dict = load_file(weight_path)
    W = state_dict['lm_head.weight'].float() # (vocab_size, hidden_size) -> (151936, 3584)
    print(f"Loaded lm_head: {W.shape}")
    
    # 2. SVD
    print("Computing SVD for 7B model... (This might take 30-60 seconds on CPU)")
    # Since we want projections of tokens, we can just do SVD on W.
    # W = U * S * V^T. We want projections = W * V = U * S
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # Projections
    projections = U * S
    
    # 3. Calculate Norms for Head, Middle, Tail
    d = S.shape[0] # 3584
    head_dim = d // 4 # 896
    mid_dim = head_dim + (d // 2) # 896 + 1792 = 2688
    
    print(f"Spectrum Ranges: Head(0-{head_dim}), Middle({head_dim}-{mid_dim}), Tail({mid_dim}-{d})")
    
    # 4. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Target tokens
    imbuhan_targets = [
        'nya', 'lah', 'kan', 'pun', 'kah', 'ku', 'mu',
        ' di', ' ter', ' ber', ' meng', ' mem', ' pe', ' ke', ' se'
    ]
    
    print("\nL2-Norm Distribution for Qwen2.5-7B:")
    print(f"{'Token':<10} | {'Head (25%)':<12} | {'Mid (50%)':<12} | {'Tail (25%)':<12}")
    print("-" * 55)
    
    for token in imbuhan_targets:
        token_id = tokenizer.encode(token, add_special_tokens=False)
        if not token_id:
            continue
        token_id = token_id[0]
        
        proj = projections[token_id]
        
        head_energy = torch.sum(proj[:head_dim]**2).item()
        mid_energy = torch.sum(proj[head_dim:mid_dim]**2).item()
        tail_energy = torch.sum(proj[mid_dim:]**2).item()
        
        total_energy = head_energy + mid_energy + tail_energy
        
        head_pct = (head_energy / total_energy) * 100
        mid_pct = (mid_energy / total_energy) * 100
        tail_pct = (tail_energy / total_energy) * 100
        
        print(f"'{token}'{repr(token)[len(token)+2:]:>10} | {head_pct:>5.1f}%      | {mid_pct:>5.1f}%      | {tail_pct:>5.1f}%")

if __name__ == "__main__":
    analyze_7b()
