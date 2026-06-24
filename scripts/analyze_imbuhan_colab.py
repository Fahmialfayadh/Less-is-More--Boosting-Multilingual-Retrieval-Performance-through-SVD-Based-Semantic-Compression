import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16, device_map="auto")
    
    W = model.lm_head.weight.detach().float() # [Vocab, 1536]
    
    print("Computing SVD to get Vh...")
    # Compute full SVD
    # W = U * S * Vh
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    print("Projecting W...")
    projected_W = torch.matmul(W, Vh.T) 
    
    head_norms = torch.norm(projected_W[:, 0:384], dim=1)
    middle_norms = torch.norm(projected_W[:, 384:1152], dim=1)
    tail_norms = torch.norm(projected_W[:, -384:], dim=1)
    
    total_norms = head_norms + middle_norms + tail_norms
    
    imbuhans = [
        "nya", "lah", "kan", "pun", "kah", "ku", "mu",
        "nya", " kan", " lah",
        " me", " di", " ter", " pe", " ber", " se", " ke",
        " meng", " peng", " meny", " peny", " memper",
        " mem", " pem", " bel"
    ]
    
    print("\n" + "="*80)
    print(f"{'Token (Imbuhan)':<15} | {'Head (%)':<10} | {'Middle (%)':<10} | {'Tail (%)':<10} | Dominant Zone")
    print("-" * 80)
    
    for imbuhan in imbuhans:
        ids = tokenizer.encode(imbuhan)
        if len(ids) == 1:
            idx = ids[0]
            tok_str = tokenizer.convert_ids_to_tokens(idx)
            if type(tok_str) == bytes:
                tok_str = tok_str.decode('utf-8', errors='ignore')
            tok_str = tok_str.replace("Ġ", " ").replace("Ċ", "\\n")
            
            hn = head_norms[idx].item()
            mn = middle_norms[idx].item()
            tn = tail_norms[idx].item()
            tot = hn + mn + tn
            
            hp = (hn / tot) * 100
            mp = (mn / tot) * 100
            tp = (tn / tot) * 100
            
            dominant = "Head"
            if mp > hp and mp > tp: dominant = "Middle"
            if tp > hp and tp > mp: dominant = "Tail"
                
            print(f"'{tok_str}' (<idx {idx}>)".ljust(15) + f" | {hp:>8.1f}% | {mp:>8.1f}% | {tp:>8.1f}% | {dominant}")

if __name__ == "__main__":
    main()
