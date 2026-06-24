import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16, device_map="auto")
    
    W = model.lm_head.weight.detach().float() # [Vocab, 1536]
    Vh = torch.load("models/vh_matrix.pt", map_location=W.device, weights_only=True) # [1536, 1536]
    
    print("Projecting W...")
    # U \Sigma = W @ Vh.T
    projected_W = torch.matmul(W, Vh.T) 
    
    # Calculate norms in the 3 subspaces
    head_norms = torch.norm(projected_W[:, 0:384], dim=1)
    middle_norms = torch.norm(projected_W[:, 384:1152], dim=1)
    tail_norms = torch.norm(projected_W[:, -384:], dim=1)
    
    def print_top_tokens(norms, name, k=200):
        top_indices = torch.argsort(norms, descending=True)[:k]
        tokens = []
        for idx in top_indices:
            idx = idx.item()
            tok_str = tokenizer.convert_ids_to_tokens(idx)
            if type(tok_str) == bytes:
                tok_str = tok_str.decode('utf-8', errors='ignore')
            elif tok_str is None:
                tok_str = "[UNK]"
            # clean for printing
            tok_str = tok_str.replace("Ġ", " ").replace("Ċ", "\\n")
            tokens.append(f"'{tok_str}'")
            
        print(f"\n{'='*50}\n--- Top {k} Tokens in {name} Spectrum ---\n{'='*50}")
        print(", ".join(tokens))

    print_top_tokens(head_norms, "HEAD (Top 25% SVs)")
    print_top_tokens(middle_norms, "MIDDLE (Middle 50% SVs)")
    print_top_tokens(tail_norms, "TAIL (Bottom 25% SVs)")

if __name__ == "__main__":
    main()
