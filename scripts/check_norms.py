import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16, device_map="cpu")
    W = model.lm_head.weight.detach().float()
    norms = torch.norm(W, dim=1)
    
    print(f"Mean norm: {norms.mean().item():.2f}")
    print(f"Max norm: {norms.max().item():.2f}")
    
    special_tokens = ["<|endoftext|>", "\n", " they", " about", " so", "ä»ĸä»¬", "s", "a"]
    for t in special_tokens:
        ids = tokenizer.encode(t)
        if len(ids) > 0:
            norm_val = norms[ids[0]].item()
            print(f"Norm of '{t}': {norm_val:.2f}")

if __name__ == "__main__":
    main()
