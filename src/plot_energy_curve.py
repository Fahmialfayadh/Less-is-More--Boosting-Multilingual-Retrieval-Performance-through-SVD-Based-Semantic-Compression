import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    model_name = "Qwen/Qwen2.5-1.5B"
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cpu")
    
    W = model.lm_head.weight.detach().float() # Shape: (Vocab, 1536)
    
    print("Computing Full SVD on lm_head.weight...")
    # W = U * Sigma * Vh
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    print("Projecting vocabulary onto SVD components...")
    # Shape: (Vocab, 1536)
    projected_W = torch.matmul(W, Vh.T)
    
    # Define English Stopwords and Indonesian Affixes
    en_stopwords = [
        " the", " and", " of", " to", " a", " in", " that", " is", " was", " for",
        " on", " with", " as", " at", " by", " an", " be", " this", " are", " from",
        "it", "he", "she", "they", "we", "you", "but", "or", "not", "your"
    ]
    
    id_affixes = [
        "nya", "lah", "kan", "pun", "kah", "ku", "mu",
        "nya", " kan", " lah",
        " me", " di", " ter", " pe", " ber", " se", " ke",
        " meng", " peng", " meny", " peny", " memper",
        " mem", " pem", " bel"
    ]
    
    def get_token_ids(tokens):
        ids = []
        for t in tokens:
            encoded = tokenizer.encode(t)
            # Match only single token representations
            if len(encoded) == 1:
                ids.append(encoded[0])
        return list(set(ids))
    
    en_ids = get_token_ids(en_stopwords)
    id_ids = get_token_ids(id_affixes)
    
    print(f"Mapped {len(en_ids)} English stopwords and {len(id_ids)} Indonesian affixes to single tokens.")
    
    # Compute absolute coefficients (representing energy)
    # Shape: (Num_Tokens, 1536)
    en_energy = torch.abs(projected_W[en_ids, :]).numpy()
    id_energy = torch.abs(projected_W[id_ids, :]).numpy()
    
    # Average across tokens
    # Shape: (1536,)
    en_mean_energy = en_energy.mean(axis=0)
    id_mean_energy = id_energy.mean(axis=0)
    
    # Apply moving average to smooth the curves
    def moving_average(a, n=64):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    window_size = 64
    en_smooth = moving_average(en_mean_energy, window_size)
    id_smooth = moving_average(id_mean_energy, window_size)
    
    # Adjust X axis to match the moving average output length
    x_axis = np.arange(len(en_smooth)) + (window_size // 2)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot curves
    plt.plot(x_axis, id_smooth, label="Indonesian Affixes (e.g., -nya, meng-, ter-)", color="crimson", linewidth=2.5)
    plt.plot(x_axis, en_smooth, label="English Stopwords (e.g., the, and, of)", color="dodgerblue", linewidth=2.5)
    
    # Add vertical bands for Head, Middle, and Tail Spectrum
    plt.axvspan(0, 384, color="lightcoral", alpha=0.15, label="Head Spectrum (0-384)")
    plt.axvspan(384, 1152, color="khaki", alpha=0.15, label="Middle Spectrum (384-1152)")
    plt.axvspan(1152, 1536, color="lightgreen", alpha=0.15, label="Tail Spectrum (1152-1536)")
    
    plt.title("SVD Energy Distribution: Indonesian Affixes vs. English Stopwords", fontsize=14, fontweight="bold")
    plt.xlabel("SVD Dimension Index (0 to 1536)", fontsize=12)
    plt.ylabel(f"Average Energy (L1-Norm, Smoothed over {window_size} dims)", fontsize=12)
    plt.xlim(0, 1536)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10)
    
    # Save visualization
    os.makedirs("data", exist_ok=True)
    save_path = "data/energy_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Energy curve visualization saved to: {save_path}")

if __name__ == "__main__":
    main()
