import torch
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import safetensors.torch
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.1-70B"

print("Llama-3.1-70B SVD Energy Curve Generator")

# Step 1: Load unembedding matrix
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device.upper()}")

shard_path = hf_hub_download(repo_id=model_name, filename="model-00030-of-00030.safetensors")

print("Loading W matrix to GPU...")
tensors = safetensors.torch.load_file(shard_path, device=device)
W_tensor = tensors["lm_head.weight"].float()

print("Computing covariance matrix C = W^T @ W...")
C = W_tensor.T @ W_tensor

print("Computing eigendecomposition...")
eigvals, eigvecs = torch.linalg.eigh(C)

# Sort in descending order
eigvals = torch.flip(eigvals, dims=[0])
eigvecs = torch.flip(eigvecs, dims=[1])

# Transfer basis matrix to CPU
Vh_t = eigvecs.T.cpu().numpy()

# Clear GPU memory
del tensors, W_tensor, C, eigvals, eigvecs
torch.cuda.empty_cache()
gc.collect()


# Step 2: Extract Token Vectors
print("Extracting token vectors...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Affixes dataset
table_affixes = ['nya', 'lah', 'kan', 'pun', ' meng', ' ber', ' ter'] 
# English stopwords dataset
stopwords = [' the', ' and', ' of', ' to', ' a', ' in', ' that', ' is', ' it', ' for', ' with', ' on']

def get_vectors(word_list):
    vecs = []
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        W_slice = f.get_slice("lm_head.weight")
        for w in word_list:
            tokens = tokenizer.encode(w, add_special_tokens=False)
            if len(tokens) == 0: continue
            tid = tokens[0]
            vec = W_slice[tid:tid+1].float().numpy().flatten()
            vecs.append(vec)
    return np.array(vecs)

vecs_affix = get_vectors(table_affixes)
vecs_stop = get_vectors(stopwords)

print("Projecting vectors to SVD dimensions...")
proj_affix = vecs_affix @ Vh_t.T
proj_stop = vecs_stop @ Vh_t.T

# Step 3: Generate Markdown Table
print("\nGenerating Markdown Table...")

results_markdown = []

for i, affix in enumerate(table_affixes):
    vec = vecs_affix[i]
    proj = proj_affix[i]
    
    total_energy = np.sum(vec**2)
    # Llama 70B dimension split (25% - 50% - 25%)
    head_energy = np.sum(proj[:2048]**2)     
    middle_energy = np.sum(proj[2048:6144]**2) 
    tail_energy = np.sum(proj[6144:]**2)     
    
    head_pct = (head_energy / total_energy) * 100
    mid_pct = (middle_energy / total_energy) * 100
    tail_pct = (tail_energy / total_energy) * 100
    
    results_markdown.append(
        f"| '{affix}' | {head_pct:.1f}% | {mid_pct:.1f}% | {tail_pct:.1f}% |"
    )

print("\n**Table 4: Affixation L2-Norm Energy Distribution in Llama-3.1-70B**\n")
print("| Affix Token | Head (0-25%) | Middle (25-75%) | Tail (75-100%) |")
print("|:---|:---:|:---:|:---:|")
for row_str in results_markdown:
    print(row_str)
print("\n")

# Step 4: Plot Energy Curve
print("Plotting Energy Curve...")

# Calculate mean absolute L1-norm energy
avg_energy_affix = np.mean(np.abs(proj_affix), axis=0)
avg_energy_stop = np.mean(np.abs(proj_stop), axis=0)

# Smoothing window
window_size = 256
smooth_affix = pd.Series(avg_energy_affix).rolling(window=window_size, min_periods=1).mean()
smooth_stop = pd.Series(avg_energy_stop).rolling(window=window_size, min_periods=1).mean()

plt.figure(figsize=(10, 6))

# Plot curves
plt.plot(smooth_affix, label="Indonesian Affixes (e.g., -nya, meng-, ter-)", color="crimson", linewidth=2.5)
plt.plot(smooth_stop, label="English Stopwords (e.g., the, and, of)", color="dodgerblue", linewidth=2.5)

# Add vertical bands for Head, Middle, and Tail Spectrum
plt.axvspan(0, 2048, color="lightcoral", alpha=0.15, label="Head Spectrum (0-2048)")
plt.axvspan(2048, 6144, color="khaki", alpha=0.15, label="Middle Spectrum (2048-6144)")
plt.axvspan(6144, 8192, color="lightgreen", alpha=0.15, label="Tail Spectrum (6144-8192)")

# Set Y-axis limits
y_max_limit = 0.04
plt.ylim(0, y_max_limit) 
plt.xlim(0, 8192)

plt.title("SVD Energy Distribution: Indonesian Affixes vs. English Stopwords (Llama-3.1-70B)", fontsize=14, fontweight="bold")
plt.xlabel("SVD Dimension Index (0 to 8192)", fontsize=12)
plt.ylabel(f"Average Energy (L1-Norm, Smoothed over {window_size} dims)", fontsize=12)
plt.legend(loc="upper right", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig('energy_curve_70b_final.png', dpi=300, bbox_inches="tight")
print("\nSaved plot as 'energy_curve_70b_final.png'.")

# Cleanup memory
del Vh_t, vecs_affix, vecs_stop, proj_affix, proj_stop
gc.collect()
print("Process completed.")
