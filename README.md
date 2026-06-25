# Less is More: Boosting Multilingual Retrieval Performance through SVD-Based Semantic Compression

This repository contains the implementation and evaluation framework for post-hoc representation purification and algebraic dimension reduction on multilingual Decoder-Only Large Language Models (LLMs) used as generative embedders. 

For the complete research details, methodology, and comprehensive experimental results, please read the full papers:
- 🇬🇧 [English Version](paper_en.md)
- 🇮🇩 [Indonesian Version](paper_id.md)

---

## 🚀 Key Highlights

* **Problem**: EmbFilter (Chen et al., 2026) discards the Head SVD spectrum, assuming it only contains high-frequency stopword noise. While true for English, this assumption fails for agglutinative languages.
* **Discovery**: Through L2-Norm profiling, we show that critical Indonesian morphological affixes (e.g., `-nya`, `-lah`, `-kan`, `meng-`, `ber-`, `ter-`) concentrate **75% to 88%** of their semantic energy in the Head and Middle spectra.
* **Solution (`Indonesian-Retention`)**: We propose shifting the retention window to dimensions `0:768`. This retains essential morphological features while discarding Tail noise.
* **Results**: Achieves a **+82% relative improvement in NDCG@10** (from 0.1592 to 0.2900) on the MIRACL Indonesian retrieval task, while simultaneously **compressing vector storage by 50%**.

### The "Smoking Gun" SVD Energy Curve
![SVD Energy Curve: Indonesian Affixes vs. English Stopwords](data/energy_curve.png)
*Indonesian affixes peak heavily in the Head Spectrum, while English stopwords peak in the Tail. Discarding the Head Spectrum severely degrades Indonesian retrieval accuracy.*

---

## 📂 Directory Structure

* `src/qwen_embed_filter.py`: Pipeline for extracting Qwen2.5 embeddings and applying SVD retention windows.
* `src/embed_filter_model.py`: Baseline pipeline class for Encoder-only models (e.g., IndoBERT).
* `src/extract_crosslingual_filters.py`: Script to extract cross-lingual noise projection filters.
* `src/evaluate_crosslingual_rag.py`: Information retrieval evaluation engine on the MIRACL dataset.
* `src/hypothesis_test_crosslingual.py`: Statistical significance analysis (Paired Student's t-test).
* `scripts/`: Auxiliary analysis scripts, Google Colab runner modules, and profiling tools.

---

## 🛠️ Installation

This repository manages its environment and Python dependencies using `uv`.

```bash
# 1. Initialize Virtual Environment
uv venv
source .venv/bin/activate

# 2. Install Dependencies
uv pip install -r requirements.txt
# Or manually: uv pip install mteb datasets ranx scikit-learn accelerate torch transformers
```

---

## 💻 Execution Guide

Before executing scripts locally, add the `src` directory to your `PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 1. Extract the frequency-correlated cross-lingual projection filters
uv run python src/extract_crosslingual_filters.py

# 2. Run retrieval evaluations on the Indonesian MIRACL dataset
uv run python src/evaluate_crosslingual_rag.py

# 3. Perform statistical significance checks
uv run python src/hypothesis_test_crosslingual.py
```

### Remote Execution via Colab CLI (Recommended for GPU)
Due to the VRAM requirements of Qwen2.5/Llama-3.1, execution can be offloaded to an external GPU on Google Colab using `colab-cli`.

```bash
# Establish a new Colab GPU Session
colab new -s embed_filter_eval --gpu T4

# Profile Indonesian Affix L2-norm Distributions
colab exec -s embed_filter_eval -f scripts/profile_imbuhan_colab.py --timeout 1200

# Run Retention Window Comparison Experiments
colab exec -f scripts/bundled_retention_colab.py --timeout 3600 | tee logs/retention_colab_output.log
```
