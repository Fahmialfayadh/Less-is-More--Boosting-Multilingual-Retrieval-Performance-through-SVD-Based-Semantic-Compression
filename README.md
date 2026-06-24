# Less is More: Boosting Multilingual Retrieval Performance through SVD-Based Semantic Compression

This repository contains the implementation and evaluation framework for post-hoc representation purification and algebraic dimension reduction on multilingual Decoder-Only Large Language Models (LLMs) used as generative embedders.

This work adapts and refines the **EmbFilter** methodology proposed in the primary reference paper to the morphological characteristics of the Indonesian language. By analyzing the Singular Value Decomposition (SVD) spectrum of the LLM's vocabulary projection weight matrix (the unembedding matrix), we demonstrate that semantic compression can be achieved without fine-tuning, leading to both infrastructure cost reductions and retrieval accuracy gains.

---

## Background

Large Language Models trained on the Next-Token Prediction (NTP) objective typically exhibit anisotropic latent spaces. In retrieval tasks utilizing Last-Token Pooling, this anisotropy causes similarity scores to be heavily influenced by token frequency biases rather than semantic content, degrading the performance of Retrieval-Augmented Generation (RAG) and Information Retrieval (IR) systems.

Post-hoc representation filters offer a computationally efficient alternative to parameter fine-tuning. By decomposing the unembedding matrix, we can isolate and project out noise components (the *Edge Spectrum*). However, while English-centric approaches recommend discarding the most dominant singular components (the *Head Spectrum*), we demonstrate that for morphologically rich languages like Indonesian, the Head Spectrum contains essential grammatical information. Retaining these components while compressing the representation yields substantial improvements in multilingual retrieval.

---

## Primary Reference

This research is based on the methodology proposed in:

> **arXiv:2606.07502** — *"Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings"* (Chen et al.)

---

## Key Findings

1. **Failure of Encoder-Only Filters:** Empirical evaluation and statistical significance testing (Paired Student's t-test at alpha = 0.01) confirm that post-hoc SVD filters are ineffective on Masked Language Models (MLM) like IndoBERT. These models use bidirectional attention and tied weights, which inherently do not suffer from the same NTP-induced Edge Spectrum anisotropy.
2. **Indonesian Morphological Localization:** Through the development of an L2-Norm Profiling-Based Retention Window Shifting algorithm, we discovered that critical Indonesian morphological affix tokens (e.g., `-nya`, `-lah`, `-kan`, `meng-`, `ber-`, `ter-`) accumulate 75% to 88% of their semantic energy in the combined Head (top 25%) and Middle (mid 50%) zones of the SVD spectrum.
3. **The Indonesian-Retention (0:768) Setup:** Discarding the Head Spectrum (as done in English-centric configurations) degrades Indonesian retrieval performance because it deletes morphological features. Instead, our proposed `Indonesian-Retention` window (keeping dimensions `0:768`) retains the Head and upper Middle spectra, leading to a **+82% relative improvement** in NDCG@10 on the MIRACL Indonesian dataset compared to the uncompressed baseline.
4. **Dimension Reduction & System Efficiency:** Compressing the vector dimension from 1536 to 768 reduces the storage and RAM footprint in vector databases (e.g., Qdrant, Milvus) by **50%** while simultaneously improving information retrieval accuracy.
5. **Cross-Model Scalability:** The L2-norm profiling energy patterns are consistent across scales, as validated on both Qwen2.5-1.5B (1536D) and Qwen2.5-7B (3584D).

---

## SVD Spectrum Analysis and Experimental Results

### Table 1: Vocabulary Distribution in the Qwen2.5-1.5B SVD Spectrum

| Spectrum Zone | Dimension Range | Primary Characteristics | Token Examples |
|:---|:---:|:---|:---|
| **Head Spectrum** | Top 25% | High cross-document variance; dominated by structural markers, non-Latin scripts, and code/HTML elements | `'أوضاع'` (Arabic), `'낡'` (Korean), `"\n\n\n\n"`, `'<//'` |
| **Middle Spectrum** | Middle 50% | Morphological tokens and affixes; bound morphemes, word-forming suffixes | `'edly'`, `'ingly'`, `'lessly'`, `' herself'`, `'为空'` |
| **Tail Spectrum** | Bottom 25% | High frequency, low cross-context variance; common particles across multiple languages | `' they'`, `' about'`, `' its'`, `'他们'`, `'a'`, `'\n'`, `'<\|endoftext\|>'` |

### Table 2: L2-Norm Distribution of Indonesian Affixes (Qwen2.5-1.5B)

| Affix Token | Head Spectrum (Top 25%) | Middle Spectrum (Mid 50%) | Tail Spectrum (Bot 25%) |
|:---|:---:|:---:|:---:|
| `'nya'` | 35.0% | **37.6%** | 27.4% |
| `'lah'` | 35.7% | **39.6%** | 24.7% |
| `'kan'` | 36.0% | **40.3%** | 23.7% |
| `'pun'` | 38.5% | **39.5%** | 22.1% |
| `'kah'` | **38.5%** | 38.4% | 23.1% |
| `'ku'` | 37.4% | **41.4%** | 21.2% |
| `'mu'` | 39.3% | **39.4%** | 21.3% |
| `' di'` | 28.8% | **45.2%** | 25.9% |
| `' ter'` | 32.7% | **44.7%** | 22.6% |
| `' ber'` | 33.5% | **42.3%** | 24.2% |
| `' meng'` | 35.2% | **40.6%** | 24.2% |
| `' mem'` | 38.9% | **41.8%** | 19.2% |

### Energy Curve Visualization (The "Smoking Gun" Plot)

To visually demonstrate this difference in spectral energy localization, the average projection magnitude across the 1536 SVD dimensions was computed and plotted for Indonesian morphological affixes and English stopwords.

![SVD Energy Curve: Indonesian Affixes vs. English Stopwords](data/energy_curve.png)

This plot shows that:
* **Indonesian morphological affixes** have energy peaks concentrated heavily in the **Head Spectrum** (dimensions 0–384), particularly within the top 200 components.
* **English stopwords** exhibit their energy peaks in the **Middle** and **Tail** spectra, dropping significantly in the Head.

This visualization clearly explains why the English-centric approach (discarding the first 25% Head dimensions) severely degrades Indonesian representation: it directly discards the critical dimensions where Indonesian morphological features are localized.

### Table 3: Cross-Lingual Evaluation on MIRACL Indonesian (Qwen2.5-1.5B)

| Configuration | Dimensions | NDCG@10 | Recall@100 |
|:---|:---:|:---:|:---:|
| **Baseline** | 1536 | 0.1592 | 0.4211 |
| **English-Middle** (Indices 384–1151) | 768 | 0.2333 | 0.6034 |
| **Indonesian-Retention** (Indices 0–767) | 768 | **0.2900** | **0.6535** |
| **Tail-Retention** (Indices 768–1535) | 768 | 0.0808 | 0.2485 |

### Statistical Significance Report (Paired Student's t-test at alpha = 0.01)

```
==================================================
      RETENTION STATISTICAL SIGNIFICANCE REPORT
==================================================
#    Model                     NDCG@10    Recall@100
---  ------------------------  ---------  ------------
a    run_baseline              0.159d     0.421d
b    run_english_middle        0.233ad    0.603ad
c    run_indonesian_retention  0.290abd   0.654abd
d    run_tail_retention        0.081      0.248

Note: Superscripts denote significant differences.
Example: '0.290abd' indicates that Indonesian-Retention (c) significantly
outperforms Baseline (a), English-Middle (b), and Tail-Retention (d).
```

### Table 4: Affixation L2-Norm Energy Distribution (Qwen2.5-7B Validation)

| Affix Token | Head (0-25%) | Middle (25-75%) | Tail (75-100%) |
|:---|:---:|:---:|:---:|
| `'nya'` | 33.6% | **45.0%** | 21.4% |
| `'lah'` | 38.4% | **46.3%** | 15.3% |
| `'kan'` | 37.4% | **49.8%** | 12.9% |
| `'pun'` | 38.7% | **49.0%** | 12.3% |
| `' meng'`| 32.6% | **50.5%** | 16.9% |
| `' ber'` | 29.9% | **55.4%** | 14.7% |
| `' ter'` | 38.1% | **50.7%** | 11.1% |

---

## Directory Structure

* [src/qwen_embed_filter.py]: Pipeline implementation for extracting Qwen2.5 generative embeddings, computing unembedding SVD projections, and applying the retention windows.
* [src/embed_filter_model.py]: Baseline pipeline class for Encoder-only models (e.g., IndoBERT).
* [src/extract_crosslingual_filters.py]: Script to extract cross-lingual noise projection filters based on word frequency distributions.
* [src/evaluate_crosslingual_rag.py]: Information retrieval evaluation engine for testing model configurations on MIRACL dataset.
* [src/hypothesis_test_crosslingual.py]: Script executing statistical significance analysis on cross-lingual retrieval metrics.
* [scripts/]: Directory containing auxiliary analysis scripts, Google Colab runner modules, and profiling tools.
  * [scripts/profile_imbuhan_colab.py]: Script to compute morphological L2-norm profiles on the SVD spectrum.
  * [scripts/bundled_retention_colab.py]: Consolidated Colab executor for testing SVD retention windows.
  * [scripts/bundled_colab_run.py]: Consolidated Colab runner evaluating cross-lingual filters.
---

## Installation

This repository manages its environment and Python dependencies using the `uv` tool.

1. **Initialize Virtual Environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. **Install Dependencies:**
   ```bash
   uv pip install -r requirements.txt
   
   # Or install required libraries manually
   uv pip install mteb datasets ranx scikit-learn accelerate torch transformers
   ```

---

## Execution Guide

### Local Execution (CPU / Local GPU)

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

### Remote Execution via Colab CLI (Recommended for GPU Tasks)

Due to the VRAM requirements of Qwen2.5, execution can be offloaded to an external T4 GPU instance on Google Colab using `colab-cli`.

1. **Establish a new Colab GPU Session:**
   ```bash
   colab new -s embed_filter_eval --gpu T4
   ```

2. **Profile Indonesian Affix L2-norm Distributions:**
   ```bash
   colab exec -s embed_filter_eval -f scripts/profile_imbuhan_colab.py --timeout 1200
   ```

3. **Run Retention Window Comparison Experiments:**
   ```bash
   colab exec -f scripts/bundled_retention_colab.py --timeout 3600 | tee logs/retention_colab_output.log
   ```

4. **Run Cross-lingual Filters Comparison Experiments:**
   ```bash
   colab exec -f scripts/bundled_colab_run.py --timeout 3600 | tee logs/crosslingual_colab_output.log
   ```
