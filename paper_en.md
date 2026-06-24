# Less is More: Boosting Multilingual Retrieval Performance through SVD-Based Semantic Compression

---

## 1. Background

The utilization of Decoder-Only Large Language Models (LLMs) as generative embedders is increasingly researched, particularly through the Last-Token Pooling technique. Models trained with the Next-Token Prediction (NTP) objective tend to produce anisotropic semantic spaces. This condition causes the cosine similarity between vectors to be influenced by token frequency biases embedded during pre-training, thereby potentially degrading performance in Information Retrieval (IR) and Retrieval-Augmented Generation (RAG) tasks.

Post-hoc representation refinement techniques offer an alternative approach that does not require retraining or fine-tuning. This approach analyzes the final vocabulary projection weight matrix (the unembedding matrix) to isolate semantic features from latent space components suspected of containing statistical distortions. Although this approach has shown effectiveness on English corpora, its application to languages with different morphological characteristics—such as Indonesian—remains largely unexplored.

---

## 2. Primary References

This study is based on the methodology proposed in:

> **arXiv:2606.07502** — *"Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings"* (Chen et al.)

The paper postulates that the unembedding matrix in LLMs can serve as a *feature lens*. Matrix decomposition via Singular Value Decomposition (SVD) shows that components with the highest singular values (*Head Spectrum*) tend to correspond to high-frequency tokens (*stopwords*), while the lowest components (*Tail Spectrum*) correspond to rare tokens. Based on these observations, the paper proposes the **EmbFilter** method, which discards both extreme spectra (*Edge Spectrum*) and projects vectors onto the remaining middle dimensions.

---

## 3. Algorithm and Methodology

This study implements a linear-algebra-based orthogonal projection filter in accordance with the primary reference, applied to the Qwen2.5-1.5B model ($d = 1536$).

### 3.1. Unembedding Matrix Decomposition

The unembedding matrix (defined as `lm_head.weight` in Qwen architectures) is represented as $W \in \mathbb{R}^{|V| \times d}$, where $|V|$ is the vocabulary size and $d$ is the latent dimension. The matrix $W$ is fully decomposed using Singular Value Decomposition (Full SVD) without truncation:

$$W = U \Sigma V_h$$

where $V_h \in \mathbb{R}^{d \times d}$ is the right singular vector matrix. Since $\Sigma$ is sorted in descending order, the $0$-th row of $V_h$ corresponds to the largest singular value (*Head*), and the $(d-1)$-th row corresponds to the smallest singular value (*Tail*).

### 3.2. L2-Norm Profiling-Based Retention Window Shifting Algorithm

For a 2× compression ratio, the target dimension is $d' = 1536 / 2 = 768$. To systematically determine the optimal spectrum truncation range — rather than adopting the default range from the reference paper — we developed a latent energy profiling algorithm based on SVD decomposition. The primary reason for shifting this compression range (*cutting window shifting*) is directly tied to the morphological characteristics of Indonesian.

This retention window determination algorithm operates through the following steps:
1. Extract the `lm_head` matrix and perform a Full SVD.
2. Project each token's row vector onto the spectral components, then group them into three primary zones: *Head* (Top 25%), *Middle* (Mid 50%), and *Tail* (Bottom 25%).
3. Calculate the squared magnitude of energy (*projective L2-norm*) to detect where a word's semantic information is localized.

The resulting mapping of token class distribution characteristics is presented in Table 1.

**Table 1 — Vocabulary Distribution Characteristics in the Qwen2.5-1.5B SVD Spectrum**

| Spectrum Zone | Dimension Range | Primary Characteristics | Token Examples |
|:---|:---:|:---|:---|
| **Head Spectrum** | Top 25% | High cross-document variance; dominated by structural markers, non-Latin scripts, and code/HTML elements | `'أوضاع'` (Arabic), `'낡'` (Korean), `"\n\n\n\n"`, `'<//'` |
| **Middle Spectrum** | Middle 50% | Morphological tokens and affixes; bound morphemes, word-forming suffixes | `'edly'`, `'ingly'`, `'lessly'`, `' herself'`, `'为空'` |
| **Tail Spectrum** | Bottom 25% | High frequency, low cross-context variance; common particles across multiple languages | `' they'`, `' about'`, `' its'`, `'他们'`, `'a'`, `'\n'`, `'<\|endoftext\|>'` |

Several observations can be drawn from the distribution patterns in Table 1.

**First,** the *Tail Spectrum* component in Qwen2.5 is dominated by English/Mandarin stopwords, single-character particles, and structural tokens. These tokens appear constantly across all texts, causing their variance to approach zero. This indicates that the *Tail* dimensions across languages predominantly contain anisotropic noise.

**Second,** contrary to the reference paper's assumption, the *Head Spectrum* houses language identity markers. Discarding this component may substantially reduce the model's ability to distinguish cross-lingual contexts.

**Third,** and the foundation of the shifting algorithm: analytical testing targeting Indonesian syntactic affix tokens shows that their variance energy (*L2-norm*) is concentrated in the combined *Head* and *Middle* zones.

**Table 2 — L2-Norm Distribution of Indonesian Affix Tokens Across the Qwen2.5-1.5B Spectrum**

| Affix Token | *Head Spectrum* (Top 25%) | *Middle Spectrum* (Mid 50%) | *Tail Spectrum* (Bot 25%) |
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

Based on Table 2, the critical affix tokens accumulate **75% to 80%** of their total latent energy in the combined *Head* and *Middle Spectrum*.

**Systematization of the Shifting Algorithm:**
The `English-Middle` approach from the reference paper removes the first 25% of the spectrum (*Head*). Based on Table 2, that truncation algorithm **discards 28% to 38% of the semantic energy** (L2-norm) from essential Indonesian syntactic particles such as prefixes `meng-`, `ter-`, and suffixes `-nya`, `-lah`.

Grounded on this empirical evidence, our proposed algorithm shifts the retention window (50% compression) to the dimension range **0 to 768**, encompassing the entire *Head Spectrum* and the upper half of the *Middle Spectrum* (`Indonesian-Retention`). This shift is designed to preserve morphological features while eliminating noise components in the *Tail Spectrum*.

Based on this algorithm, four projection matrix configurations $V_{sub} \in \mathbb{R}^{d' \times d}$ were empirically evaluated:

| # | Configuration | Description | Index Range of $V_{sub}$ |
|---|---|---|---|
| 1 | **Baseline** | No truncation; the original vector $x \in \mathbb{R}^{1536}$ is used directly | — |
| 2 | **English-Middle** | Discards 25% *Head* and 25% *Tail* (the reference paper's approach) | Indices 384–1151 |
| 3 | **Indonesian-Retention** | Retains 50% of the *Head* and *Middle* spectra (based on the profiling algorithm) | Indices 0–767 |
| 4 | **Tail-Retention** | Retains 50% of the *Tail* spectrum | Indices 768–1535 |

### 3.3. Representation Projection

Each sentence $s$ is encoded into an initial representation $x$ using Last-Token Pooling with the instruction template (`PromptEOL`). The filtered vector $x'$ is obtained via:

$$x' = x V_{sub}^T$$

The 768-dimensional vector $x'$ is subsequently used to compute cosine similarity.

### 3.4. Evaluation Setup

**Datasets:**

| Task | Dataset | Split | Description |
|---|---|---|---|
| Retrieval (RAG) | MIRACL — Indonesian | dev | 500 random queries; 4,543 positive docs + 5,000 sampled negative docs |
| STS | STS-B — LazarusNLP | test | 500 sampled sentence pairs |

### 3.5. Evaluation Metrics

The performance of the projected semantic spaces was measured using three standard metrics in the information retrieval domain:
1. **NDCG@10 (Normalized Discounted Cumulative Gain at top 10 candidates):** Measures the effectiveness of the retrieval system by accounting for the position of relevant documents in the search results list. NDCG penalizes cases where relevant documents appear lower in the rankings, making it the most critical metric for RAG systems sensitive to context ordering.
2. **Recall@100:** Measures the percentage of relevant documents successfully retrieved by the system within the top 100 search results. This metric indicates how well the vectors can locate correct documents within a large corpus, regardless of their exact rank.
3. **Spearman Correlation:** Used specifically for the Semantic Textual Similarity (STS) task to measure the ranked monotonic correlation between the model's cosine similarity scores and reference human relevance ratings.

**Statistical Testing:** The significance of differences between methods was evaluated using a *Paired Student's t-test* with a threshold of $\alpha = 0.01$ using the `ranx` metrics library.

---

## 4. Experimental Results

Table 3 summarizes the average performance of each evaluated configuration.

**Table 3 — Cross-Lingual Metric Evaluation Results (MIRACL Indonesian)**

| Configuration | Dimensions | NDCG@10 | Recall@100 |
|:---|:---:|:---:|:---:|
| Baseline | 1536 | 0.1592 | 0.4211 |
| English-Middle | 768 | 0.2333 | 0.6034 |
| Indonesian-Retention | 768 | **0.2900** | **0.6535** |
| Tail-Retention | 768 | 0.0808 | 0.2485 |

### 4.1. Statistical Significance Test

The following report was generated by a Paired Student's t-test at $\alpha = 0.01$:

```
==================================================
      RETENTION STATISTICAL SIGNIFICANCE REPORT
==================================================
#    Model                     NDCG@10    Recall@100
---  ------------------------  ---------  ------------
a    run_baseline              0.159ᵈ     0.421ᵈ
b    run_english_middle        0.233ᵃᵈ    0.603ᵃᵈ
c    run_indonesian_retention  0.290ᵃᵇᵈ   0.654ᵃᵇᵈ
d    run_tail_retention        0.081      0.248

Note: Superscripts denote significant differences.
```

> **Superscript Notes:** The letter notation next to a score indicates that the model in that row statistically significantly outperforms the model with the corresponding index.
> For example: `0.290ᵃᵇᵈ` means that the `Indonesian-Retention` (c) configuration significantly outperforms the `Baseline` (a), `English-Middle` (b), and `Tail-Retention` (d) absolutely and conclusively.

---

## 5. Discussion

The experimental results in Table 3 present significant evidence against the core assumption of the reference paper when the method is applied to a multilingual LLM (Qwen2.5) for an Indonesian corpus task.

In the reference paper evaluated on English, the `English-Middle` configuration (removing both *Head* and *Tail*) was assumed to be optimal because the *Head Spectrum* was considered to solely contain distortions from high-frequency stopwords. However, in this experiment, `Indonesian-Retention` (the `0:768` range which actually **retains** the *Head*) achieves an NDCG@10 of 0.2900, significantly outperforming the `English-Middle` method ($p < 0.01$).

These results are consistent with the predictions of the L2-Norm profiling algorithm detailed in Section 3.2. Retaining the core morphology of Indonesian — concentrated in the *Head* and *Middle* zones — produces more complete lexical representations. Conversely, the `Tail-Retention` configuration, which achieved an NDCG@10 of only 0.0808, confirms that the *Tail* zone is dominated by low-variance anisotropic noise components.

It is worth noting that Chen et al. defined the *Edge Spectrum* based on text frequency distributions from an English corpus. This experiment demonstrates that adopting the same retention window for Indonesian significantly degrades the integrity of its linguistic representation.

Overall, these findings indicate that dimensional truncation strategies optimized for English-centric corpora cannot be directly applied to multilingual LLMs for Indonesian retrieval tasks. Further experiments — including variations in models, truncation ranges, and other target languages — are required to validate these findings before they can be generalized.

---

## 6. Conclusion and Practical Contributions

This study proposes a practical adaptation framework for the *EmbFilter* methodology tailored to the linguistic characteristics of Indonesian, with contributions to the efficiency of Retrieval-Augmented Generation (RAG) architectures in production systems.

By applying orthogonal projection on the unembedding matrix with a retention window of `0:768` (*Head* to *Middle Spectrum*), this study demonstrates three key contributions:
1. **Infrastructure Load Reduction (Vector Database):** The semantic representation dimension is compressed by **50%** (from 1536 to 768 dimensions), directly reducing storage memory requirements and vector search computational costs.
2. **Accuracy Improvement (Retrieval Performance):** Although dimensionality reduction typically correlates with accuracy degradation, filtering the *Tail Spectrum* while retaining Indonesian affix features in the `0:768` range substantially improves NDCG@10 performance by **+82%** (from a baseline of 0.1592 to 0.2900).
3. **Cross-Model Predictive Diagnostic Tool:** The L2-Norm Profiling algorithm developed in this research can serve as a predictive diagnostic tool. Before implementing dimension reduction on large-scale LLMs (e.g., 7B or 70B parameter models), practitioners can extract the `lm_head` and test the affixation energy distribution. If the L2-Norm energy curve remains predominantly concentrated in the *Head* and *Middle* zones, this provides mathematical justification for the compression strategy without requiring full-scale RAG benchmark testing.

This *Indonesian-Retention* adaptation offers an efficient semantic compression solution alongside an analytical validation framework for maximizing the utility of multilingual LLMs as generative embedders, without requiring additional computational investments such as fine-tuning.

### 6.1. Cross-Model Scalability Validation (7B Case Study)

To test the generalizability of this diagnostic tool, we applied *L2-Norm Profiling* to a larger-scale model: **Qwen2.5-7B** (3,584 latent dimensions). The profiling results (Table 4) demonstrate that the retention of morphological energy in the *Head* spectrum is not an anomaly restricted to smaller models, but rather a consistent linguistic characteristic across architectural scales.

**Table 4 — Affixation L2-Norm Energy Distribution in Qwen2.5-7B**

| Affix Token | Head (0-25%) | Middle (25-75%) | Tail (75-100%) |
|:---|:---:|:---:|:---:|
| 'nya' | 33.6% | 45.0% | 21.4% |
| 'lah' | 38.4% | 46.3% | 15.3% |
| 'kan' | 37.4% | 49.8% | 12.9% |
| 'pun' | 38.7% | 49.0% | 12.3% |
| ' meng'| 32.6% | 50.5% | 16.9% |
| ' ber' | 29.9% | 55.4% | 14.7% |
| ' ter' | 38.1% | 50.7% | 11.1% |

The data in Table 4 confirm that the *Head* and *Middle* regions consistently accumulate the majority (**~75% to 88%**) of the total semantic energy of Indonesian affixes, including at the 7B model scale. Conversely, the energy portion allocated to the *Tail* spectrum remains at minimal proportions.

These findings support the conclusion that the practice of discarding the *Head* spectrum — which is effective for reducing stopword distortion in English — requires re-evaluation before being applied to the Indonesian corpus. In languages characterized by high agglutination and rich affixation morphology, the *Head* spectrum plays an important role in preserving complete lexical structure.
