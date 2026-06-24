# Rekapitulasi Linimasa dan Arsitektur Proyek Riset

**Topik:** Analisis Lintas-Arsitektur dan Adaptasi Lintas-Bahasa pada Teknik *Post-Hoc Unembedding Filter* untuk Representasi Teks Bahasa Indonesia
**Referensi Utama:** arXiv:2606.07502 — *"Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings"*

Dokumen ini merangkum secara objektif dan kronologis seluruh proses riset, mulai dari inisiasi masalah hingga formulasi kontribusi kebaruan (*novelty*). Rangkuman ini disusun agar dapat langsung dipakai sebagai kerangka Bab Metodologi dan Pembahasan pada laporan akhir atau publikasi ilmiah.

---

## 1. Inisiasi dan Formulasi Masalah

Riset ini berawal dari upaya mengoptimalkan sistem *Retrieval-Augmented Generation* (RAG) untuk domain UMKM tanpa melakukan *fine-tuning* ulang model, mengingat biaya komputasinya yang tinggi. Pendekatan yang dipilih adalah *Post-Hoc Representation Purification*, yaitu menerapkan dekomposisi *Singular Value Decomposition* (SVD) dan proyeksi ortogonal pada matriks *unembedding* (`lm_head`) untuk memisahkan ruang semantik dari *noise* frekuensi tinggi yang disebut *Edge Spectrum*.

Efektivitas pendekatan ini diukur melalui dua jenis metrik: metrik *Information Retrieval* (NDCG@10 dan Recall@100) menggunakan dataset MIRACL, serta metrik *Semantic Textual Similarity* (korelasi Spearman) menggunakan STS-B.

## 2. Fase Eksperimen 1: Evaluasi pada Arsitektur Encoder-Only (IndoBERT)

Tahap pertama menerapkan filter pada model IndoBERT P1, sebuah *Masked Language Model* (MLM). Dua variabel diuji pada tahap ini: parameter $k$, yang mengatur jumlah komponen *Edge Spectrum* yang dibuang, dan parameter $\tau$, yang mengatur rasio kompresi dimensi vektor melalui *Truncated SVD*.

Temuan empiris menunjukkan dua hal yang kontras. Penerapan filter $k$ tidak mengubah metrik NDCG maupun Recall secara signifikan. Sebaliknya, kompresi dimensi pada $\tau=256$ justru meningkatkan skor STS — yang mencerminkan penyelarasan semantik global yang lebih baik — namun pada saat yang sama menurunkan skor NDCG pada dataset yang mengandung *hard-negatives*.

## 3. Validasi Statistik dan Penemuan Batas Arsitektur

Untuk memastikan temuan di atas bukan sekadar variasi acak, dilakukan uji signifikansi statistik menggunakan *Paired Student's t-test* pada taraf signifikansi $\alpha = 0.01$. Hasilnya menegaskan pola yang sama: variasi parameter $k$ dari 1 hingga 20 pada IndoBERT menghasilkan skor yang identik secara statistik dengan *baseline*, sehingga hipotesis nol diterima. Sementara itu, kompresi dimensi pada $\tau < 768$ terbukti menurunkan performa *retrieval* (NDCG) secara statistik signifikan dibandingkan *baseline*.

Dari hasil ini ditarik satu kesimpulan analitis penting: arsitektur *Encoder-Only* (MLM), yang menggunakan *tied-weights* dan konteks dua arah (*bidirectional*), secara inheren tidak menderita anisotropi *Edge Spectrum* separah arsitektur *Decoder-Only*. Artinya, teknik filter dari paper rujukan tidak relevan secara statistik untuk arsitektur MLM, karena metode tersebut pada dasarnya dirancang khusus untuk model *Decoder-Only*.

## 4. Redefinisi Masalah dan Pivot Riset: Koreksi Metodologi EmbFilter

Temuan pada Fase 1 membuka sebuah kesenjangan riset: paper rujukan memvalidasi tekniknya pada model *Decoder-Only* (seperti Llama dan Qwen) menggunakan korpus berbahasa Inggris (RedPajama). Model *Decoder-Only* yang dilatih dengan *Next-Token Prediction* (NTP) memiliki bias frekuensi kata yang menciptakan anisotropi (diklaim sebagai *Edge Spectrum*).

Awalnya, riset ini mencoba mengajukan konsep **Indonesian-Specific Edge Spectrum** dengan membuang 10 dimensi SVD yang memiliki korelasi Pearson tertinggi terhadap distribusi frekuensi teks Indonesia. Namun, analisis *source code* secara mendalam membuktikan bahwa pendekatan ini menyimpang secara fatal dari metodologi asli paper rujukan (arXiv:2606.07502). Paper asli tidak menggunakan korelasi dinamis maupun pembuangan komponen parsial (*null-space projection*), melainkan melakukan reduksi dimensi aljabar absolut.

Oleh karena itu, riset ini diredefinisi untuk mengimplementasikan **The True EmbFilter** secara murni pada ekosistem teks Indonesia:
- Beralih ke model *Decoder-Only* (Qwen2.5-1.5B) yang difungsikan sebagai *Generative Embedder* menggunakan **Last-Token Pooling** dipadukan dengan *Task-Aware Instruction* (`PromptEOL`).
- Filter *Edge Spectrum* diimplementasikan murni secara aljabar (Language-Agnostic): Melakukan Dekomposisi Nilai Singular (SVD) penuh pada matriks *unembedding* (`lm_head.weight`) secara utuh.
- Mengurangi dimensi representasi hingga setengahnya (1536D $\rightarrow$ 768D) dengan cara mengalikan *raw embeddings* dengan matriks singular yang telah dipotong bagian *Head* (frekuensi sangat tinggi) dan *Tail*-nya (frekuensi sangat rendah).

Sebagai desain eksperimen lanjutan, untuk membuktikan kebenaran postulat paper pada tugas *retrieval* lintas-bahasa (Indonesia), dilakukan komparasi empiris terhadap empat konfigurasi proyeksi:
1. **Baseline** (Tanpa Filter, dimensi utuh 1536)
2. **Edge-Filter** (Membuang *Head* & *Tail*, dimensi 768) — *Implementasi Asli EmbFilter*
3. **Head-Filter** (Membuang *Tail*, menyisakan *Head*, dimensi 768)
4. **Tail-Filter** (Membuang *Head*, menyisakan *Tail*, dimensi 768)

## 5. Analisis Dampak: Efisiensi Infrastruktur RAG

Pendekatan reduksi dimensi aljabar dari *EmbFilter* memberikan keuntungan ganda yang fundamental bagi arsitektur *Vector Database* (seperti Qdrant atau Milvus):
1. **Pemangkasan Dimensi (Dimensionality Reduction):** Vektor dipangkas secara ekstrem dari 1536 dimensi menjadi 768 dimensi (reduksi ukuran sebesar 50%). Ini menurunkan jejak memori RAM/*storage* secara masif untuk jutaan dokumen RAG.
2. **Pembersihan Noise (Semantic Purification):** Membuang *Edge Spectrum* dari ruang vektor dipercaya meredam distorsi *cosine similarity* yang disebabkan oleh distribusi partikel dan sintaksis bawaan NTP. 

---

## Status Riset Saat Ini

1. **Fase 1 (IndoBERT / Encoder)** — Selesai. Data empiris dan uji signifikansi statistik telah dihasilkan, membuktikan bahwa metode SVD Filter tidak relevan untuk arsitektur MLM (*Encoder-Only*).
2. **Fase 2 (Qwen / Decoder & Penemuan Sweet Spot)** — Selesai. Melalui *micro-profiling* spektrum Qwen2.5-1.5B, ditemukan bahwa **Filter Ratio dari paper rujukan berbahasa Inggris (membuang Head) terbukti cacat untuk bahasa Indonesia**. Energi token imbuhan morfologis bahasa Indonesia sangat terpusat pada *Head Spectrum* (dimensi `0:384`). Membuang *Head* sama dengan membuang lebih dari 50% informasi imbuhan krusial.
3. **Fase 3 (Implementasi Retensi Head-to-Middle)** — Sedang berjalan. Berdasarkan temuan Fase 2, *sweet spot* pemotongan dimensi untuk Bahasa Indonesia telah ditetapkan pada rentang **0:768**. Eksperimen komparatif pada dataset MIRACL dan STS-B sedang disiapkan untuk membuktikan konfigurasi kompresi 50% ini.

### Detail Algoritma Ekstraksi *EmbFilter* (Fase 3 - Jendela Retensi)
Proses pemetaan dan pemotongan spektrum dilakukan dengan aljabar linier murni:
1. **SVD Penuh pada Matriks Mentah:** Matriks *unembedding* Qwen2.5 ($W \in \mathbb{R}^{|V| \times d}$) didekomposisi secara utuh (Full SVD) tanpa *truncation* menjadi $W = U \Sigma V_h$. Menghasilkan matriks vektor singular kanan $V_h$ berdimensi $1536 \times 1536$.
2. **Pemotongan Komponen Edge (Tail-Only Truncation):** Alih-alih menghapus *Head* seperti pada literatur Inggris, reduksi dimensi untuk bahasa Indonesia dilakukan dengan membuang 50% bagian *Tail* (`768:1536`), sekaligus mengamankan (me-retain) jendela `0:768` yang menyimpan 61.46% total energi L2-Norm dari varian morfologis (imbuhan).
3. **Proyeksi Reduksi Dimensi:** Matriks komponen utama yang dipertahankan, $V_{opt}$ (berdimensi $768 \times 1536$), digunakan sebagai matriks transformasi linier. Representasi *Last-Token* dari LLM dikalikan dengan $V_{opt}^T$, mereduksi ukuran vektor dari 1536D ke 768D secara optimal tanpa merusak *Information Retrieval* berbahasa Indonesia.