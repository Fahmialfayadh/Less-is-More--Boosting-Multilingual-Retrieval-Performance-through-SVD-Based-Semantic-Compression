# Goal: Mengoreksi Metodologi EmbFilter Sesuai arXiv:2606.07502

Tujuan utama dari plan ini adalah memperbaiki *pipeline* ekstraksi dan aplikasi filter agar 100% patuh pada metodologi `EmbFilter` dari paper "Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings". 

## Background
Seperti yang dianalisis sebelumnya, *pipeline* lama memiliki kesalahan metodologi fatal:
1. Hanya menghapus 10 dimensi dari 1536 (terlalu kecil sehingga secara matematis tidak mengubah *cosine similarity*).
2. Memilih dimensi berdasarkan korelasi Pearson dengan distribusi frekuensi *unigram* corpus spesifik (Indonesian/English).
3. Menggunakan proyeksi *null-space* matriks Identitas `I - v_k v_k^T` tanpa mereduksi dimensi.

Pada arsitektur baru ini, kita akan mengikuti metode asli paper:
1. Melakukan SVD Penuh pada `lm_head.weight` ($V_h$ berdimensi 1536 x 1536).
2. Memotong dimensi *Head* (frekuensi tinggi) dan *Tail* (frekuensi sangat rendah). Untuk `filter_ratio=2`, kita akan mengekstrak matriks tengah `Vh_mid` berdimensi `768 x 1536`.
3. Memproyeksikan embeddings mentah dengan `Vh_mid.T`, menghasilkan embedding padat berdimensi 768.

## User Review Required

> [!WARNING]
> **Perubahan Paradigma "Language-Specific" Filter:**
> Pada paper aslinya, *Edge Spectrum Filter* adalah murni operasi aljabar linier terhadap matriks kosa kata (`lm_head.weight`) secara global. Filter ini bersifat **Language-Agnostic** (tidak membedakan Indonesia vs Inggris). Oleh karena itu, kita tidak akan lagi membandingkan `English-Filter` vs `Indonesian-Filter`. 
> 
> Sebagai gantinya, untuk membuktikan kebenaran paper tersebut di domain RAG Indonesia, kita akan mengevaluasi:
> 1. **Baseline** (Tanpa Filter, 1536 dimensi)
> 2. **Edge-Filter** (Membuang Head & Tail, 768 dimensi) -- *Ini adalah The Real EmbFilter*
> 3. **Head-Filter** (Menyisakan Head, membuang Tail, 768 dimensi)
> 4. **Tail-Filter** (Menyisakan Tail, membuang Head, 768 dimensi)

Apakah Anda setuju untuk mengganti komparasi bahasa dengan komparasi spektrum aljabar (Baseline vs Edge vs Head vs Tail)?

## Open Questions

> [!IMPORTANT]
> Apakah kita perlu mempertahankan skrip yang menghitung korelasi Pearson dengan token Indonesia? 
> Jika Anda **harus** mengklaim kontribusi *novelty* bahwa "Spektrum Frekuensi Indonesia berbeda dengan Inggris", kita bisa mengekstrak 384 vektor dengan korelasi Pearson tertinggi untuk teks Indonesia, dan menghapusnya. Namun, jika tujuannya murni validasi/implementasi *paper arXiv:2606.07502*, lebih baik kita *stick* pada Full SVD murni. Saya merekomendasikan Full SVD murni.

## Proposed Changes

### 1. `bundled_colab_run.py`
Ini adalah skrip *entry-point* utama yang akan di-*rewrite* isinya untuk menghasilkan skrip-skrip berikut:

#### [MODIFY] `src/extract_crosslingual_filters.py`
- Menghapus logika `TruncatedSVD`, `load_dataset`, dan penghitungan unigram frekuensi.
- Mengubahnya menjadi skrip sederhana yang melakukan SVD penuh dari `model.lm_head.weight.float()`.
- Menyimpan matriks $V_h$ berukuran `1536 x 1536` penuh ke dalam `models/vh_matrix.pt`.

#### [MODIFY] `src/qwen_embed_filter.py`
- Mengubah inisialisasi agar hanya memuat `models/vh_matrix.pt`.
- Membuat fungsi `encode_filtered(precomputed_raw_embs, filter_type="edge", filter_ratio=2)`.
- Mengimplementasikan pemotongan spektrum persis seperti source code asli:
  - `edge`: mengambil `Vh[384:1152, :]`
  - `head`: mengambil `Vh[-768:, :]`
  - `tail`: mengambil `Vh[0:768, :]`
- Memproyeksikan raw embeddings (1536D) ke ruang baru (768D) via operasi `torch.matmul(raw_embs, V_filter.T)`.

#### [MODIFY] `src/evaluate_crosslingual_rag.py`
- Memodifikasi fungsi `eval_sts_custom` dan `eval_rag_custom` untuk menerima `filter_type` (`None`, `edge`, `head`, `tail`).
- Membandingkan hasil dari `Baseline`, `Edge-Filter`, `Head-Filter`, dan `Tail-Filter`.

#### [MODIFY] `src/hypothesis_test_crosslingual.py`
- Melakukan t-test menggunakan `ranx.compare` antara `run_baseline`, `run_edge_filter`, `run_head_filter`, dan `run_tail_filter`.

## Verification Plan

### Automated Tests
1. Saya akan menjalankan skrip ekstraksi SVD lokal atau di terminal menggunakan `python -m py_compile src/extract_crosslingual_filters.py` (hanya memastikan syntax).
2. Skrip akhir tidak akan saya eksekusi via `colab exec` secara langsung tanpa persetujuan Anda karena memakan kredit. Skrip akan saya buat siap dieksekusi oleh Anda.

### Manual Verification
- Anda dapat mengeksekusi `bash run_crosslingual_colab.sh` menggunakan `colab-cli` Anda.
- Kita akan melihat apakah `Edge-Filter` secara signifikan mengungguli `Baseline` dan varian filter lainnya, membuktikan bahwa menekan *edge spectrum* meningkatkan kualitas representasi semantik.
