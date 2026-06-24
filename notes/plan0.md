
---

### 🛠️ IMPLEMENTATION PLAN: "EmbedFilter" untuk RAG UMKM

#### Fase 1: Ekstraksi "Kacamata" (The Feature Lens)
Tujuannya adalah mencari tahu arah vektor mana di dalam model yang merepresentasikan "kata-kata sampah / *stop words*".
1. **Load Model:** Muat model *pre-trained* (misal: `indobenchmark/indobert-lite-base-p1` atau `intfloat/multilingual-e5-large`).
2. **Ekstrak Matriks Unembedding ($W$):** Ambil bobot dari layer terakhir yang memetakan *hidden state* kembali ke kosakata (`lm_head.weight`). Ukuran matriks ini adalah `[Vocab_Size, Hidden_Dim]`.
3. **Definisikan Subruang "Noise" (Stop Words):**
   * Kumpulkan 200-500 kata frekuensi tinggi di Bahasa Indonesia (misal: *yang, dan, di, ke, dari, adalah, dengan*).
   * Ambil vektor *embedding* mentah dari kata-kata tersebut.
   * Lakukan **SVD (Singular Value Decomposition)** atau **PCA** pada kumpulan vektor *stop word* tersebut. Ambil $k$ komponen utama (misal $k=10$). Matriks hasil ini (sebut saja $V_{noise}$) adalah "arah" dari token sampah di dalam ruang vektor model.

#### Fase 2: Pembuatan Filter (Transformasi Linear)
Kita buat matriks proyeksi $P$ yang akan membuang komponen $V_{noise}$ dari vektor *output*.
*   **Rumus Aljabar Linear:** $P = I - (V_{noise} \times V_{noise}^T)$
    *(Di mana $I$ adalah matriks identitas. Operasi ini secara harfiah "menolak" bayangan vektor ke arah ruang noise).*
*   Matriks $P$ ini dihitung **hanya sekali** di awal menggunakan NumPy/PyTorch CPU, lalu disimpan (save) sebagai file `.pt` atau `.npy`.

#### Fase 3: Pipeline Inference (Real-time RAG & Klasifikasi)
Ini adalah tahap yang Anda jalankan setiap kali ada *user* yang bertanya atau ada ulasan baru yang masuk.
1. **Encode Teks Asli:** Masukkan teks UMKM utuh (tanpa buang *stop word*) ke model. Dapatkan vektor embedding mentah $E_{raw}$ (ukuran: $1 \times 1024$).
2. **Terapkan Filter (Post-Hoc):** Kalikan vektor dengan matriks proyeksi.
   * $E_{filtered} = E_{raw} \times P$
3. **Reduksi Dimensi (Bonus dari Paper):**
   * Lakukan Truncated SVD pada $E_{filtered}$ untuk memampatkannya dari 1024 dimensi menjadi 256 atau 128 dimensi.
   * *Hasil:* Vektor Anda sekarang murni secara semantik, dan ukurannya 75% lebih kecil!
4. **Simpan / Query:** Masukkan ke Vector Database (seperti ChromaDB/Faiss) atau gunakan untuk klasifikasi.

---

### 📊 METRIK EVALUASI (Cara Mengukur Keberhasilan)

Paper aslinya menggunakan standar evaluasi representasi yang ketat. Anda harus membandingkan **Baseline (Embedding Mentah)** vs **EmbedFilter** menggunakan metrik berikut:

**1. Untuk RAG / Information Retrieval (Pencarian Produk/FAQ UMKM)**
*   **Recall@K (K=5, 10):** Dari 10 dokumen teratas yang ditarik sistem, berapa persen yang benar-benar relevan dengan *query* pelanggan?
*   **NDCG@10 (Normalized Discounted Cumulative Gain):** Mengukur kualitas urutan. Apakah dokumen yang *paling* relevan muncul di urutan nomor 1, bukan nomor 5?
*   **Latency & Storage:** Ukur waktu *query* (ms) dan ukuran *index* database (MB). (EmbedFilter harusnya menang telak di metrik ini karena vektornya dipangkas).

**2. Untuk Klasifikasi Zero-Shot (Misal: Sentimen Ulasan atau Deteksi Niat/Intent)**
*   **Accuracy & F1-Score:** Gunakan *Cosine Similarity* antara vektor ulasan dengan vektor label ("Positif", "Negatif"). Ukur akurasinya tanpa *training* (zero-shot).

**3. Untuk Kesamaan Semantik (Semantic Textual Similarity / STS)**
*   **Spearman’s Rank Correlation:** Membandingkan skor *cosine similarity* antar kalimat dengan skor penilaian manusia.

---

### 🗂️ DATA APA SAJA YANG DIPERLUKAN UNTUK PENGUJIAN?

Untuk membuktikan bahwa teknik ini bekerja di Bahasa Indonesia (khususnya domain UMKM), Anda tidak butuh dataset jutaan baris. Anda butuh dataset evaluasi yang *kualitasnya terjamin*. Siapkan 4 jenis data ini:

#### 1. Data "Kamus Noise" (Untuk Fase 1)
*   **Format:** List teks (CSV/TXT).
*   **Isi:** Kumpulan *stop words* Bahasa Indonesia (bisa ambil dari Sastrawi) + 500 kata paling umum dari kamus produk UMKM (misal: "promo", "gratis", "ongkir", "ready" — kata-kata ini sering muncul di mana-mana dan bisa mengaburkan makna spesifik produk jika tidak difilter).

#### 2. Data Evaluasi Retrieval (Untuk RAG)
*   **Format:** Tabular (Query, Relevant_Doc_ID, Irrelevant_Doc_IDs).
*   **Isi:** Buat 50 - 100 *query* nyata dari pelanggan.
    *   *Query:* "Bahan baju ini luntur gak kalau dicuci?"
    *   *Dokumen Relevan:* "Gunakan deterjen lembut, bahan katun kami sudah melalui proses *color-lock*."
    *   *Dokumen Pengecoh (Hard Negatives):* "Baju ini tersedia dalam berbagai warna cerah dan siap dikirim." (Mengandung kata kunci mirip tapi tidak menjawab pertanyaan).
*   *Sumber:* Log chat customer service UMKM Anda, atau buat secara manual/sintetis menggunakan LLM dari katalog produk.

#### 3. Data Evaluasi Klasifikasi (Sentimen / Intent)
*   **Format:** Tabular (Teks_Ulasan, Label_Sentimen).
*   **Isi:** 200-500 ulasan pelanggan dari *e-commerce* (Shopee/Tokopedia) yang sudah dilabeli (Positif, Negatif, Netral) atau label komplain (Pengiriman, Kualitas Barang, Pelayanan).
*   *Sumber:* Dataset publik seperti *Indonesian Shopee Reviews Tagged* atau *GoEmotions* yang diterjemahkan/diadaptasi ke lokal.

#### 4. Data Evaluasi STS (Semantic Textual Similarity) - *Opsional tapi Disarankan*
*   **Format:** Tabular (Kalimat_A, Kalimat_B, Skor_Kemiripan_1_sampai_5).
*   **Isi:** Pasangan kalimat untuk menguji apakah model paham bahwa "Toko tutup hari libur" dan "Kami tidak beroperasi pada hari Minggu" itu sama maknanya, meskipun kata-katanya beda.
*   *Sumber:* Dataset **SemEval STS Bahasa Indonesia** atau **IndoSTS** (tersedia di Hugging Face).

---

### 💡 Cara Mengemasnya (The "Aha!" Moment untuk Portofolio)

Saat Anda menjalankan eksperimen ini, buatlah sebuah *dashboard* sederhana atau *Jupyter Notebook* yang menampilkan **Visualisasi t-SNE / UMAP**:
1. Plot vektor *raw embedding* dari 100 ulasan komplain dan 100 ulasan pujian. Anda akan melihat mereka **bercampur aduk** (karena sama-sama mengandung banyak kata seperti "yang", "saya", "produk", "beli").
2. Plot vektor **EmbedFilter** dari data yang sama. Anda akan melihat **klaster yang terpisah sempurna** antara komplain dan pujian.

Itu adalah bukti visual tak terbantahkan bahwa *Unembedding Matrix Feature Lens* berhasil "mencuci" vektor dari kontaminasi *stop word*, memberikan Anda representasi fitur gratis yang langsung bisa dipakai untuk meningkatkan kualitas RAG dan Analitik UMKM Anda!
