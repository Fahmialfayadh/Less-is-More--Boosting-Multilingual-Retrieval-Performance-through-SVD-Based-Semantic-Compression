# Laporan Analisis Signifikansi Statistik: Evaluasi Metode *Unembedded* pada Model Dense Retriever MLM

Dokumen ini merangkum hasil uji signifikansi statistik (Paired Student's t-test, $\alpha = 0.01$) yang dilakukan terhadap 17 variasi modifikasi ruang vektor (*vector space manipulation*) menggunakan metode *Unembedded* (penghapusan *edge spectrum* $k$, dan kompresi PCA $\tau$). Evaluasi dilakukan menggunakan *framework* `ranx` pada korpus RAG berbahasa Indonesia (MTEB MIRACL).

## 1. Ringkasan Temuan Statistik
Berdasarkan hasil pengujian pada tabel `ranx.compare` terhadap metrik utama (NDCG@10 dan Recall@100), ditemukan fakta-fakta objektif berikut:

1. **Penurunan Signifikan Akibat Kompresi ($\tau$)**
   - Varian **Baseline** (`Model a: k=0, tau=None`) meraih NDCG@10 sebesar **0.720**.
   - Seluruh model yang mengalami kompresi ruang vektor (nilai $\tau = 128, 256, 512$) menunjukkan superskrip signifikansi di bawah *Baseline*. Ini mengindikasikan bahwa kompresi dimensi secara mutlak menurunkan performa *retrieval* secara drastis (hingga ~0.539 pada $\tau=128$) dengan $p < 0.01$. 
   - Penurunan ini bukan kebetulan matematis (*statistical fluke*), melainkan kerugian nyata pada menurunkan performa retrieval secara drastis (hingga ~0.539 pada 
𝜏
=
128
τ=128) dengan 
𝑝
<
0.01
p<0.01ruang fitur.

2. **Absennya Dampak Signifikan pada Penghapusan *Noise Spectrum* ($k$)**
   - Varian yang hanya menghapus *edge spectrum* atas tanpa melakukan kompresi (`Model m, n, p, q` untuk $k=20, 10, 1, 5$ pada $\tau=None$) **tidak menunjukkan perbedaan yang signifikan** secara statistik melawan *Baseline*. 
   - Skor NDCG@10 mereka stagnan di kisaran **0.718 - 0.720**. Hal ini membuktikan bahwa menghapus komponen *singular vector* teratas pada model MLM modern tidak memberikan perbaikan (maupun kerusakan) yang berarti pada performa Information Retrieval.

## 2. Analisis Kritis: Kegagalan Premis *Unembedded* pada RAG
Makalah asli *Unembedded* berhipotesis bahwa model representasi leksikal sering menderita sindrom *anisotropy* (distribusi vektor menyempit dalam bentuk kerucut). Penghapusan spektrum *singular* atas ($k$) dan reduksi dimensi ($\tau$) diklaim mampu membuang komponen "noise" dominan sehingga menyebarkan vektor menjadi lebih isotropik dan representatif. 

Namun, pengujian pada sistem RAG lintas dokumen (MIRACL) dengan *encoder* berbasis *Masked Language Model* (MLM) seperti `multilingual-e5-base` (arsitektur XLM-RoBERTa / keluarga IndoBERT) membantah universalitas klaim tersebut:

1. **"Noise" dalam STS adalah "Signal" dalam IR**
   - Reduksi *anisotropy* mungkin terbukti menaikkan metrik korelasi linier pada *Semantic Textual Similarity* (STS) yang bersifat simetris (kalimat vs kalimat). 
   - Namun, dalam tugas *Information Retrieval* (IR) asimetris (kueri pendek vs dokumen panjang), komponen dominan yang dilabeli sebagai *noise* oleh metode *Unembedded* faktanya mengandung informasi diskriminatif krusial. Informasi ini (seperti entitas leksikal langka, kecocokan persis *token*, atau pola frekuensi *term*) sangat penting bagi Dense Retriever untuk memisahkan satu dokumen relevan dari 20.000 dokumen distraktor.
   
2. **Keterbatasan pada Model Berbasis MLM Modern**
   - Arsitektur MLM modern yang dilatih dengan *contrastive loss* (seperti e5 atau model berbasis IndoBERT terkini) telah belajar mendistribusikan representasinya secara lebih terukur dibanding generasi awal representasi densa. 
   - Manipulasi ruang vektor linier (SVD/PCA) secara *post-hoc* paska-pelatihan tidak dapat memperbaiki *alignment* ruang kueri-dokumen. Sebaliknya, kompresi paksa ($\tau$) justru menghancurkan resolusi ruang representasi, yang dibuktikan secara telak oleh anjloknya NDCG@10 secara signifikan ($p < 0.01$).

## 3. Kesimpulan Objektif
Berdasarkan bukti empiris dan statistik, metodologi *Unembedded* tidak dapat diterapkan sebagai solusi generalisasi pada arsitektur representasi Dense Retriever berbasis MLM modern (seperti IndoBERT atau e5) untuk kasus penggunaan RAG. Penyesuaian ruang vektor *post-hoc* yang bertujuan untuk *anisotropy reduction* justru berisiko tinggi merusak fitur diskriminatif yang esensial bagi mekanisme temu-kembali informasi (*information retrieval*).
