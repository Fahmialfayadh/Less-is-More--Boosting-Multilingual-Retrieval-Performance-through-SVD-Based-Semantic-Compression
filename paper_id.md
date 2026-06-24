# Less is More: Meningkatkan Performa Retrieval Multilingual melalui Kompresi Semantik Berbasis SVD

---

## 1. Latar Belakang

Pemanfaatan *Large Language Models* (LLM) berarsitektur *Decoder-Only* sebagai *generative embedder* semakin banyak diteliti, salah satunya melalui teknik *Last-Token Pooling*. Model yang dilatih dengan objektif *Next-Token Prediction* (NTP) cenderung menghasilkan ruang semantik yang anisotropik. Kondisi ini menyebabkan nilai *cosine similarity* antar-vektor terpengaruh oleh bias frekuensi token yang tertanam selama prapelatihan, sehingga berpotensi menurunkan performa pada tugas *Information Retrieval* (IR) dan *Retrieval-Augmented Generation* (RAG).

Teknik pemurnian representasi secara *post-hoc* menawarkan pendekatan alternatif yang tidak memerlukan proses *fine-tuning* ulang. Pendekatan ini menganalisis matriks bobot proyeksi kosakata akhir (*unembedding matrix*) untuk memisahkan komponen ruang laten yang berkaitan dengan fitur semantik dari komponen yang diduga mengandung distorsi statistik. Meskipun pendekatan ini menunjukkan efektivitas pada korpus berbahasa Inggris, penerapannya pada bahasa dengan karakteristik morfologis yang berbeda — seperti bahasa Indonesia — belum banyak dieksplorasi.

---

## 2. Referensi Utama

Penelitian ini didasarkan pada metodologi yang diusulkan dalam:

> **arXiv:2606.07502** — *"Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings"* (Chen dkk.)

Paper tersebut mendalilkan bahwa matriks *unembedding* pada LLM dapat berfungsi sebagai *feature lens*. Dekomposisi matriks melalui *Singular Value Decomposition* (SVD) menunjukkan bahwa komponen dengan nilai singular tertinggi (*Head Spectrum*) cenderung berkaitan dengan token berfrekuensi tinggi (*stopwords*), sementara komponen terendah (*Tail Spectrum*) berkaitan dengan token yang jarang muncul. Berdasarkan pengamatan ini, paper tersebut mengusulkan metode **EmbFilter**, yaitu membuang kedua spektrum ekstrem (*Edge Spectrum*) dan memproyeksikan vektor pada dimensi pertengahan yang tersisa.

---

## 3. Algoritma dan Metodologi

Penelitian ini mengimplementasikan filter proyeksi ortogonal berbasis aljabar linier sesuai referensi utama, yang diaplikasikan pada model Qwen2.5-1.5B ($d = 1536$).

### 3.1. Dekomposisi Matriks *Unembedding*

Matriks *unembedding* (pada arsitektur Qwen didefinisikan sebagai `lm_head.weight`) direpresentasikan sebagai $W \in \mathbb{R}^{|V| \times d}$, di mana $|V|$ adalah ukuran kosakata dan $d$ adalah dimensi laten. Matriks $W$ didekomposisi secara penuh (*Full SVD*):

$$W = U \Sigma V_h$$

di mana $V_h \in \mathbb{R}^{d \times d}$ adalah matriks vektor singular kanan. Karena $\Sigma$ diurutkan secara *descending*, baris ke-0 pada $V_h$ berkorespondensi dengan nilai singular terbesar (*Head*), dan baris ke-$(d-1)$ berkorespondensi dengan nilai singular terkecil (*Tail*).

### 3.2. Algoritma Pergeseran Jendela Retensi Berbasis *Profiling* L2-Norm

Untuk rasio kompresi 2×, dimensi target adalah $d' = 1536 / 2 = 768$. Alih-alih mengadopsi rentang *default* dari paper rujukan untuk menentukan rentang pemotongan spektrum secara sistematis, kami mengembangkan algoritma *profiling* energi laten berbasis dekomposisi SVD. Alasan utama pergeseran rentang kompresi (*cutting window shifting*) ini berkaitan langsung dengan karakteristik morfologis bahasa Indonesia.

Algoritma penentuan jendela retensi ini bekerja melalui langkah berikut:
1. Mengekstraksi matriks `lm_head` dan melakukan *Full SVD* tanpa pemotongan.
2. Memproyeksikan vektor baris setiap token ke dalam komponen-komponen spektral, kemudian mengelompokkannya ke dalam tiga zona utama: *Head* (Top 25%), *Middle* (Mid 50%), dan *Tail* (Bot 25%).
3. Menghitung porsi kuadrat besaran energi (*L2-norm proyeksional*) untuk mendeteksi di mana informasi semantik suatu kata bersarang.

Hasil pemetaan karakteristik distribusi kelas token disajikan pada Tabel 1.

**Tabel 1 — Karakteristik Distribusi Kosakata pada Spektrum SVD Qwen2.5-1.5B**

| Zona Spektrum | Rentang Dimensi | Karakteristik Utama | Contoh Token |
|:---|:---:|:---|:---|
| **Head Spectrum** | Top 25% | Variansi tinggi antar-dokumen; didominasi penanda struktural, aksara non-Latin, dan elemen kode/HTML | `'أوضاع'` (Arab), `'낡'` (Korea), `"\n\n\n\n"`, `'<//'` |
| **Middle Spectrum** | Middle 50% | Token morfologis dan afiksasi; imbuhan, pronomina berimbuhan, sufiks pembentuk kata | `'edly'`, `'ingly'`, `'lessly'`, `' herself'`, `'为空'` |
| **Tail Spectrum** | Bottom 25% | Frekuensi tinggi, variansi rendah lintas-konteks; partikel umum di berbagai bahasa | `' they'`, `' about'`, `' its'`, `'他们'`, `'a'`, `'\n'`, `'<\|endoftext\|>'` |

Beberapa pengamatan dapat dicatat dari pola distribusi pada Tabel 1.

**Pertama,** komponen *Tail Spectrum* pada Qwen2.5 didominasi oleh *stopwords* bahasa Inggris/Mandarin, abjad tunggal, serta spasi kosong. Token-token ini muncul secara konstan di semua teks sehingga variansinya mendekati nol. Hal ini mengindikasikan bahwa dimensi *Tail* pada berbagai bahasa sebagian besar berisi *noise* anisotropik.

**Kedua,** berbeda dengan asumsi paper rujukan, *Head Spectrum* menyimpan penanda identitas bahasa (*language markers*). Membuang komponen ini berpotensi mereduksi kemampuan model dalam membedakan konteks lintas-bahasa.

**Ketiga,** dan menjadi landasan algoritma pergeseran: pengujian analitik terhadap token-token imbuhan (afiksasi) bahasa Indonesia menunjukkan bahwa energi varians (*L2-norm*) mereka terkonsentrasi di kombinasi zona *Head* dan *Middle*.

**Tabel 2 — Distribusi L2-Norm Token Afiksasi Bahasa Indonesia pada Spektrum Qwen2.5-1.5B**

| Token Imbuhan | *Head Spectrum* (Top 25%) | *Middle Spectrum* (Mid 50%) | *Tail Spectrum* (Bot 25%) |
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

Berdasarkan Tabel 2, token afiksasi kritis mengakumulasi **75% hingga 80%** total energi latennya pada gabungan *Head* dan *Middle Spectrum*.

### 3.2.1. Analisis Kurva Energi (Plot "Smoking Gun")

Untuk mengonfirmasi perbedaan lokalisasi energi ini secara visual, kami memplot distribusi energi rata-rata SVD (L1-norm) untuk token afiksasi bahasa Indonesia dan stopwords bahasa Inggris di seluruh spektrum 1536 dimensi. Kurva yang dihasilkan, yang telah dihaluskan menggunakan jendela bergerak berukuran 64, disajikan pada Gambar 1.

![Gambar 1 — Kurva Energi SVD: Afiks Indonesia vs. Stopwords Inggris](data/energy_curve.png)

Sebagaimana diilustrasikan pada Gambar 1, pola sebaran energi antara stopwords Inggris dan afiks Indonesia menunjukkan perbedaan yang sangat kontras:
1. **Stopwords Inggris:** Kurva energi memuncak di spektrum Middle dan Tail, namun menurun drastis di spektrum Head. Hal ini selaras dengan temuan Chen dkk., yang mengindikasikan bahwa dimensi singular tertinggi (Head) didominasi oleh noise struktural dan frekuensi pada bahasa Inggris.
2. **Afiks Indonesia:** Sebaliknya, kurva afiksasi bahasa Indonesia memuncak sangat tajam di spektrum Head (dimensi 0-384), khususnya pada 200 dimensi komponen pertama.

Plot ini menjadi bukti nyata (*smoking gun*) mengapa filter `English-Middle` (yang membuang 25% spektrum pertama) menurunkan performa representasi semantik bahasa Indonesia. Pembuangan tersebut secara langsung mengeliminasi wilayah dengan konsentrasi energi afiksasi tertinggi.

### 3.2.2. Relevansi dengan Bahasa Aglutinatif Lainnya

Konsentrasi energi partikel morfologis (afiks) pada komponen variansi tinggi dalam spektrum SVD merupakan karakteristik struktural bawaan dari bahasa aglutinatif. Berbeda dengan bahasa isolatif atau fleksional (seperti bahasa Inggris) yang banyak menggunakan kata terpisah untuk fungsi sintaksis, bahasa aglutinatif (seperti bahasa Turki, Finlandia, Hungaria, Korea, dan Jepang) membangun hubungan tata bahasa melalui penggabungan morfem terikat (prefiks, sufiks, infiks, konfiks) secara berantai ke kata dasar.

Dalam model LLM multilingual yang dilatih menggunakan kosakata bersama (*joint vocabulary*), morfem terikat krusial ini memiliki frekuensi dokumen yang sangat tinggi dan variansi kontekstual yang rendah, sehingga diproyeksikan sangat kuat pada komponen singular tertinggi (*Head Spectrum*). Oleh karena itu, asumsi filter *English-Middle* yang menggeneralisasikan *Head Spectrum* semata-mata sebagai noise frekuensi stopwords adalah cacat secara metodologis untuk bahasa-bahasa aglutinatif. Praktisi yang bekerja dengan korpus bahasa aglutinatif perlu mempertahankan spektrum *Head* guna mencegah hilangnya fitur morfologis esensial selama kompresi semantik *post-hoc*.

**Sistematisasi Algoritma Pergeseran:**

Pendekatan `English-Middle` dari paper rujukan menghapus 25% spektrum pertama (*Head*). Berdasarkan Tabel 2, algoritma pemotongan tersebut **menghilangkan 28% hingga 38% energi semantik** (*L2-norm*) dari partikel sintaksis bahasa Indonesia seperti awalan `meng-`, `ter-`, dan akhiran `-nya`, `-lah`.

Berdasarkan bukti empiris ini, algoritma yang kami usulkan menggeser jendela retensi (kompresi 50%) ke rentang dimensi **0 hingga 768**, yang mencakup seluruh *Head Spectrum* dan separuh atas *Middle Spectrum* (`Indonesian-Retention`). Pergeseran ini dirancang untuk mempertahankan fitur morfologis, sekaligus mengeliminasi komponen *noise* di *Tail Spectrum*.

Berdasarkan algoritma tersebut, empat konfigurasi matriks proyeksi $V_{sub} \in \mathbb{R}^{d' \times d}$ dievaluasi secara empiris:

| # | Konfigurasi | Deskripsi | Rentang Indeks $V_{sub}$ |
|---|---|---|---|
| 1 | **Baseline** | Tanpa pemotongan; vektor asli $x \in \mathbb{R}^{1536}$ digunakan langsung | — |
| 2 | **English-Middle** | Membuang 25% *Head* dan 25% *Tail* (pendekatan paper rujukan) | Indeks 384–1151 |
| 3 | **Indonesian-Retention** | Mempertahankan 50% spektrum *Head* dan *Middle* (berdasarkan algoritma *profiling*) | Indeks 0–767 |
| 4 | **Tail-Retention** | Mempertahankan 50% spektrum *Tail* | Indeks 768–1535 |

### 3.3. Proyeksi Representasi

Setiap kalimat $s$ dikodekan menjadi representasi awal $x$ menggunakan *Last-Token Pooling* dengan templat instruksi (`PromptEOL`). Vektor terfilter $x'$ diperoleh melalui:

$$x' = x V_{sub}^T$$

Vektor $x'$ berdimensi 768 selanjutnya digunakan untuk komputasi *cosine similarity*.

### 3.4. Pengaturan Evaluasi

**Dataset:**

| Tugas | Dataset | Split | Keterangan |
|---|---|---|---|
| *Retrieval* (RAG) | MIRACL — Bahasa Indonesia | dev | 500 kueri acak; 4.543 dok. positif + 5.000 dok. negatif sampel |
| STS | STS-B — LazarusNLP | test | 500 pasang kalimat sampel |

### 3.5. Metrik Evaluasi

Kinerja ruang semantik hasil proyeksi diukur menggunakan tiga metrik standar dalam domain penelusuran informasi:
1. **NDCG@10 (*Normalized Discounted Cumulative Gain* pada 10 kandidat teratas):** Mengukur efektivitas sistem *retrieval* dengan memperhitungkan posisi dokumen relevan di daftar hasil pencarian. NDCG memberikan penalti jika dokumen relevan muncul di peringkat bawah, menjadikannya metrik paling krusial untuk sistem RAG yang sensitif terhadap urutan konteks.
2. **Recall@100:** Mengukur persentase dokumen relevan yang berhasil diambil oleh sistem dalam 100 hasil pencarian teratas. Metrik ini mengindikasikan seberapa baik vektor mampu menemukan dokumen yang tepat di tengah lautan korpus yang luas tanpa terlalu mempedulikan urutan pastinya.
3. **Korelasi Spearman:** Digunakan spesifik untuk tugas *Semantic Textual Similarity* (STS) guna mengukur korelasi monotonik berperingkat antara skor *cosine similarity* model dengan skor relevansi referensi dari penilaian manusia.

**Pengujian Statistik:** Tingkat signifikansi antar-metode dievaluasi menggunakan *Paired Student's t-test* dengan batas $\alpha = 0.01$ memanfaatkan pustaka metrik `ranx`.

---

## 4. Hasil Eksperimen

Tabel 3 merangkum rata-rata performa dari masing-masing konfigurasi yang diuji.

**Tabel 3 — Hasil Evaluasi Metrik Lintas Bahasa (MIRACL Indonesia)**

| Konfigurasi | Dimensi | NDCG@10 | Recall@100 |
|:---|:---:|:---:|:---:|
| Baseline | 1536 | 0.1592 | 0.4211 |
| English-Middle | 768 | 0.2333 | 0.6034 |
| Indonesian-Retention | 768 | **0.2900** | **0.6535** |
| Tail-Retention | 768 | 0.0808 | 0.2485 |

### 4.1. Uji Signifikansi Statistik

Laporan berikut dihasilkan oleh uji *Paired Student's t-test* dengan $\alpha = 0.01$:

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

> **Keterangan superskrip:** Notasi huruf di samping angka menunjukkan bahwa model pada baris tersebut memiliki skor yang secara statistik signifikan melampaui model dengan indeks yang disebutkan.
> Contoh: `0.290ᵃᵇᵈ` berarti konfigurasi `Indonesian-Retention` (c) mengungguli `Baseline` (a), `English-Middle` (b), dan `Tail-Retention` (d) secara absolut dan meyakinkan.

---

## 5. Pembahasan

Hasil eksperimen pada Tabel 3 menunjukkan pola yang bertentangan dengan asumsi paper rujukan ketika metode tersebut diterapkan pada LLM multilingual (Qwen2.5) untuk tugas korpus berbahasa Indonesia.

Pada paper rujukan yang dievaluasi menggunakan bahasa Inggris, konfigurasi `English-Middle` (pembuangan *Head* dan *Tail*) diasumsikan optimal karena *Head Spectrum* dinilai hanya berisi distorsi dari *stopwords* berfrekuensi tinggi. Namun, dalam eksperimen ini, `Indonesian-Retention` (rentang `0:768` yang justru **mempertahankan** *Head*) mencapai NDCG@10 sebesar 0.2900, melampaui `English-Middle` secara signifikan ($p < 0.01$).

Hasil ini konsisten dengan prediksi algoritma *profiling* L2-Norm yang dijabarkan pada Bab 3.2. Mempertahankan morfologi inti bahasa Indonesia — yang terkonsentrasi di zona *Head* dan *Middle* — menghasilkan representasi leksikal yang lebih utuh. Sebaliknya, konfigurasi `Tail-Retention` yang hanya mencapai NDCG@10 sebesar 0.0808 mengonfirmasi bahwa zona *Tail* didominasi oleh komponen *noise* anisotropik bervariansi rendah.

Perlu dicatat bahwa Chen dkk. mendefinisikan *Edge Spectrum* berdasarkan distribusi frekuensi teks berbahasa Inggris. Eksperimen ini menunjukkan bahwa mengadopsi jendela retensi yang sama untuk bahasa Indonesia berdampak negatif signifikan terhadap keutuhan representasi linguistiknya.

Secara keseluruhan, temuan ini mengindikasikan bahwa strategi pemotongan dimensi yang dioptimalkan untuk korpus berbahasa Inggris tidak dapat diterapkan secara langsung pada LLM multilingual untuk tugas *retrieval* berbahasa Indonesia. Eksperimen lanjutan — mencakup variasi model, rentang pemotongan, dan bahasa target lainnya — diperlukan untuk memvalidasi temuan ini sebelum dapat digeneralisasi.

---

## 6. Kesimpulan dan Kontribusi Praktis

Penelitian ini mengusulkan kerangka adaptasi praktis bagi metode *EmbFilter* yang disesuaikan dengan karakteristik linguistik bahasa Indonesia, dengan kontribusi terhadap efisiensi arsitektur *Retrieval-Augmented Generation* (RAG) pada sistem produksi.

Melalui proyeksi ortogonal pada matriks *unembedding* dengan jendela retensi `0:768` (*Head* hingga *Middle Spectrum*), penelitian ini menunjukkan tiga kontribusi utama:
1. **Reduksi Beban Infrastruktur (*Vector Database*):** Dimensi representasi semantik dikompresi sebesar **50%** (dari 1536 menjadi 768 dimensi), yang secara langsung mengurangi kebutuhan alokasi memori penyimpanan dan komputasi *vector search*.
2. **Peningkatan Akurasi (*Retrieval Performance*):** Meskipun reduksi dimensi umumnya berkorelasi dengan degradasi akurasi, penyaringan komponen *Tail Spectrum* dengan tetap mempertahankan fitur afiksasi bahasa Indonesia di rentang `0:768` mampu meningkatkan performa NDCG@10 sebesar **+82%** (dari 0.1592 menjadi 0.2900).
3. **Alat Diagnostik Prediktif Lintas-Model:** Algoritma L2-Norm Profiling yang dikembangkan dalam penelitian ini dapat berfungsi sebagai alat diagnostik prediktif. Sebelum mengimplementasikan reduksi pada model LLM berskala besar (misalnya 7B atau 70B parameter), praktisi dapat mengekstrak `lm_head` dan menguji sebaran energi L2-Norm dari daftar kata afiksasi. Apabila kurva energi tetap dominan di area *Head* dan *Middle*, hal tersebut memberikan justifikasi matematis bagi strategi kompresi ini tanpa memerlukan *benchmark* RAG berskala penuh.

Adaptasi *Indonesian-Retention* ini menawarkan solusi kompresi semantik yang efisien sekaligus kerangka validasi analitis untuk memaksimalkan utilitas LLM multilingual sebagai *generative embedder*, tanpa memerlukan investasi komputasi tambahan seperti *fine-tuning*.

### 6.1. Validasi Skalabilitas Lintas-Model (Studi Kasus 7B)

Untuk menguji generalisasi alat diagnostik tersebut, kami menerapkan *L2-Norm Profiling* pada model berskala lebih besar, yakni **Qwen2.5-7B** (3.584 dimensi laten). Hasil profil (Tabel 4) menunjukkan bahwa pola retensi energi morfologis di spektrum *Head* bukan merupakan anomali yang terbatas pada model kecil, melainkan karakteristik linguistik yang konsisten di berbagai skala arsitektur.

**Tabel 4 — Distribusi Energi L2-Norm Imbuhan pada Qwen2.5-7B**

| Token Afiks | Head (0-25%) | Middle (25-75%) | Tail (75-100%) |
|:---|:---:|:---:|:---:|
| 'nya' | 33.6% | 45.0% | 21.4% |
| 'lah' | 38.4% | 46.3% | 15.3% |
| 'kan' | 37.4% | 49.8% | 12.9% |
| 'pun' | 38.7% | 49.0% | 12.3% |
| ' meng'| 32.6% | 50.5% | 16.9% |
| ' ber' | 29.9% | 55.4% | 14.7% |
| ' ter' | 38.1% | 50.7% | 11.1% |

Data pada Tabel 4 mengonfirmasi bahwa rentang *Head* dan *Middle* secara konsisten mengakumulasi sebagian besar (**~75% hingga 88%**) total energi semantik dari imbuhan bahasa Indonesia, termasuk pada model berskala 7B. Sebaliknya, porsi energi pada *Tail* tetap berada pada proporsi minimal.

Temuan ini mendukung kesimpulan bahwa praktik membuang spektrum *Head* — yang efektif untuk mereduksi distorsi *stopwords* pada bahasa Inggris — perlu dievaluasi ulang sebelum diterapkan pada korpus bahasa Indonesia. Pada bahasa dengan karakteristik aglutinasi dan afiksasi yang kaya, spektrum *Head* berperan penting dalam merepresentasikan struktur leksikal secara utuh.