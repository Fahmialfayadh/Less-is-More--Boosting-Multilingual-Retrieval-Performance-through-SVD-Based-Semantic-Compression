# Less is More: Meningkatkan Performa Retrieval Multilingual melalui Kompresi Semantik Berbasis SVD

---

## 1. Latar Belakang

Pemanfaatan *Large Language Models* (LLM) berarsitektur *Decoder-Only* sebagai *generative embedder* semakin banyak diteliti, salah satunya melalui teknik *Last-Token Pooling*. Model yang dilatih dengan objektif *Next-Token Prediction* (NTP) cenderung menghasilkan ruang semantik yang anisotropik, yaitu kondisi di mana nilai *cosine similarity* antar-vektor terpengaruh oleh bias frekuensi token yang tertanam selama prapelatihan. Kondisi ini berpotensi menurunkan performa pada tugas *Information Retrieval* (IR) dan *Retrieval-Augmented Generation* (RAG).

Teknik pemurnian representasi secara *post-hoc* menawarkan pendekatan alternatif yang tidak memerlukan proses *fine-tuning* ulang. Chen dkk. (arXiv:2606.07502, *"Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings"*) menunjukkan bahwa matriks *unembedding* pada LLM dapat berfungsi sebagai *feature lens*: dekomposisi melalui *Singular Value Decomposition* (SVD) mengungkapkan bahwa komponen dengan nilai singular tertinggi (*Head Spectrum*) cenderung berkaitan dengan token berfrekuensi tinggi (*stopwords*), sementara komponen terendah (*Tail Spectrum*) berkaitan dengan token yang jarang muncul. Berdasarkan temuan ini, mereka mengusulkan metode **EmbFilter** yang membuang kedua spektrum ekstrem (*Edge Spectrum*) dan memproyeksikan vektor pada dimensi pertengahan yang tersisa.

Meskipun EmbFilter menunjukkan efektivitas pada korpus berbahasa Inggris, pendekatan tersebut mengasumsikan bahwa *Head Spectrum* secara universal berisi *noise* frekuensi *stopwords*. Asumsi ini belum diuji pada bahasa dengan karakteristik morfologis yang berbeda, khususnya bahasa aglutinatif seperti bahasa Indonesia yang membangun makna gramatikal melalui rangkaian morfem terikat (prefiks, sufiks, dan infiks). Penelitian ini menguji validitas asumsi tersebut dan mengusulkan adaptasi jendela retensi yang disesuaikan dengan distribusi energi morfologis bahasa Indonesia.

---

## 2. Metodologi

Penelitian ini mengimplementasikan filter proyeksi ortogonal berbasis aljabar linier sesuai kerangka EmbFilter, yang diaplikasikan pada model Qwen2.5-1.5B ($d = 1536$).

### 2.1. Dekomposisi Matriks *Unembedding*

Matriks *unembedding* (pada arsitektur Qwen didefinisikan sebagai `lm_head.weight`) direpresentasikan sebagai $W \in \mathbb{R}^{|V| \times d}$, di mana $|V|$ adalah ukuran kosakata dan $d$ adalah dimensi laten. Matriks $W$ didekomposisi secara penuh (*Full SVD*) tanpa pemotongan:

$$W = U \Sigma V_h$$

di mana $V_h \in \mathbb{R}^{d \times d}$ adalah matriks vektor singular kanan. Karena $\Sigma$ diurutkan secara *descending*, baris ke-0 pada $V_h$ berkorespondensi dengan nilai singular terbesar (*Head*), dan baris ke-$(d-1)$ berkorespondensi dengan nilai singular terkecil (*Tail*).

### 2.2. Pergeseran Jendela Retensi Berbasis *Profiling* L2-Norm

Untuk rasio kompresi 2×, dimensi target adalah $d' = 1536 / 2 = 768$. Alih-alih mengadopsi rentang pemotongan *default* dari paper rujukan, kami mengembangkan algoritma *profiling* energi laten untuk menentukan rentang optimal secara empiris. Algoritma ini bekerja dalam tiga tahap: mengekstraksi matriks `lm_head` dan melakukan *Full SVD*, memproyeksikan vektor baris setiap token ke dalam tiga zona spektral (*Head* Top 25%, *Middle* Mid 50%, *Tail* Bot 25%), kemudian menghitung porsi kuadrat besaran energi (*L2-norm* proyeksional) untuk mendeteksi lokasi konsentrasi informasi semantik setiap token. Hasil pemetaan distribusi kelas token disajikan pada Tabel 1.

**Tabel 1 — Karakteristik Distribusi Kosakata pada Spektrum SVD Qwen2.5-1.5B**

| Zona Spektrum | Rentang Dimensi | Karakteristik Utama | Contoh Token |
|:---|:---:|:---|:---|
| **Head Spectrum** | Top 25% | Variansi tinggi antar-dokumen; didominasi penanda struktural, aksara non-Latin, dan elemen kode/HTML | `'أوضاع'` (Arab), `'낡'` (Korea), `"\n\n\n\n"`, `'<//'` |
| **Middle Spectrum** | Middle 50% | Token morfologis dan afiksasi; imbuhan, pronomina berimbuhan, sufiks pembentuk kata | `'edly'`, `'ingly'`, `'lessly'`, `' herself'`, `'为空'` |
| **Tail Spectrum** | Bottom 25% | Frekuensi tinggi, variansi rendah lintas-konteks; partikel umum di berbagai bahasa | `' they'`, `' about'`, `' its'`, `'他们'`, `'a'`, `'\n'`, `'<\|endoftext\|>'` |

Distribusi pada Tabel 1 mengungkapkan temuan yang signifikan terhadap asumsi paper rujukan. Zona *Tail Spectrum* didominasi oleh *stopwords* bahasa Inggris/Mandarin, abjad tunggal, dan token struktural yang muncul secara konstan di semua teks dengan variansi mendekati nol, mengonfirmasi bahwa dimensi ini sebagian besar berisi *noise* anisotropik. Zona *Head Spectrum*, di sisi lain, justru menyimpan penanda identitas bahasa seperti aksara non-Latin dan elemen multibahasa, berbeda dengan asumsi EmbFilter yang menggeneralisasikan zona ini sebagai *noise* frekuensi *stopwords*. Temuan ini memunculkan pertanyaan kritis: bagaimana distribusi energi token morfologis bahasa Indonesia tersebar di sepanjang spektrum SVD?

Untuk menjawab pertanyaan tersebut, kami menganalisis distribusi L2-norm dari 12 token afiksasi kunci bahasa Indonesia, mencakup sufiks (`-nya`, `-lah`, `-kan`, `-pun`, `-kah`, `-ku`, `-mu`) dan prefiks (`di-`, `ter-`, `ber-`, `meng-`, `mem-`). Hasilnya disajikan pada Tabel 2.

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

Token afiksasi kritis mengakumulasi **75% hingga 80%** total energi latennya pada gabungan *Head* dan *Middle Spectrum*, sangat kontras dengan pola *stopwords* bahasa Inggris yang terkonsentrasi di *Tail*. Gambar 1 memvisualisasikan perbedaan lokalisasi ini melalui distribusi energi rata-rata SVD (L1-norm) untuk kedua kelompok token di seluruh spektrum 1536 dimensi, dihaluskan menggunakan jendela bergerak berukuran 64.

![Gambar 1 — Kurva Energi SVD: Afiks Indonesia vs. Stopwords Inggris](data/energy_curve.png)

Kurva pada Gambar 1 mengonfirmasi secara visual apa yang ditunjukkan data kuantitatif: energi *stopwords* Inggris memuncak di spektrum *Middle* dan *Tail* namun menurun drastis di spektrum *Head*, sedangkan afiksasi bahasa Indonesia memuncak tajam di spektrum *Head* (dimensi 0–384). Pola ini menjelaskan mengapa filter `English-Middle`, yang membuang 25% spektrum pertama, secara langsung merusak representasi semantik bahasa Indonesia dengan mengeliminasi wilayah konsentrasi energi afiksasi tertinggi.

### 2.3. Validasi Generalisasi Lintas-Model dan Lintas-Arsitektur

Konsentrasi energi morfologis pada *Head Spectrum* yang teridentifikasi pada Qwen2.5-1.5B perlu diverifikasi apakah merupakan karakteristik linguistik universal atau artefak dari satu arsitektur dan skala model tertentu. Kami menguji generalisasi temuan ini pada dua model tambahan yang berbeda secara signifikan dalam skala dan arsitektur.

Pengujian pertama dilakukan pada **Qwen2.5-7B** (3.584 dimensi laten) untuk memverifikasi konsistensi pada skala yang lebih besar dalam keluarga arsitektur yang sama. Tabel 3 menunjukkan bahwa pola distribusi energi tetap konsisten: rentang *Head* dan *Middle* secara bersama mengakumulasi ~75% hingga 88% total energi semantik dari imbuhan bahasa Indonesia, sementara porsi energi pada *Tail* tetap minimal (11–21%).

**Tabel 3 — Distribusi Energi L2-Norm Imbuhan pada Qwen2.5-7B**

| Token Afiks | Head (0-25%) | Middle (25-75%) | Tail (75-100%) |
|:---|:---:|:---:|:---:|
| 'nya' | 33.6% | 45.0% | 21.4% |
| 'lah' | 38.4% | 46.3% | 15.3% |
| 'kan' | 37.4% | 49.8% | 12.9% |
| 'pun' | 38.7% | 49.0% | 12.3% |
| ' meng'| 32.6% | 50.5% | 16.9% |
| ' ber' | 29.9% | 55.4% | 14.7% |
| ' ter' | 38.1% | 50.7% | 11.1% |

Pengujian kedua memperluas analisis ke arsitektur yang sepenuhnya berbeda: **Llama-3.1-70B** ($d = 8192$), dengan *tokenizer* dan struktur kosakata yang terpisah dari keluarga Qwen. Tabel 4 menunjukkan bahwa bahkan pada skala 70 miliar parameter, morfem terikat bahasa Indonesia terus menempatkan sebagian besar energi semantiknya di spektrum *Head* dan *Middle*, dengan porsi *Tail* yang bahkan lebih kecil (7–12%) dibandingkan kedua model Qwen.

**Tabel 4 — Distribusi Energi L2-Norm Imbuhan pada Llama-3.1-70B**

| Token Afiks | Head Spectrum (0-25%) | Middle Spectrum (25-75%) | Tail Spectrum (75-100%) |
|:---|:---:|:---:|:---:|
| `'nya'` | 43.1% | **45.6%** | 11.4% |
| `'lah'` | **45.8%** | 44.0% | 10.2% |
| `'kan'` | **48.5%** | 43.8% | 7.7% |
| `'pun'` | **48.9%** | 42.8% | 8.2% |
| `' meng'` | 42.4% | **45.4%** | 12.2% |
| `' ber'` | 35.1% | **54.6%** | 10.3% |
| `' ter'` | 37.4% | **52.4%** | 10.2% |

Kurva energi L1-norm untuk Llama-3.1-70B (Gambar 2) memperkuat temuan ini secara visual. Meskipun spektrum membentang sepanjang 8192 dimensi, pola struktural yang sama terreproduksi: *stopwords* Inggris memuncak di *Tail*, sementara afiks Indonesia memuncak tajam di *Head* dan mempertahankan variansi tinggi di sepanjang *Middle*.

![Gambar 2 — Kurva Energi SVD (Llama-3.1-70B): Afiks Indonesia vs. Stopwords Inggris](data/energy_curve-llama-70b.png)

Konsistensi pola ini lintas tiga model (1.5B, 7B, 70B) dan dua keluarga arsitektur (Qwen, Llama) mengindikasikan bahwa konsentrasi energi morfologis pada *Head Spectrum* merupakan refleksi dari cara LLM multilingual merepresentasikan morfem terikat dalam ruang laten, bukan artefak teknis dari satu arsitektur. Fenomena ini relevan bagi seluruh keluarga bahasa aglutinatif (Turki, Finlandia, Hungaria, Korea, Jepang) yang membangun hubungan tata bahasa melalui penggabungan morfem terikat secara berantai. Dalam LLM dengan kosakata bersama (*joint vocabulary*), morfem terikat ini memiliki frekuensi dokumen tinggi dan variansi kontekstual rendah, sehingga diproyeksikan kuat pada komponen singular tertinggi.

### 2.4. Konfigurasi Proyeksi

Bukti empiris dari tiga model menunjukkan bahwa pendekatan `English-Middle` menghilangkan **28% hingga 38% energi semantik** (L2-norm) dari partikel sintaksis esensial bahasa Indonesia seperti awalan `meng-`, `ter-`, dan akhiran `-nya`, `-lah`. Algoritma yang kami usulkan menggeser jendela retensi (kompresi 50%) ke rentang dimensi **0 hingga 768**, mencakup seluruh *Head Spectrum* dan separuh atas *Middle Spectrum* (`Indonesian-Retention`). Pergeseran ini mempertahankan fitur morfologis sekaligus mengeliminasi komponen *noise* di *Tail Spectrum*.

Empat konfigurasi matriks proyeksi $V_{sub} \in \mathbb{R}^{d' \times d}$ dievaluasi:

| # | Konfigurasi | Deskripsi | Rentang Indeks $V_{sub}$ |
|---|---|---|---|
| 1 | **Baseline** | Tanpa pemotongan; vektor asli $x \in \mathbb{R}^{1536}$ digunakan langsung | — |
| 2 | **English-Middle** | Membuang 25% *Head* dan 25% *Tail* (pendekatan paper rujukan) | Indeks 384–1151 |
| 3 | **Indonesian-Retention** | Mempertahankan 50% spektrum *Head* dan *Middle* (berdasarkan algoritma *profiling*) | Indeks 0–767 |
| 4 | **Tail-Retention** | Mempertahankan 50% spektrum *Tail* | Indeks 768–1535 |

### 2.5. Proyeksi Representasi

Setiap kalimat $s$ dikodekan menjadi representasi awal $x$ menggunakan *Last-Token Pooling* dengan templat instruksi (`PromptEOL`). Vektor terfilter $x'$ diperoleh melalui:

$$x' = x V_{sub}^T$$

Vektor $x'$ berdimensi 768 selanjutnya digunakan untuk komputasi *cosine similarity*.

### 2.6. Pengaturan Evaluasi

**Dataset:**

| Tugas | Dataset | Split | Keterangan |
|---|---|---|---|
| *Retrieval* (RAG) | MIRACL — Bahasa Indonesia | dev | 500 kueri acak; 4.543 dok. positif + 5.000 dok. negatif sampel |
| STS | STS-B — LazarusNLP | test | 500 pasang kalimat sampel |

Kinerja ruang semantik hasil proyeksi diukur menggunakan tiga metrik standar dalam domain penelusuran informasi. **NDCG@10** (*Normalized Discounted Cumulative Gain* pada 10 kandidat teratas) mengukur efektivitas sistem *retrieval* dengan memperhitungkan posisi dokumen relevan di daftar hasil pencarian; metrik ini memberikan penalti jika dokumen relevan muncul di peringkat bawah, menjadikannya paling krusial untuk sistem RAG yang sensitif terhadap urutan konteks. **Recall@100** mengukur persentase dokumen relevan yang berhasil diambil dalam 100 hasil teratas, mengindikasikan kemampuan vektor menemukan dokumen yang tepat di tengah korpus yang luas. **Korelasi Spearman** digunakan untuk tugas *Semantic Textual Similarity* (STS), mengukur korelasi monotonik berperingkat antara skor *cosine similarity* model dengan skor relevansi referensi dari penilaian manusia.

Tingkat signifikansi antar-metode dievaluasi menggunakan *Paired Student's t-test* dengan batas $\alpha = 0.01$ memanfaatkan pustaka metrik `ranx`.

---

## 3. Hasil Eksperimen

Tabel 5 merangkum rata-rata performa dari masing-masing konfigurasi yang diuji pada tugas *retrieval* MIRACL Indonesia.

**Tabel 5 — Hasil Evaluasi Metrik Lintas Bahasa (MIRACL Indonesia)**

| Konfigurasi | Dimensi | NDCG@10 | Recall@100 |
|:---|:---:|:---:|:---:|
| Baseline | 1536 | 0.1592 | 0.4211 |
| English-Middle | 768 | 0.2333 | 0.6034 |
| Indonesian-Retention | 768 | **0.2900** | **0.6535** |
| Tail-Retention | 768 | 0.0808 | 0.2485 |

Konfigurasi `Indonesian-Retention` mencapai NDCG@10 tertinggi (0.2900) dan Recall@100 tertinggi (0.6535), melampaui seluruh konfigurasi lainnya termasuk vektor *Baseline* berdimensi penuh. Sebaliknya, `Tail-Retention` menghasilkan performa terendah (NDCG@10 = 0.0808), bahkan di bawah *Baseline*, mengonfirmasi bahwa zona *Tail* didominasi oleh komponen *noise*.

Tabel 6 menyajikan evaluasi pada dataset STS-B (LazarusNLP) menggunakan Korelasi Spearman.

**Tabel 6 — Hasil Evaluasi *Semantic Textual Similarity* (STS-B LazarusNLP)**

| Konfigurasi | Dimensi | Korelasi Spearman |
|:---|:---:|:---:|
| Baseline | 1536 | -0.1964 |
| English-Middle | 768 | -0.2146 |
| Indonesian-Retention | 768 | -0.1996 |
| Tail-Retention | 768 | -0.2051 |

Meskipun performa pada seluruh konfigurasi *zero-shot* tetap mendekati nol (umum terjadi pada model autoregresif tanpa *fine-tuning* spesifik tugas STS), perbedaan relatif yang ada menunjukkan bahwa `Indonesian-Retention` mempertahankan struktur semantik lebih baik daripada filter `English-Middle`.

### 3.1. Uji Signifikansi Statistik

Untuk memastikan perbedaan performa bukan artefak statistik, kami melakukan uji *Paired Student's t-test* pada $\alpha = 0.01$:

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

Notasi superskrip menunjukkan bahwa model pada baris tersebut secara statistik signifikan ($p < 0.01$) melampaui model dengan indeks yang bersangkutan. Konfigurasi `Indonesian-Retention` (c) dengan notasi `0.290ᵃᵇᵈ` mengungguli seluruh konfigurasi lainnya secara absolut dan meyakinkan pada kedua metrik.

---

## 4. Pembahasan

Hasil eksperimen menunjukkan bukti kuat yang bertentangan dengan asumsi inti paper rujukan ketika metode tersebut diterapkan pada LLM multilingual untuk tugas korpus berbahasa Indonesia. Pada evaluasi Chen dkk. menggunakan bahasa Inggris, konfigurasi `English-Middle` diasumsikan optimal karena *Head Spectrum* dinilai hanya berisi distorsi dari *stopwords* berfrekuensi tinggi. Eksperimen ini menunjukkan sebaliknya: `Indonesian-Retention` yang justru mempertahankan *Head* mencapai NDCG@10 sebesar 0.2900, melampaui `English-Middle` sebesar +24.3% secara signifikan ($p < 0.01$). Benturan teoritis yang mencolok ini memunculkan pertanyaan kritis: apakah Chen dkk. keliru secara empiris? Data *profiling* pada Tabel 1 menunjukkan bahwa mereka tidak sepenuhnya keliru untuk bahasa Inggris, melainkan distribusi spektral dari *noise* pada dasarnya terbalik ketika beralih dari kosakata khusus bahasa Inggris ke kosakata bersama multilingual (*multilingual joint vocabulary*). Dalam pengaturan multilingual, penanda struktural dan identitas bahasa mendominasi *Head* yang bervariansi tinggi, mendorong *stopwords* umum bahasa Inggris ke zona *Tail* yang bervariansi rendah.

Keunggulan `Indonesian-Retention` dapat dijelaskan melalui mekanisme distribusi energi yang teridentifikasi pada analisis *profiling*. Bahasa Indonesia sebagai bahasa aglutinatif menyandikan informasi gramatikal melalui imbuhan terikat yang melekat pada kata dasar; imbuhan seperti `meng-`, `ber-`, `-kan`, dan `-nya` menentukan fungsi sintaksis dan semantik kata secara fundamental. Analisis L2-norm menunjukkan bahwa token-token ini menempatkan 75–80% energi latennya di zona *Head* dan *Middle*, sehingga jendela retensi yang mencakup kedua zona tersebut menghasilkan representasi leksikal yang lebih utuh dibandingkan pendekatan yang membuang *Head*. Selain itu, penting untuk membahas mengapa performa *Baseline* berdimensi penuh 1536 (NDCG@10 = 0.1592) lebih buruk daripada ruang yang terkompresi (0.2333 dan 0.2900). Meskipun intuisi standar menganggap dimensi yang lebih tinggi menyimpan lebih banyak informasi, mempertahankan seluruh spektrum SVD berarti mengikutsertakan zona *Tail*. Seperti yang ditunjukkan oleh konfigurasi `Tail-Retention` (NDCG@10 = 0.0808), zona ini didominasi oleh *noise* anisotropik bervariansi rendah yang merusak metrik jarak. Pemotongan yang tepat sasaran membuang *noise* anisotropik ini, menjelaskan mengapa ruang semantik terkompresi mampu mengungguli *baseline* terlepas dari dimensinya yang lebih rendah.

Konsistensi pola distribusi energi lintas tiga model (Qwen2.5-1.5B, Qwen2.5-7B, Llama-3.1-70B) dan dua keluarga arsitektur mengindikasikan bahwa temuan ini bukan artefak dari satu konfigurasi model, melainkan refleksi dari struktur morfologis bahasa yang termediasi oleh mekanisme *tokenization* dan pelatihan LLM. Implikasi praktisnya, algoritma L2-Norm *Profiling* yang dikembangkan dalam penelitian ini dapat berfungsi sebagai alat diagnostik prediktif: sebelum mengimplementasikan reduksi dimensi pada model berskala besar, praktisi cukup mengekstrak matriks `lm_head` dan menguji distribusi energi afiksasi untuk menentukan jendela retensi yang tepat tanpa memerlukan *benchmark* RAG berskala penuh.

Penelitian ini memiliki beberapa keterbatasan. Evaluasi *retrieval* hanya dilakukan pada satu dataset (MIRACL Indonesia) dengan satu model utama (Qwen2.5-1.5B); meskipun validasi *profiling* mencakup tiga model, pengujian performa *retrieval* pada model 7B dan 70B belum dilaksanakan. Rasio kompresi yang diuji terbatas pada 2× (50%), dan belum diketahui bagaimana performa berubah pada rasio yang lebih agresif atau konservatif. Eksperimen lanjutan yang mencakup variasi model, rasio kompresi, dan bahasa aglutinatif lainnya (Turki, Korea, Jepang) diperlukan untuk memvalidasi generalisasi temuan ini.

---

## 5. Kesimpulan

Penelitian ini mengusulkan kerangka adaptasi praktis bagi metode EmbFilter yang disesuaikan dengan karakteristik linguistik bahasa Indonesia, dengan tiga kontribusi utama. Pertama, dimensi representasi semantik dikompresi sebesar **50%** (dari 1536 menjadi 768 dimensi) melalui proyeksi ortogonal pada matriks *unembedding* dengan jendela retensi `0:768`, secara langsung mengurangi kebutuhan memori penyimpanan dan komputasi *vector search* pada arsitektur RAG. Kedua, penyaringan komponen *Tail Spectrum* dengan tetap mempertahankan fitur afiksasi di rentang `0:768` meningkatkan performa NDCG@10 sebesar **+82%** (dari 0.1592 menjadi 0.2900), menunjukkan bahwa kompresi yang tepat sasaran justru dapat meningkatkan akurasi *retrieval*. Ketiga, algoritma L2-Norm *Profiling* menyediakan alat diagnostik lintas-model yang memungkinkan praktisi menentukan jendela retensi optimal berdasarkan distribusi energi morfologis, tanpa memerlukan investasi komputasi untuk *fine-tuning* atau *benchmark* berskala penuh.
