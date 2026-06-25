# Less is More: Boosting Multilingual Retrieval Performance through SVD-Based Semantic Compression
**Mengapa Retensi Spektrum Head Mengungguli SVD Filtering Berpusat-Inggris untuk Retrieval Bahasa Indonesia**

Fahmi Alfayadh  
25 Juni 2026

## Abstrak
Pemanfaatan Large Language Models (LLMs) Decoder-Only sebagai generative embedders semakin banyak diteliti, dengan studi terbaru menunjukkan efektivitas causal LLM yang di-instruction-tuned melalui Last-Token Pooling. Teknik perbaikan representasi post-hoc seperti EmbFilter membuang komponen singular value ekstrem (spektrum tepi) dari matriks unembedding untuk menghilangkan noise. Namun, EmbFilter mengasumsikan bahwa spektrum head hanya berisi noise dari stopword — sebuah asumsi yang belum teruji untuk bahasa aglutinatif, di mana tokenizer subword standar memecah morfem terikat menjadi subword yang secara semantik tidak transparan, menyebabkan perilaku spektralnya berbeda secara mendasar dari stopword bahasa Inggris. Studi ini mengusulkan algoritma L2-norm profiling untuk mengevaluasi distribusi energi dari afiks bahasa Indonesia melintasi spektrum SVD dari model Qwen2.5 dan Llama-3.1. Kami mendemonstrasikan bahwa afiks penting bahasa Indonesia mengonsentrasikan sebagian besar energi semantiknya di spektrum head dan middle. Berdasarkan temuan ini, kami mengusulkan konfigurasi jendela retensi yang digeser (Indonesian-Retention, mencakup indeks 0–767). Dievaluasi pada pengaturan zero-shot yang tidak disupervisi, pendekatan ini mengompresi representasi sebesar 50% sekaligus meningkatkan NDCG@10 secara signifikan dari baseline 0.1592 menjadi 0.2900 pada tugas retrieval MIRACL bahasa Indonesia.

---

## 1. Latar Belakang
Pemanfaatan Large Language Models (LLMs) Decoder-Only sebagai generative embedders semakin banyak diteliti, dengan model seperti E5-Mistral (Wang et al., 2023) dan GritLM (Muennighoff et al., 2024) mengalihfungsikan causal LLMs melalui instruction tuning dan Last-Token Pooling. Namun, pendekatan ini mewarisi bias geometris dari objektif Next-Token Prediction (NTP), menghasilkan ruang semantik anisotropik di mana cosine similarity terdistorsi oleh bias frekuensi token — yang mendegradasi performa pada tugas Information Retrieval (IR) dan Retrieval-Augmented Generation (RAG).

Teknik perbaikan representasi post-hoc menawarkan alternatif tanpa pelatihan ulang. Chen et al. (2026) menunjukkan bahwa dekomposisi SVD pada matriks unembedding mengungkap spektrum yang terstruktur: Spektrum Head menangkap noise stopword berfrekuensi tinggi, sementara Tail menangkap token langka. Metode EmbFilter mereka membuang kedua ekstrem tersebut, memproyeksikan vektor ke dimensi tengah yang tersisa.

Meskipun efektif pada bahasa Inggris, EmbFilter mengasumsikan Spektrum Head secara universal mengandung noise stopword — sebuah asumsi yang tidak teruji untuk bahasa aglutinatif. BPE tokenizer memfragmentasi morfem terikat dari bahasa semacam itu menjadi subword yang secara semantik tersembunyi (Bostan et al., 2023), menyebabkan mereka berperilaku berbeda dari stopword Inggris dalam spektrum SVD. Studi ini menguji asumsi tersebut untuk bahasa Indonesia dan mengusulkan jendela retensi yang diadaptasi terhadap distribusi energi morfologisnya.

---

## 2. Algoritma dan Metodologi
Studi ini mengimplementasikan filter proyeksi ortogonal berbasis aljabar linier sesuai dengan kerangka kerja EmbFilter, diaplikasikan pada model Qwen2.5-1.5B ($d=1536$).

### 2.1. Dekomposisi Matriks Unembedding
Matriks unembedding (didefinisikan sebagai `lm_head.weight` pada arsitektur Qwen) direpresentasikan sebagai $W \in \mathbb{R}^{|V| \times d}$, di mana $|V|$ adalah ukuran kosakata (vocabulary) dan $d$ adalah dimensi laten. Matriks $W$ didekomposisi secara penuh menggunakan Singular Value Decomposition (Full SVD) tanpa pemotongan:

$$W = U \Sigma V^T$$

di mana $U$ adalah matriks singular kiri, $\Sigma$ adalah matriks diagonal dari nilai singular yang diurutkan secara menurun, dan $V^T \in \mathbb{R}^{d \times d}$ adalah matriks singular kanan transpos. Karena $\Sigma$ diurutkan secara menurun, baris-baris teratas dari $V^T$ berkorespondensi dengan nilai singular terbesar (Spektrum Head), dan baris-baris terbawah berkorespondensi dengan nilai singular terkecil (Spektrum Tail).

### 2.2. Algoritma Pergeseran Jendela Retensi Berbasis Profiling L2-Norm
Untuk rasio kompresi $2\times$, target dimensinya adalah $d' = 1536/2 = 768$. Daripada mengadopsi rentang pemotongan default dari paper referensi, kami mengembangkan algoritma latent energy profiling untuk menentukan rentang optimal secara empiris. Algoritma ini beroperasi dalam tiga tahap: mengekstraksi matriks `lm_head` dan melakukan Full SVD, memproyeksikan vektor baris tiap token ke dalam tiga zona spektral (Head Top 25%, Middle Mid 50%, Tail Bottom 25%), kemudian menghitung kuadrat magnitudo energi (projective L2-norm) untuk mendeteksi di mana informasi semantik tiap token terkonsentrasi. Hasil pemetaan distribusi kosakata disajikan pada Tabel 1.

<div align="center">

**Tabel 1: Distribusi Kosakata Berdasarkan Zona Spektrum**

| Zona Spektrum | Rentang Dimensi | Karakteristik Utama | Contoh Token |
|:---|:---:|:---|:---|
| **Spektrum Head** | Top 25% | Variansi antar-dokumen yang tinggi; didominasi oleh penanda struktural, aksara non-Latin, dan elemen kode/HTML | `'أوضاع'` (Arab), `'낡'` (Korea), `"\n\n\n\n"`, `'<//'` |
| **Spektrum Middle** | Mid 50% | Token morfologis dan afiks; morfem terikat, sufiks pembentuk kata | `'edly'`, `'ingly'`, `'lessly'`, `' herself'`, `'为空'` |
| **Spektrum Tail** | Bottom 25% | Frekuensi tinggi, variansi lintas-konteks yang rendah; partikel umum dari berbagai bahasa | `' they'`, `' about'`, `' its'`, `'他们'`, `'a'`, `'\n'`, `'<\|endoftext\|>'` |

</div>

Distribusi pada Tabel 1 mengungkap temuan yang menantang asumsi paper referensi. Spektrum Tail didominasi oleh stopword Inggris/Mandarin, karakter tunggal, dan token struktural yang muncul konstan di seluruh teks dengan variansi mendekati nol, mengkonfirmasi bahwa dimensi-dimensi ini sebagian besar berisi noise anisotropik. Sebaliknya, Spektrum Head menampung penanda identitas bahasa seperti aksara non-Latin dan elemen multibahasa, bertentangan dengan asumsi EmbFilter bahwa zona ini hanya memuat noise frekuensi stopword. Temuan ini memunculkan pertanyaan kritis: bagaimana energi dari token morfologis bahasa Indonesia didistribusikan melintasi spektrum SVD?

Untuk menjawab pertanyaan ini, kami menganalisis distribusi L2-norm dari 12 token afiksasi penting bahasa Indonesia, mencakup sufiks (`-nya`, `-lah`, `-kan`, `-pun`, `-kah`, `-ku`, `-mu`) dan prefiks (`di-`, `ter-`, `ber-`, `meng-`, `mem-`). Hasilnya disajikan pada Tabel 2.

<div align="center">

**Tabel 2: Distribusi L2-Norm Afiks Bahasa Indonesia (Qwen2.5-1.5B)**

| Token Afiks | Spektrum Head (Top 25%) | Spektrum Middle (Mid 50%) | Spektrum Tail (Bot 25%) |
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

</div>

Token-token afiks kritis mengakumulasi **75% hingga 80%** dari total energi laten mereka pada gabungan Spektrum Head dan Middle, sangat kontras dengan stopword Inggris yang terkonsentrasi di Tail. Gambar 1 memvisualisasikan perbedaan lokalisasi ini melalui distribusi energi SVD rata-rata (L1-norm) untuk kedua kelompok token di seluruh spektrum 1536-dimensi, yang dihaluskan dengan moving window berukuran 64.

![Gambar 1: Kurva Energi SVD: Afiks Bahasa Indonesia vs. Stopwords Inggris](data/energy_curve.png)

Kurva pada Gambar 1 secara visual mengonfirmasi data kuantitatif yang ada: energi stopword Inggris memuncak pada spektrum Middle dan Tail namun turun tajam di Head, sedangkan energi afiks bahasa Indonesia memuncak tajam di Spektrum Head (dimensi 0–384). Pola ini secara langsung menjelaskan mengapa filter English-Middle, yang membuang 25% spektrum pertama, merusak representasi semantik bahasa Indonesia karena menghilangkan wilayah dengan konsentrasi energi afiks tertinggi.

### 2.3. Validasi Generalisasi Lintas-Model dan Lintas-Arsitektur
Konsentrasi energi morfologis pada Spektrum Head yang diidentifikasi pada Qwen2.5-1.5B membutuhkan verifikasi: apakah ini karakteristik linguistik universal, atau hanya sekadar artefak dari arsitektur dan skala model tertentu? Kami menguji generalisasi temuan ini pada dua model tambahan yang berbeda secara signifikan dalam hal skala dan arsitektur.

Uji pertama dilakukan pada **Qwen2.5-7B** (3.584 dimensi laten) untuk memverifikasi konsistensi pada skala yang lebih besar dalam rumpun arsitektur yang sama. Tabel 3 menunjukkan bahwa pola distribusi energi tetap konsisten: rentang Head dan Middle secara gabungan mengakumulasi ~75% hingga 88% total energi semantik dari afiks bahasa Indonesia, sementara porsi Tail tetap minimal (11–21%).

<div align="center">

**Tabel 3: Distribusi L2-Norm Afiks Bahasa Indonesia (Qwen2.5-7B)**

| Token Afiks | Spektrum Head (0-25%) | Spektrum Middle (25-75%) | Spektrum Tail (75-100%) |
|:---|:---:|:---:|:---:|
| `'nya'` | 33.6% | **45.0%** | 21.4% |
| `'lah'` | 38.4% | **46.3%** | 15.3% |
| `'kan'` | 37.4% | **49.8%** | 12.9% |
| `'pun'` | 38.7% | **49.0%** | 12.3% |
| `' meng'`| 32.6% | **50.5%** | 16.9% |
| `' ber'` | 29.9% | **55.4%** | 14.7% |
| `' ter'` | 38.1% | **50.7%** | 11.1% |

</div>

Uji kedua memperluas analisis ini ke arsitektur yang sama sekali berbeda: **Llama-3.1-70B** ($d=8192$), dengan struktur tokenizer dan kosakata yang sepenuhnya terpisah dari keluarga Qwen. Tabel 4 menunjukkan bahwa bahkan pada parameter 70 miliar, morfem terikat bahasa Indonesia terus menyimpan mayoritas energi semantiknya ke spektrum Head dan Middle, dengan porsi Tail yang bahkan lebih kecil (7–12%) dibandingkan dengan kedua model Qwen.

<div align="center">

**Tabel 4: Distribusi L2-Norm Afiks Bahasa Indonesia (Llama-3.1-70B)**

| Token Afiks | Spektrum Head (0-25%) | Spektrum Middle (25-75%) | Spektrum Tail (75-100%) |
|:---|:---:|:---:|:---:|
| `'nya'` | 43.1% | **45.6%** | 11.4% |
| `'lah'` | **45.8%** | 44.0% | 10.2% |
| `'kan'` | **48.5%** | 43.8% | 7.7% |
| `'pun'` | **48.9%** | 42.8% | 8.2% |
| `' meng'` | 42.4% | **45.4%** | 12.2% |
| `' ber'` | 35.1% | **54.6%** | 10.3% |
| `' ter'` | 37.4% | **52.4%** | 10.2% |

</div>

Kurva energi L1-norm untuk Llama-3.1-70B (Gambar 2) memperkuat temuan ini secara visual. Meskipun spektrumnya membentang sepanjang 8192 dimensi, pola struktural yang sama kembali terulang: stopword bahasa Inggris memuncak di Tail, sementara afiks bahasa Indonesia memuncak tajam di Head dan mempertahankan variansi yang tinggi sepanjang Middle.

![Gambar 2: Kurva Energi SVD (Llama-3.1-70B): Afiks Bahasa Indonesia vs. Stopwords Inggris](data/energy_curve-llama-70b.png)

Konsistensi pola ini di ketiga model (1.5B, 7B, 70B) dan dua keluarga arsitektur (Qwen, Llama) mengindikasikan bahwa konsentrasi energi morfologis di Spektrum Head mencerminkan cara multilingual LLMs merepresentasikan morfem terikat dalam ruang laten, bukan sekadar artefak teknis dari sebuah arsitektur tunggal. Fenomena ini relevan untuk seluruh kelompok bahasa aglutinatif (Turki, Finlandia, Hungaria, Korea, Jepang) yang membangun relasi tata bahasa melalui pelekatan sekuensial morfem terikat. Pada LLMs yang dilatih dengan kosakata gabungan, morfem-morfem terikat ini memiliki frekuensi dokumen yang tinggi dan variansi kontekstual yang rendah, menyebabkannya terproyeksi secara kuat ke komponen singular tertinggi.

### 2.4. Konfigurasi Proyeksi
Bukti empiris dari ketiga model mendemonstrasikan bahwa pendekatan English-Middle membuang **28% hingga 38% energi semantik** (L2-norm) dari partikel sintaksis bahasa Indonesia yang esensial, seperti prefiks `meng-`, `ter-`, dan sufiks `-nya`, `-lah`. Algoritma yang kami usulkan menggeser jendela retensi (kompresi 50%) ke rentang dimensi 0 hingga 768, melingkupi seluruh Spektrum Head dan separuh atas dari Spektrum Middle (Indonesian-Retention). Pergeseran ini mempertahankan fitur morfologis sekaligus mengeliminasi komponen noise pada Spektrum Tail.

Empat konfigurasi matriks proyeksi $V_{sub} \in \mathbb{R}^{d' \times d}$ dievaluasi:

<div align="center">

**Tabel 5: Konfigurasi Matriks Proyeksi**

| Konfigurasi | Deskripsi | Rentang Indeks $V_{sub}$ |
|:---|:---|:---|
| **Baseline** | Tanpa pemotongan; vektor asli $x \in \mathbb{R}^{1536}$ digunakan secara langsung | — |
| **English-Middle** | Membuang 25% Head dan 25% Tail (pendekatan paper referensi) | Indeks 384–1151 |
| **Indonesian-Retention** | Mempertahankan 50% spektrum Head dan Middle (berdasarkan algoritma profiling) | Indeks 0–767 |
| **Tail-Retention** | Mempertahankan 50% spektrum Tail | Indeks 768–1535 |

</div>

### 2.5. Proyeksi Representasi
Untuk memfilter representasi, kami mendefinisikan matriks proyeksi terpotong $V_{sub} \in \mathbb{R}^{d' \times d}$, yang hanya mempertahankan baris-baris sesuai dengan indeks spektral target kami. Setiap kalimat di-encode menjadi representasi awal $x \in \mathbb{R}^d$ menggunakan Last-Token Pooling. Vektor yang telah difilter dan dikompresi $x' \in \mathbb{R}^{d'}$ didapatkan melalui proyeksi ortogonal: 

$$x' = x V_{sub}^T$$

Perkalian ini secara efektif menyaring dimensi spektral yang dibuang sambil memetakan vektor asli ke dalam subruang padat yang baru, yang selanjutnya digunakan untuk menghitung cosine similarity.

---

## 3. Hasil Eksperimen

### 3.1. Pengaturan Evaluasi
**Dataset:**

<div align="center">

**Tabel 6: Deskripsi Dataset untuk Evaluasi**

| Tugas | Dataset | Split | Deskripsi |
|:---|:---|:---|:---|
| Retrieval (RAG) | MIRACL — Indonesian | dev | 500 kueri acak; 4.543 dokumen positif + 5.000 dokumen negatif sampel |

</div>

Performa dari ruang semantik yang terproyeksi diukur menggunakan dua metrik standard information retrieval. **NDCG@10** (Normalized Discounted Cumulative Gain at top 10 candidates) mengukur efektivitas retrieval dengan memperhitungkan posisi dokumen relevan dalam daftar hasil pencarian; ia mempenalti kasus di mana dokumen relevan muncul lebih rendah dalam peringkat, menjadikannya metrik paling kritis untuk sistem RAG yang sensitif terhadap pengurutan konteks. **Recall@100** mengukur persentase dokumen relevan yang berhasil terambil dalam 100 hasil teratas, mengindikasikan seberapa baik vektor tersebut mampu menemukan dokumen yang benar dalam korpus berukuran besar.

Signifikansi perbedaan antar metode dievaluasi menggunakan Paired Student’s t-test dengan threshold $\alpha=0.01$ menggunakan pustaka metrik `ranx`.

### 3.2. Hasil Utama
Tabel 7 merangkum performa rata-rata dari setiap konfigurasi yang dievaluasi pada tugas MIRACL Indonesian retrieval.

<div align="center">

**Tabel 7: Performa Rata-rata pada Tugas Retrieval MIRACL Indonesia**

| Konfigurasi | Dimensi | NDCG@10 | Recall@100 |
|:---|:---:|:---:|:---:|
| Baseline | 1536 | 0.1592 | 0.4211 |
| English-Middle | 768 | 0.2333 | 0.6034 |
| Indonesian-Retention | 768 | **0.2900** | **0.6535** |
| Tail-Retention | 768 | 0.0808 | 0.2485 |

</div>

Konfigurasi Indonesian-Retention mencapai NDCG@10 tertinggi (0.2900) dan Recall@100 (0.6535), melampaui semua konfigurasi lainnya termasuk vektor Baseline dimensi penuh. Sebaliknya, Tail-Retention menghasilkan performa terendah (NDCG@10 = 0.0808), bahkan di bawah Baseline, mengonfirmasi bahwa zona Tail didominasi oleh komponen noise.

### 3.3. Uji Signifikansi Statistik
Untuk memastikan perbedaan performa bukan merupakan kebetulan statistik (statistical artifact), kami melakukan Paired Student’s t-test pada tingkat $\alpha=0.01$:

**Tabel 8: Hasil Paired Student's t-test untuk Signifikansi**

| # | Model | NDCG@10 | Recall@100 |
|:---|:---|:---|:---|
| a | run_baseline | 0.159ᵈ | 0.421ᵈ |
| b | run_english_middle | 0.233ᵃ,ᵈ | 0.603ᵃ,ᵈ |
| c | run_indonesian_retention | 0.290ᵃ,ᵇ,ᵈ | 0.654ᵃ,ᵇ,ᵈ |
| d | run_tail_retention | 0.081 | 0.248 |

*Catatan: Superskrip menunjukkan perbedaan yang signifikan secara statistik ($p<0.01$) terhadap indeks model di baris tersebut.*

Notasi superskrip menunjukkan bahwa model di baris tersebut secara signifikan statistik ($p<0.01$) mengungguli model dengan indeks terkait. Konfigurasi Indonesian-Retention (c) dengan notasi `0.290ᵃ,ᵇ,ᵈ` mengungguli seluruh konfigurasi secara absolut dan meyakinkan pada kedua metrik.

---

## 4. Diskusi
Hasil eksperimen menyajikan bukti kuat yang menyanggah asumsi inti dari paper referensi ketika metode tersebut diterapkan pada LLM multibahasa untuk tugas korpus bahasa Indonesia. Pada evaluasi Chen et al. dalam bahasa Inggris (Chen et al., 2026), konfigurasi English-Middle dianggap optimal karena Spektrum Head dianggap hanya berisi distorsi dari stopword berfrekuensi tinggi. Eksperimen ini mendemonstrasikan kebalikannya: Indonesian-Retention, yang mempertahankan Head, mencapai NDCG@10 sebesar 0.2900, mengungguli English-Middle sebesar +24.3% dengan signifikansi statistik ($p<0.01$). Benturan teoritis yang mencolok ini memunculkan sebuah pertanyaan kritis: apakah Chen et al. salah secara empiris? Data profiling menunjukkan bahwa mereka tidak sepenuhnya salah untuk bahasa Inggris, melainkan distribusi spektral dari noise terbalik secara fundamental ketika berpindah dari kosakata eksklusif Inggris ke kosakata gabungan multibahasa. Dalam pengaturan multibahasa, penanda struktural dan token identitas lintas bahasa mendominasi Head dengan variansi tinggi, mendorong stopword umum Inggris turun ke dalam Spektrum Tail yang memiliki variansi rendah.

Keunggulan dari Indonesian-Retention dapat dijelaskan melalui mekanisme distribusi energi yang diidentifikasi pada analisis profiling. Bahasa Indonesia menyandikan informasi tata bahasa melalui afiks terikat (`meng-`, `ber-`, `-kan`, `-nya`) yang secara fundamental menentukan fungsi sintaksis dan semantik. Tokenizer berbasis BPE secara sistematis menghasilkan pemecahan token untuk bahasa yang kaya morfologis (Petrov et al., 2023), menyebabkan morfem-morfem ini menempati spektrum yang berbeda dari stopword Inggris. Analisis L2-norm mengonfirmasi hal ini: token afiks mendeposit 75–80% dari energi latennya di zona Head dan Middle, sementara zona Tail yang didominasi oleh noise anisotropik dengan variansi rendah (Chen et al., 2026) menyimpan memori semantik dalam jumlah minimal, dibuktikan oleh performa NDCG@10 nyaris menyentuh dasar pada Tail-Retention (0.0808). Selain itu, mekanisme ini menjelaskan mengapa Baseline dengan dimensi penuh 1536 (NDCG@10 = 0.1592) berperforma lebih buruk dari ruang yang terpotong. Kendati intuisi standar mengharapkan dimensi yang lebih tinggi menyimpan informasi lebih banyak, mempertahankan spektrum penuh SVD memaksa penyertaan noise anisotropik Tail ini, yang secara aktif menurunkan kualitas metrik jarak. Pemotongan yang tepat secara eksplisit menghapus noise ini, sehingga ruang semantik yang tereduksi mampu mengungguli baseline utuh terlepas dari dimensinya yang lebih rendah.

Konsistensi dari pola distribusi energi di tiga model dan dua arsitektur keluarga mengindikasikan bahwa konsentrasi energi morfologis di Spektrum Head merupakan cerminan properti yang lebih luas tentang bagaimana LLM multibahasa merepresentasikan morfem terikat, kendati verifikasi lebih lanjut masih diperlukan.

Penelitian ini memiliki beberapa keterbatasan. Evaluasi retrieval dilakukan pada dataset tunggal (MIRACL Indonesian) dengan satu model utama (Qwen2.5-1.5B); meskipun validasi profiling telah mencakup tiga model, uji coba performa retrieval pada model 7B dan 70B belum dilakukan. Rasio kompresi dibatasi pada batas 2× (50%), dan performanya pada rasio yang lebih ekstrem masih belum diketahui. Eksperimen lebih lanjut pada variasi model lain, rasio kompresi lainnya, serta rumpun bahasa aglutinatif (Turki, Korea, Jepang) sangat diperlukan untuk memastikan jangkauan generalisasi ini.

---

## 5. Kesimpulan
Penelitian ini mengusulkan sebuah kerangka kerja adaptasi praktis bagi metodologi EmbFilter yang disesuaikan terhadap karakteristik linguistik bahasa Indonesia, dengan tiga kontribusi utama. Pertama, kompresi dimensi representasi semantik sebesar 50% (dari 1536 ke 768 dimensi) melalui proyeksi ortogonal dari matriks unembedding dengan rentang jendela retensi 0:768, secara langsung mengurangi kebutuhan memori dan biaya komputasi vector search pada arsitektur RAG. Kedua, filtering Spektrum Tail sekaligus mempertahankan fitur afiks di spektrum 0:768 secara nyata meningkatkan capaian NDCG@10 sebesar +82% (dari 0.1592 menjadi 0.2900), menunjukkan bahwa bentuk kompresi yang akurat justru mampu meningkatkan ketajaman akurasi retrieval. Dengan membuktikan bahwa morfem terikat memusatkan energi latennya di Spektrum Head, studi ini menjembatani kesenjangan antara penyaringan representasi menggunakan aljabar linier dan morfologi lintas bahasa, memperlihatkan bahwa elemen yang selama ini dianggap noise frekuensi pada model berpusat pada Inggris justru merupakan fondasi utama perancah struktur gramatikal dalam bahasa aglutinatif—temuan ini pun selaras dengan bukti terbaru ihwal kelemahan tokenisasi BPE pada bahasa kaya struktur kata (Bostan et al., 2023). Ketiga, Algoritma L2-Norm Profiling memfasilitasi metode diagnosis adaptif antar-model, memudahkan pemakai model dalam menetapkan jeda pemotongan rentang retensi optimal berdasarkan distribusi energi morfologis, tanpa membutuhkan beban komputasi tambahan seperti proses fine-tuning atau full-scale benchmarking.

---

## Referensi
1 Y. Chen et al., “Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings,” arXiv preprint arXiv:2606.07502, 2026
2 L. A. M. Bostan et al., “The Impact of Morphology on Cross-Lingual LLMs,” Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 2023
3 N. Muennighoff et al., “Generative Representational Instruction Tuning,” arXiv preprint arXiv:2402.09906, 2024
4 L. Wang et al., “Improving Text Embeddings with Large Language Models,” arXiv preprint arXiv:2401.00368, 2023
5 T. Gao et al., “SimCSE: Simple Contrastive Learning of Sentence Embeddings,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021 
6 P. BehnamGhader et al., “LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders,” arXiv preprint arXiv:2404.05961, 2024
