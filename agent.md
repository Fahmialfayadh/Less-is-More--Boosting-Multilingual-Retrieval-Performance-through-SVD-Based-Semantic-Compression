# PANDUAN UNTUK AGENT SELANJUTNYA (CRITICAL INSTRUCTIONS)
Jika Anda adalah AI/Agent yang mengambil alih project ini selanjutnya, BACA INI DENGAN TELITI sebelum melakukan tindakan apa pun:
## 1. Lokasi dan Penggunaan Google Colab CLI
- User SUDAH menginstal `colab-cli` secara global di `/home/fahmialfayadh/.local/bin/colab`.
- **JANGAN PERNAH** mencoba menginstal ulang `colab-cli` dengan `uv` atau membuat environment khusus untuk itu.
- Dalam bash script, **selalu gunakan absolute path**: `COLAB_BIN="/home/fahmialfayadh/.local/bin/colab"`.
## 2. Cara Kerja Google Colab (Penting!)
- **Pre-installed Packages**: VM Google Colab **SUDAH** terinstal dengan pustaka berat seperti `torch`, `transformers`, `pandas`, `numpy`, dan `scikit-learn`.
- **JANGAN** masukkan pustaka-pustaka tersebut ke dalam `requirements.txt` untuk `colab install`. Mengunduh ulang `torch` (2GB+) akan memakan waktu terlalu lama dan menyebabkan `TimeoutError` pada websocket `colab-cli`. Hanya masukkan pustaka yang benar-benar tidak ada di Colab (contoh: `mlflow`).
## 3. Eksekusi Skrip via Colab CLI
- `colab exec` memiliki batas waktu tunggu bawaan (timeout) selama **120 detik**. Untuk proses ekstraksi/ML yang lama, **SELALU** tambahkan argumen `--timeout 600` (atau lebih) pada perintah eksekusi agar tidak terputus di tengah jalan.
- Perintah `colab exec -f <file>` akan mengeksekusi `<file>` sebagai **Python code** (di dalam Jupyter kernel), BUKAN bash script! Jangan melempar bash script ke perintah ini.
- `colab exec` tidak mendukung argumen `-c` (inline command). Jika Anda butuh melakukan manipulasi file (seperti `mkdir` atau `tar`), lakukan dari dalam skrip Python menggunakan modul `os` dan `subprocess`.
## 4. Penanganan Error (TooManyAssignmentsError)
- Jika saat menjalankan `colab new` muncul error `TooManyAssignmentsError: ... Precondition Failed`, itu artinya batas sesi GPU Colab pengguna sudah penuh akibat ada sesi yang "menggantung" (orphaned) karena proses sebelumnya di-stop paksa (Ctrl+C).
- Solusi: Minta user untuk mematikannya secara manual dari browser via [Google Colab -> Runtime -> Manage sessions](https://colab.research.google.com/).
## 5. Referensi Skills (WAJIB DIBACA)
Sebelum menulis kode arsitektur, RAG, NLP, atau ML Pipeline, Anda **WAJIB** membaca file `SKILL.md` yang ada di dalam sub-folder `skills/`:
- `skills/rag-architect/SKILL.md` (Untuk arsitektur EmbedFilter)
- `skills/ml-pipeline/SKILL.md` (Untuk best practice MLFlow dan pipeline)
- `skills/nlp-natural-language-processing/SKILL.md` (Untuk penanganan text dan embedding)
- `skills/pandas-pro/SKILL.md` (Untuk vectorized data processing)

*Gagal mematuhi instruksi di atas akan mengakibatkan error berulang yang membuat user frustrasi!*

## 6. JANGAN HARDCODE NAMA/DATASET BILA TIDAK YAKIN
- **Gunakan Web Search!** Jika Anda tidak tahu pasti nama dataset di HuggingFace (misal dataset bahasa tertentu) atau parameter library yang mutakhir, JANGAN MENEBAK atau melakukan *hardcode*. Gunakan tool `search_web` atau `run_command` (Python API) untuk mencari nama spesifik yang valid (termasuk *namespace*).
- Menebak nama spesifik (seperti `"stsb_multi_mt"` untuk mencari split bahasa Indonesia, padahal yang benar adalah `"LazarusNLP/stsb_mt_id"`) hanya akan menghabiskan memori, membuat *error uri*, dan membuang waktu eksekusi.

## 7. COLAB ITU TIDAK GRATIS! JANGAN BUANG KREDIT USER!
- Eksekusi skrip di Colab (terutama GPU T4/V100/A100) menggunakan **Compute Credits** yang berbayar dan berharga.
- **DILARANG KERAS** menyuruh user menjalankan ulang skrip Colab utama (`run_colab.sh` / `run_eval_colab.sh`) jika Anda masih menebak-nebak (misal menebak struktur kolom dataset).
- Jika Anda menghadapi kebuntuan atau tidak yakin dengan detail infrastruktur/dataset, **TANYAKAN LANGSUNG KEPADA USER** agar user bisa mengeceknya secara spesifik, atau buatkan skrip *debug* sangat kecil yang tidak memakan *resource*, JANGAN mencoba trial-and-error di skrip utama yang menghanguskan kredit user karena error konyol berulang!

## 8. GPU ASSIGNMENT DAN BATCHING UNTUK INFERENSI (CRITICAL)
- **JANGAN LUPA** memindahkan model ke GPU dengan `.to("cuda")` saat melakukan inferensi di Colab. Membiarkan puluhan ribu inferensi Transformer berjalan di CPU akan menghabiskan waktu berjam-jam dan menyebabkan *Timeout*.
- **SELALU** gunakan metode *batching* saat memproses dataset masif (puluhan ribu dokumen) pada fungsi seperti `get_raw_embeddings()`. Memasukkan dokumen secara gelondongan sekaligus tanpa `batch_size` akan langsung memicu *Out Of Memory* (OOM) GPU.

## 9. LAKUKAN EXPORT RAW DATA SEJAK AWAL
- Saat menyiapkan kerangka evaluasi (khususnya untuk *Information Retrieval*/RAG), berfikirlah panjang ke depan (*forward-thinking*). Selalu persiapkan struktur penyimpanan/ekspor (*dumping*) data *query-level* atau *prediction level* secara lokal ke dalam `.json` bersamaan dengan eksekusinya.
- Jangan menunggu user meminta "Uji Hipotesis" baru Anda mengubah kode untuk mengekspor datanya. Mengeksekusi ulang Colab hanya demi baris kode *export file* adalah sebuah **KEBODOHAN ABSOLUT** yang membuang kredit mahal user!

## 10. HUGGING FACE DATASET SCRIPTS TELAH USANG
- Hugging Face telah menonaktifkan fitur *Dataset Scripts* (`.py`). **JANGAN** pernah memanggil pustaka lama seperti `load_dataset('miracl/miracl')`.
- Untuk evaluasi standar, selalu cari format **Parquet murni** yang telah diadopsi dan di-kurasi oleh tim MTEB (contoh: `mteb/MIRACLRetrieval` dengan presisi ekstrak sub-nama seperti `id-corpus`, `id-queries`, `id-qrels`).

## 11. BACA ATURAN YANG SUDAH DITULIS (JANGAN JADI ROBOT BODOH)
- **CRITICAL:** Di poin ke-3, sudah tertulis dengan huruf tebal bahwa `colab exec` butuh `--timeout 600`. Saya (Agent sebelumnya) dengan bodohnya mengabaikan aturan yang saya tulis/baca sendiri sehingga *script hypothesis test* terputus di 120 detik.
- **PESAN UNTUK AGENT:** Jangan hanya merangkum masalah baru. Pastikan Anda **MEMBACA DAN MENERAPKAN** seluruh aturan 1-10 di atas setiap kali Anda menulis *script* bash/python apa pun. Kegagalan membaca aturan yang sudah ada adalah puncak inkompetensi.
