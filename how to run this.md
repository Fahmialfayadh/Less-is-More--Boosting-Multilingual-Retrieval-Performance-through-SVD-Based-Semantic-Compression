# Cara Menjalankan Repositori (How to Run)

Repositori ini bergantung pada `uv` untuk manajemen dependensi dan `colab-cli` untuk komputasi GPU jarak jauh (remote execution) pada Google Colab. Semua *script* bash/shell lama telah dihapus dan digantikan oleh panduan eksekusi ini untuk menjaga kebersihan repositori.

## 1. Setup Awal (Lokal)
Pastikan Anda sudah menginstal `uv`. Lingkungan (*virtual environment*) dapat dibuat dengan:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
# Jika butuh instal manual
uv pip install mteb datasets ranx scikit-learn accelerate
```

## 2. Eksekusi Skrip via Colab CLI (Direkomendasikan)
Karena Qwen2.5-1.5B membutuhkan memori yang cukup besar, sebagian besar *script* dirancang untuk didelegasikan ke T4 GPU di Google Colab.

### a. Membuat Sesi Colab GPU
```bash
colab new -s embed_filter_eval --gpu T4
```

### b. Menjalankan Skrip Profiling Imbuhan
Untuk melihat distribusi *L2-norm* pada spektrum SVD Qwen2.5:
```bash
colab exec -s embed_filter_eval -f scripts/profile_imbuhan_colab.py --timeout 1200
```
*(Hasil bisa dilihat di konsol atau log output)*

### c. Menjalankan Evaluasi RAG & STS (Lintas Bahasa / Retensi)
Kami menyediakan *bundled scripts* (skrip tunggal yang membungkus semua file *source* `src/` menjadi satu file eksekusi untuk kemudahan Colab). Skrip ini akan melakukan ekstraksi vektor, RAG, STS, dan langsung melakukan uji *T-Test*:

```bash
# Menjalankan evaluasi Retensi (Indonesian-Retention vs Baseline dsb)
colab exec -f scripts/bundled_retention_colab.py --timeout 3600 | tee logs/retention_colab_output.log

# Menjalankan evaluasi Cross-lingual/Edge Filter
colab exec -f scripts/bundled_colab_run.py --timeout 3600 | tee logs/crosslingual_colab_output.log
```

## 3. Eksekusi Lokal (CPU/Local GPU)
Jika Anda memiliki resource lokal yang mumpuni, pastikan `PYTHONPATH` telah di-set sebelum menjalankan *script* dalam `src/`:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Ekstraksi Embeddings
uv run python src/extract_crosslingual_filters.py

# Evaluasi RAG
uv run python src/evaluate_crosslingual_rag.py

# Hypothesis Test
uv run python src/hypothesis_test_crosslingual.py
```

## Struktur Repositori yang Baru:
- `/src`: Berisi kode-kode inti (modul *filter*, implementasi dekomposisi, pipeline evaluasi).
- `/scripts`: Berisi *script runner*, *orchestrator* Colab, dan alat analitik/profiling (*standalone scripts*).
- `/docs` atau `/notes`: Catatan *planning*, dokumentasi agen.
- `/logs`: Tempat membuang log eksekusi dan output terminal.
- `/models`: Menyimpan artefak model atau tensor yang didownload (*dijaga oleh gitignore*).
- `/data`: Direktori *dataset* lokal atau *cache* evaluasi.
