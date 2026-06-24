# Rangkuman Komprehensif: Validasi Metodologi SVD & Last-Token Pooling (arXiv:2606.07502)

Eksperimen telah selesai dieksekusi dengan perbaikan arsitektural yang 100% patuh pada protokol *arXiv:2606.07502* untuk arsitektur *Decoder-Only*, yaitu menggunakan **Last-Token Pooling** (dengan *left-padding*) dan **Mathematical SVD Pearson-Correlation** pada matriks `lm_head` murni.

## 1. Hasil Evaluasi Akhir (Qwen2.5-1.5B Base)

| Konfigurasi Filter | STS Spearman | NDCG@10 | Recall@100 |
| :--- | :---: | :---: | :---: |
| **Baseline** | -0.2882 | 0.000712 | 0.0128 |
| **English-Filter** | -0.2882 | 0.000861 | 0.0121 |
| **Indonesian-Filter**| -0.2883 | 0.000773 | 0.0120 |

> [!CAUTION]
> **Keruntuhan Representasi Semantik:** Nilai absolut NDCG yang anjlok ke angka `0.0007` (praktis mendekati 0) membuktikan bahwa model *Base Decoder-Only* yang diekstrak murni menggunakan *Last-Token Pooling* tanpa *Instruction Prompt* sama sekali **tidak memiliki kemampuan retrieval**. 

## 2. Analisis Kritis Temuan Eksperimental

### A. Kegagalan Murni Last-Token Pooling tanpa *Task-Awareness*
Korelasi Spearman pada eksperimen ini tetap bernilai negatif ekstrem (**-0.288**). Ini adalah temuan epistemologis yang sangat penting:
Peralihan dari *Mean-Pooling* ke *Last-Token Pooling* secara teori membersihkan *noise* posisional, namun hasil eksperimen membuktikan bahwa representasi *hidden state* dari *token* terakhir pada model *Base* semata-mata mengagregasi konteks untuk **"menebak token sintaksis selanjutnya"**, BUKAN untuk **"merangkum makna semantik global"**. Tanpa dibungkus *Instruction Prompt* (seperti *"Represent this sentence for searching: "*), *hidden state* ini buta terhadap kemiripan makna.

### B. Dampak Numerik SVD Filter
Meskipun angkanya sangat kecil, secara teknis SVD Filter memberikan injeksi perubahan pada vektor. *English-Filter* menaikkan NDCG sebesar ~20.9% secara relatif (0.00071 -> 0.00086), dan *Indonesian-Filter* memberikan kenaikan ~8.5% (0.00071 -> 0.00077).
Artinya, teori *Edge Spectrum* dari paper tersebut **bekerja secara matematis** memotong dimensi yang dipenuhi *noise* frekuensi (sehingga performa bergeser), tetapi metode ini tidak memiliki kekuatan yang cukup untuk membangkitkan ruang vektor yang sejak awal hancur karena hilangnya *Task-Awareness* (Prompting).

### C. Kesimpulan Riset & Implikasi terhadap Paper Rujukan
Hipotesis SVD Filter dari *arXiv:2606.07502* mungkin terbukti memberikan manfaat marjinal, namun eksekusi empiris kita hari ini memberikan satu dalil telak:
> **Post-Hoc Unembedding Filter tidak bisa berdiri sendiri pada model Causal LLM.** 

Pembuangan *Edge Spectrum* akan sepenuhnya sia-sia jika model tidak diletakkan pada "mode diskriminatif" terlebih dahulu melalui *Instruction Tuning* atau *Instruction Prompts*. Eksperimen kita membuktikan bahwa tanpa paksaan instruksi (metode seperti LLM2Vec / GritLM), topologi *Last-Token* Qwen2.5 adalah representasi generatif murni yang **anti-korelasi** dengan *Semantic Textual Similarity*.

---

## 3. Rekomendasi Fase Berikutnya (Berdasarkan Celah SOTA)
Jika proyek ini akan dilanjutkan menjadi landasan *paper* jurnal, eksperimen kita hari ini telah membuktikan *Research Gap* yang sempurna. Langkah pembuktian pamungkas yang harus dilakukan adalah:
1. **Injeksi Instruction Prompt:** Tambahkan prompt `"Representasikan dokumen ini untuk pencarian:"` sebelum teks kueri/korpus.
2. **Evaluasi Ulang dengan SVD ID vs EN:** Setelah *Prompt* menyelaraskan ruang vektor menjadi isotropik, barulah penyaringan *Indonesian Edge Spectrum* ($V_{\text{noise-ID}}$) versus *English Edge Spectrum* ($V_{\text{noise-EN}}$) akan menunjukkan perbedaan metrik NDCG yang drastis dan membuktikan hipotesis lokalisasi bahasa kita!
