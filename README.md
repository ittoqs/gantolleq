# Sentiment App (5 kelas) — Julia CLI + Pretrained lokal

Aplikasi CLI untuk analisis sentimen dengan output **5 kelas**:
- sangat positif
- positif
- netral
- negatif
- sangat negatif

Mode:
1) **trained**: TF-IDF + classifier multi-kelas (murni Julia) dari dataset berlabel.
2) **pretrained**: transformer sentiment **multilingual** berjalan lokal lewat Python (lokal download sekali), lalu dipetakan ke 5 kelas berbasis skor.

## Prasyarat
- Julia 1.9+ (atau 1.10+)
- Python 3.10+ (ada di PATH sebagai `python`)
- Disk untuk download model transformer saat mode `pretrained` pertama kali

## Install dependencies

**Python (untuk mode pretrained):**
```bash
pip install -r python/requirements.txt
```

**Julia (untuk mode trained & core app):**
Jalankan:
```bash
julia -e 'using Pkg; Pkg.add(["ArgParse","CSV","DataFrames","JSON3","BSON","TextAnalysis","MLJ","MLJLinearModels"])'
