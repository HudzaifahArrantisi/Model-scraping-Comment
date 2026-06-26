# AGENTS.md — Sentimen Kebijakan Indonesia

## Project
Pipeline analisis sentimen bahasa Indonesia untuk 5 topik kebijakan: `prabowo_gibran`, `omnibus_law`, `danantara`, `makan_bergizi_gratis`, `efisiensi_anggaran`.

## Entrypoints
- `scraper_kebijakan.py` — generate synthetic dataset or scrape Twitter/TikTok
- `naive_bayes_sentimen.py` — train/run Naive Bayes classifier
- `scan_link.py` — YouTube comment scanner (depends on trained model)

## Required execution order
1. `python scraper_kebijakan.py --sample --max 500` → produces `./data_raw/semua_topik.csv`
2. `python naive_bayes_sentimen.py --input ./data_raw/semua_topik.csv` → trains model, saves to `./hasil/model_nb.pkl`
3. `python scan_link.py --url <youtube_url> --limit 300` → requires trained model from step 2

## Commands
- Generate synthetic data (recommended, no API keys): `python scraper_kebijakan.py --sample --max 500`
- Quick demo with small sample: `python naive_bayes_sentimen.py --sample`
- Predict single sentence: `python naive_bayes_sentimen.py --prediksi "danantara merugikan rakyat kecil"`
- Load saved model + predict: `python naive_bayes_sentimen.py --load-model ./hasil/model_nb.pkl --prediksi "makan bergizi gratis program terbaik"`

### New: Online Learning & Active Learning
- **Online training with SGD + partial_fit**: `python naive_bayes_sentimen.py --sample --use-online --model-type sgd_log`
- **Character n-grams** (catch typos/slang): `python naive_bayes_sentimen.py --sample --use-char-ngrams`
- **Retrain from feedback**: `python naive_bayes_sentimen.py --retrain-feedback --input ./data_raw/semua_topik.csv`
- **Scan + Active Learning**: `python scan_link.py --url <youtube_url> --use-online --model-type sgd_log`
  - User corrects low-confidence predictions → model updates via `partial_fit()` in real-time
  - Corrections saved to `./hasil_scanning/feedback_history.csv`

## Output (binary: hanya positif & negatif)
Hasil output hanya 2 kelas sentimen: **positif** dan **negatif**. Semua label `netral`/`neutral` otomatis dipetakan ke `negatif`.

## Output directories
- `./data_raw/` — scraped/generated CSV data
- `./hasil/` — model, CSV results, plots, JSON report
- `./hasil_scanning/` — YouTube scan results, plots, user feedback history
- `./hasil_scanning/feedback_history.csv` — accumulated user corrections for periodic retrain

## Framework & toolchain quirks
- **ComplementNB** (default) is recommended over MultinomialNB for class imbalance
- **SGDClassifier** (`sgd_log`, `sgd_hinge`, `sgd_huber`) supports `partial_fit()` for online learning
- **SmartSentimentLabeler** replaces simple LexiconLabeler with chain-of-thought reasoning:
  - Sarcasm detection (kata positif + konteks negatif = negatif)
  - Negation reversal ("tidak bagus" -> negatif)
  - Position-weighted scoring (kalimat akhir bobot 1.5x)
  - Contradiction analysis (tapi/namun/sayangnya -> segmen akhir dominan)
  - `LexiconLabeler` alias preserved for backward compatibility
- **Emoji handling**: emoji mapped to sentiment tokens (`positif_emoji`, `negatif_emoji`, `netral_emoji`) before TF-IDF
- `matplotlib.use("Agg")` — non-interactive backend, plots save to file only
- `scan_link.py` imports from `naive_bayes_sentimen.py` (must be co-located)
- Preprocessing uses PySastrawi for Indonesian stemming and slang normalization
- Requirements file is named `reqeuirments.txt` (typo intentional)
- Twitter (`snscrape`) and TikTok (`TikTokApi`) scrapers are unstable — prefer `--sample` mode

## Model types
| Flag | Classifier | Online? | Use case |
|------|-----------|---------|----------|
| `complement` (default) | ComplementNB | No | Imbalanced data |
| `multinomial` | MultinomialNB | No | Balanced data |
| `sgd_log` | SGD + log_loss | Yes | Logistic Regression online |
| `sgd_hinge` | SGD + hinge | Yes | SVM online |
| `sgd_huber` | SGD + modified_huber | Yes | Robust online |

## Learning loop (how model gets smarter)
1. User scans YouTube via `scan_link.py --use-online`
2. Model predicts sentiment for each comment
3. Low-confidence predictions are shown to user for correction
4. User corrections update model via `partial_fit()` in real-time
5. All corrections saved to `./hasil_scanning/feedback_history.csv`
6. Periodic `--retrain-feedback` merges original dataset + all feedback, retrains model

## Important constraints
- No tests, no CI, no linter config
- No formal Python package structure — scripts run as standalone files
- Indonesian text preprocessing pipeline: clean -> normalize_slang -> remove_stopwords -> stem
- Lexicon-based auto-labeling available if CSV lacks `sentimen`/`label` column
