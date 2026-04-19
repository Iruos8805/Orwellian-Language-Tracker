# Orwellian Language Decay Tracker

Longitudinal NLP pipeline for measuring linguistic pruning in U.S. Congressional speech.

## Scope and data reality

- The code supports the full historical window (1870s-2010s) when raw congress files are present.
- In this workspace, `hein-daily` currently contains congress 97-114 (roughly 1981-2017), so produced decade outputs will be a subset of the full 15 bins.
- `vocabulary/vocab.txt` in this workspace is an unlabeled bigram list (no year column), so year-level validation falls back to available congressional 2-gram files.

## Repository layout

```
orwellian-tracker/
├── data/
│   ├── raw/
│   ├── processed/
│   └── validation/
├── preprocessing/
├── student1_lexical/
├── student2_syntactic/
├── student3_semantic/
├── integration/
├── visualization/
└── outputs/
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## End-to-end run order

Run from `orwellian-tracker/`.

```bash
python preprocessing/cleaner.py \
  --hein-dir ../hein-daily \
  --vocab-procedural ../vocabulary/vocabulary/procedural.txt \
  --output-dir data/processed

python preprocessing/tokenizer.py \
  --input-csv data/processed/cleaned_speeches.csv \
  --output-dir data/processed

python preprocessing/amr_parser.py \
  --sentences-csv data/processed/sentence_records.csv.gz \
  --output-dir data/processed \
  --workers 2

python preprocessing/embedder.py \
  --cleaned-csv data/processed/cleaned_speeches.csv \
  --sentences-csv data/processed/sentence_records.csv.gz \
  --tokens-csv data/processed/token_records.csv.gz \
  --output-dir data/processed

python student1_lexical/compute_lexical.py \
  --cleaned-csv data/processed/cleaned_speeches.csv \
  --tokens-csv data/processed/token_records.csv.gz \
  --hein-dir ../hein-daily \
  --vocab-dir ../vocabulary/vocabulary \
  --output-dir student1_lexical

python student2_syntactic/compute_syntactic.py \
  --amr-metrics data/processed/amr_speech_metrics.csv \
  --speech-stats data/processed/speech_token_stats.csv \
  --output-dir student2_syntactic

python student3_semantic/compute_semantic.py \
  --cleaned-csv data/processed/cleaned_speeches.csv \
  --w2v-dir data/processed/w2v_models \
  --targets-file student3_semantic/target_words.txt \
  --output-dir student3_semantic

python integration/newspeak_index.py \
  --lexical student1_lexical/diversity_scores.csv \
  --syntactic student2_syntactic/depth_scores.csv \
  --semantic student3_semantic/drift_rate_by_decade.csv \
  --output-dir integration \
  --outputs-dir outputs

python integration/speakermap_join.py \
  --cleaned-csv data/processed/cleaned_speeches.csv \
  --newspeak-csv integration/newspeak_index.csv \
  --output-dir integration \
  --outputs-dir outputs

python visualization/plots.py \
  --lexical student1_lexical/diversity_scores.csv \
  --syntactic student2_syntactic/depth_scores.csv \
  --semantic student3_semantic/drift_rate_by_decade.csv \
  --index integration/newspeak_index.csv \
  --party integration/index_by_party.csv \
  --output-dir outputs/figures
```

Dashboard:

```bash
streamlit run visualization/app.py
```
