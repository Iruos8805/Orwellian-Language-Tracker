python3 preprocessing/clean_hidden.py \
  --root ../hein-daily && \

python3 preprocessing/cleaner.py \
  --hein-dir ../hein-daily \
  --vocab-procedural ../vocabulary/vocabulary/procedural.txt \
  --output-dir data/processed \
  --sample-per-congress 5000 \
  --balance-decades && \

python3 preprocessing/tokenizer.py \
  --input-csv data/processed/cleaned_speeches_balanced.csv \
  --output-dir data/processed \
  --spacy-model en_core_web_lg \
  --flush-every 1000 && \

python3 preprocessing/amr_parser.py \
  --sentences-csv data/processed/sentence_records.csv \
  --output-dir data/processed \
  --workers 2 \
  --max-sentences-per-speech 10 \
  --flush-every 1000 \
  --require-gpu && \

python3 preprocessing/embedder.py \
  --cleaned-csv data/processed/cleaned_speeches_balanced.csv \
  --sentences-csv data/processed/sentence_records.csv \
  --tokens-csv data/processed/token_records.csv \
  --output-dir data/processed \
  --contextual-model microsoft/deberta-v3-base \
  --seed-words-file student3_semantic/target_words_full.txt && \

PYTHONPATH=. python3 student1_lexical/compute_lexical.py \
  --cleaned-csv data/processed/cleaned_speeches_balanced.csv \
  --tokens-csv data/processed/token_records.csv \
  --hein-dir ../hein-daily \
  --vocab-dir ../vocabulary/vocabulary \
  --mattr-window 500 \
  --output-dir student1_lexical && \

PYTHONPATH=. python3 student2_syntactic/compute_syntactic.py \
  --amr-metrics data/processed/amr_speech_metrics.csv \
  --speech-stats data/processed/speech_token_stats.csv \
  --sentence-embed-index data/processed/deberta_embeddings/sentence_cls_index.csv \
  --output-dir student2_syntactic && \

PYTHONPATH=. python3 student3_semantic/compute_semantic.py \
  --cleaned-csv data/processed/cleaned_speeches_balanced.csv \
  --w2v-dir data/processed/w2v_models \
  --targets-file student3_semantic/target_words_full.txt \
  --min-shared-vocab 500 \
  --min-word-count 5 \
  --deberta-seed-index data/processed/deberta_embeddings/seed_word_index.csv \
  --neighbor-decade-from 1990s \
  --neighbor-decade-to 2010s \
  --neighbor-topn 5 \
  --neighbor-output student3_semantic/word_shift_neighbors.csv \
  --neighbor-summary-output student3_semantic/word_shift_summary.csv \
  --output-dir student3_semantic && \

PYTHONPATH=. python3 integration/newspeak_index.py \
  --lexical student1_lexical/diversity_scores.csv \
  --syntactic student2_syntactic/depth_scores.csv \
  --semantic student3_semantic/drift_rate_by_decade.csv \
  --output-dir integration \
  --outputs-dir outputs \
  --global-scaler-file integration/global_scaler_ranges.csv && \

PYTHONPATH=. python3 integration/speakermap_join.py \
  --cleaned-csv data/processed/cleaned_speeches_balanced.csv \
  --newspeak-csv integration/newspeak_index.csv \
  --output-dir integration \
  --outputs-dir outputs && \

PYTHONPATH=. python3 visualization/plots.py \
  --lexical student1_lexical/diversity_scores.csv \
  --syntactic student2_syntactic/depth_scores.csv \
  --semantic student3_semantic/drift_rate_by_decade.csv \
  --index integration/newspeak_index.csv \
  --party integration/index_by_party.csv \
  --semantic-words student3_semantic/drift_scores.csv \
  --output-dir outputs/figures

  