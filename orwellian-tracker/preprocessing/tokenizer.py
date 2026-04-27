from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

import pandas as pd
import spacy

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocessing.utils import ensure_dir


def load_nlp(model_name: str):
    return spacy.load(model_name, disable=["ner"])


def process_speech(nlp, speech_id: str, text: str):
    doc = nlp(text)
    tokens = []
    sentences = []
    for sent_idx, sent in enumerate(doc.sents):
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        sentences.append(
            {
                "speech_id": speech_id,
                "sentence_id": f"{speech_id}_{sent_idx:05d}",
                "sentence_idx": sent_idx,
                "sentence_text": sent_text,
            }
        )

    words = [t.lemma_.lower() for t in doc if t.is_alpha]
    lemma_counts = Counter(words)

    for token_idx, token in enumerate(doc):
        if token.is_space:
            continue
        lemma = token.lemma_.lower() if token.lemma_ else token.text.lower()
        tokens.append(
            {
                "speech_id": speech_id,
                "token_idx": token_idx,
                "token": token.text,
                "lemma": lemma,
                "pos": token.pos_,
                "dep": token.dep_,
                "is_alpha": bool(token.is_alpha),
                "is_stop": bool(token.is_stop),
                "is_hapax": bool(token.is_alpha and lemma_counts.get(lemma, 0) == 1),
            }
        )

    return tokens, sentences


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize, tag POS, and lemmatize speeches."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--spacy-model", default="en_core_web_lg")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=1000)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    token_out = output_dir / "token_records.csv"
    sentence_out = output_dir / "sentence_records.csv"
    stats_out = output_dir / "speech_token_stats.csv"

    for path in [token_out, sentence_out, stats_out]:
        if path.exists():
            path.unlink()

    df = pd.read_csv(args.input_csv, dtype={"speech_id": str})
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    nlp = load_nlp(args.spacy_model)

    token_rows: list[dict] = []
    sentence_rows: list[dict] = []
    speech_stats: list[dict] = []

    def flush_rows() -> None:
        if token_rows:
            pd.DataFrame(token_rows).to_csv(
                token_out,
                index=False,
                mode="a",
                header=not token_out.exists(),
            )
            token_rows.clear()
        if sentence_rows:
            pd.DataFrame(sentence_rows).to_csv(
                sentence_out,
                index=False,
                mode="a",
                header=not sentence_out.exists(),
            )
            sentence_rows.clear()
        if speech_stats:
            pd.DataFrame(speech_stats).to_csv(
                stats_out,
                index=False,
                mode="a",
                header=not stats_out.exists(),
            )
            speech_stats.clear()

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        speech_id = str(row.speech_id)
        text = str(row.clean_text)
        tokens, sentences = process_speech(nlp, speech_id, text)
        token_rows.extend(tokens)
        sentence_rows.extend(sentences)

        token_df = pd.DataFrame(tokens)
        if token_df.empty:
            continue

        alpha = token_df[token_df["is_alpha"]]
        total_tokens = int(alpha.shape[0])
        unique_tokens = int(alpha["lemma"].nunique())
        hapax_count = int(alpha[alpha["is_hapax"]].shape[0])
        adj = alpha[alpha["pos"] == "ADJ"]
        sub_clause = alpha[alpha["dep"].isin(["advcl", "relcl", "ccomp", "xcomp"])]

        speech_stats.append(
            {
                "speech_id": speech_id,
                "year": row.year,
                "decade": row.decade,
                "total_tokens": total_tokens,
                "unique_tokens": unique_tokens,
                "hapax_count": hapax_count,
                "total_adjectives": int(adj.shape[0]),
                "unique_adjectives": int(adj["lemma"].nunique())
                if not adj.empty
                else 0,
                "subordinate_clause_count": int(sub_clause.shape[0]),
                "sentence_count": len(sentences),
            }
        )

        if idx % max(args.flush_every, 1) == 0:
            flush_rows()

    flush_rows()

    print(f"Saved tokens: {token_out}")
    print(f"Saved sentences: {sentence_out}")
    print(f"Saved speech stats: {stats_out}")


if __name__ == "__main__":
    main()
