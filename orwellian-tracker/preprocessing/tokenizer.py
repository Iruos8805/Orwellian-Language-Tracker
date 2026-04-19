from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy

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
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    df = pd.read_csv(args.input_csv, dtype={"speech_id": str})
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    nlp = load_nlp(args.spacy_model)

    token_rows = []
    sentence_rows = []
    speech_stats = []

    for row in df.itertuples(index=False):
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

    token_df = pd.DataFrame(token_rows)
    sentence_df = pd.DataFrame(sentence_rows)
    stats_df = pd.DataFrame(speech_stats)

    token_df.to_csv(
        output_dir / "token_records.csv.gz", index=False, compression="gzip"
    )
    sentence_df.to_csv(
        output_dir / "sentence_records.csv.gz", index=False, compression="gzip"
    )
    stats_df.to_csv(output_dir / "speech_token_stats.csv", index=False)

    print(f"Saved tokens: {output_dir / 'token_records.csv.gz'}")
    print(f"Saved sentences: {output_dir / 'sentence_records.csv.gz'}")
    print(f"Saved speech stats: {output_dir / 'speech_token_stats.csv'}")


if __name__ == "__main__":
    main()
