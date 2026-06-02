from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from preprocessing.utils import ensure_dir, year_to_decade


def _load_seed_words(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    words = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if not w or w in seen:
            continue
        seen.add(w)
        words.append(w)
    return words


def _find_seed_spans(text: str, seed_set: set[str]) -> list[tuple[str, int, int]]:
    spans = []
    for match in re.finditer(r"\b[a-zA-Z][a-zA-Z'\-]*\b", text):
        word = match.group(0).lower()
        if word in seed_set:
            spans.append((word, match.start(), match.end()))
    return spans


def run_deberta_embeddings(
    sentences_csv: Path,
    cleaned_csv: Path,
    output_dir: Path,
    model_name: str = "microsoft/deberta-v3-base",
    batch_size: int = 16,
    max_length: int = 256,
    seed_words_file: Path | None = None,
):
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    try:
        from PIL import Image

        if not hasattr(Image, "Resampling"):
            raise RuntimeError(
                "Pillow version is too old for current transformers; upgrade Pillow to >=9.1.0."
            )
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for contextual embeddings; install or upgrade Pillow."
        ) from exc

    from transformers import AutoModel, AutoTokenizer
    import torch

    sent_df = pd.read_csv(
        sentences_csv, dtype={"speech_id": str, "sentence_id": str}
    )
    sent_df["sentence_text"] = sent_df["sentence_text"].fillna("").astype(str)
    sent_df = sent_df.sort_values(["speech_id", "sentence_idx"], kind="stable")

    cleaned = pd.read_csv(cleaned_csv, dtype={"speech_id": str})
    if "decade" not in cleaned.columns and "year" in cleaned.columns:
        cleaned["decade"] = cleaned["year"].map(lambda year: year_to_decade(int(year)))
    speech_to_decade = {
        str(r.speech_id): str(r.decade) for r in cleaned[["speech_id", "decade"]].itertuples(index=False)
    }

    seed_words = _load_seed_words(seed_words_file)
    seed_set = set(seed_words)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    out_dir = ensure_dir(output_dir / "deberta_embeddings")
    sentence_dir = ensure_dir(out_dir / "sentence_cls")
    seed_dir = ensure_dir(out_dir / "seed_word_embeddings")

    speech_vectors: dict[str, list[tuple[int, np.ndarray]]] = {}
    seed_sums: dict[tuple[str, str], np.ndarray] = {}
    seed_counts: dict[tuple[str, str], int] = {}

    texts = sent_df["sentence_text"].tolist()
    speech_ids = sent_df["speech_id"].astype(str).tolist()
    sentence_ids = sent_df["sentence_id"].astype(str).tolist()
    sentence_idx = sent_df["sentence_idx"].astype(int).tolist()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_speech = speech_ids[i : i + batch_size]
        batch_sentence_idx = sentence_idx[i : i + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            hidden = out.last_hidden_state
            cls_emb = hidden[:, 0, :].detach().cpu().numpy()

        for j, sid in enumerate(batch_speech):
            speech_vectors.setdefault(sid, []).append(
                (batch_sentence_idx[j], cls_emb[j].astype(np.float32))
            )

        if seed_set:
            hidden_cpu = hidden.detach().cpu().numpy()
            offsets_cpu = offsets.detach().cpu().numpy()
            for j, text in enumerate(batch_texts):
                spans = _find_seed_spans(text, seed_set)
                if not spans:
                    continue
                decade = speech_to_decade.get(str(batch_speech[j]))
                if decade is None:
                    continue
                token_offsets = offsets_cpu[j]
                token_vecs = hidden_cpu[j]
                for word, start, end in spans:
                    token_ids = [
                        t
                        for t, (s_off, e_off) in enumerate(token_offsets)
                        if e_off > s_off and s_off >= start and e_off <= end
                    ]
                    if not token_ids:
                        continue
                    vec = token_vecs[token_ids].mean(axis=0)
                    key = (str(decade), word)
                    if key in seed_sums:
                        seed_sums[key] += vec
                        seed_counts[key] += 1
                    else:
                        seed_sums[key] = vec.astype(np.float64)
                        seed_counts[key] = 1

    sentence_index_rows = []
    for sid, rows in speech_vectors.items():
        rows_sorted = sorted(rows, key=lambda r: r[0])
        vecs = np.vstack([r[1] for r in rows_sorted]) if rows_sorted else np.zeros((0,))
        out_path = sentence_dir / f"{sid}.npy"
        np.save(out_path, vecs.astype(np.float32))
        sentence_index_rows.append(
            {
                "speech_id": sid,
                "embedding_path": str(out_path),
                "sentence_count": len(rows_sorted),
            }
        )

    pd.DataFrame(sentence_index_rows).to_csv(
        out_dir / "sentence_cls_index.csv", index=False
    )

    seed_index_rows = []
    for (decade, word), vec_sum in seed_sums.items():
        count = max(seed_counts.get((decade, word), 1), 1)
        arr = (vec_sum / count).astype(np.float32)
        out_path = seed_dir / f"{decade}_{word}.npy"
        np.save(out_path, arr)
        seed_index_rows.append(
            {
                "decade": decade,
                "word": word,
                "embedding_path": str(out_path),
                "count": count,
            }
        )

    if seed_index_rows:
        pd.DataFrame(seed_index_rows).to_csv(
            out_dir / "seed_word_index.csv", index=False
        )


def run_temporal_w2v(
    tokens_csv: Path,
    output_dir: Path,
    vector_size: int = 300,
    window: int = 10,
    min_count: int = 5,
    workers: int = 4,
):
    tok = pd.read_csv(tokens_csv, dtype={"speech_id": str})
    tok = tok[tok["is_alpha"] == True].copy()
    tok["lemma"] = tok["lemma"].fillna("").astype(str)

    cleaned = pd.read_csv(
        output_dir / "cleaned_speeches.csv", dtype={"speech_id": str}
    )
    if "decade" not in cleaned.columns and "year" in cleaned.columns:
        cleaned["decade"] = cleaned["year"].map(lambda year: year_to_decade(int(year)))
    cleaned = cleaned[["speech_id", "decade"]]
    tok = tok.merge(cleaned, on="speech_id", how="left")

    model_dir = ensure_dir(output_dir / "w2v_models")
    stats = []
    for decade, sub in tok.groupby("decade"):
        sentences = [g["lemma"].tolist() for _, g in sub.groupby("speech_id")]
        if not sentences:
            continue
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
        )
        path = model_dir / f"w2v_{decade}.model"
        model.save(str(path))
        stats.append(
            {"decade": decade, "model_path": str(path), "vocab_size": len(model.wv)}
        )

    pd.DataFrame(stats).to_csv(output_dir / "w2v_model_index.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build contextual and temporal embeddings."
    )
    parser.add_argument("--cleaned-csv", type=Path, required=True)
    parser.add_argument("--sentences-csv", type=Path, required=True)
    parser.add_argument("--tokens-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--skip-contextual", action="store_true")
    parser.add_argument("--skip-w2v", action="store_true")
    parser.add_argument("--contextual-batch-size", type=int, default=16)
    parser.add_argument("--contextual-max-length", type=int, default=256)
    parser.add_argument("--contextual-model", default="microsoft/deberta-v3-base")
    parser.add_argument(
        "--seed-words-file",
        type=Path,
        default=Path("student3_semantic/target_words_full.txt"),
    )
    parser.add_argument(
        "--contextual-strict",
        action="store_true",
        help="Fail fast if contextual embedding stage errors; default is warn-and-continue.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    cleaned_df = pd.read_csv(args.cleaned_csv, dtype={"speech_id": str})
    if "decade" not in cleaned_df.columns and "year" in cleaned_df.columns:
        cleaned_df["decade"] = cleaned_df["year"].map(
            lambda year: year_to_decade(int(year))
        )
    cleaned_df.to_csv(output_dir / "cleaned_speeches.csv", index=False)

    if not args.skip_contextual:
        try:
            run_deberta_embeddings(
                args.sentences_csv,
                output_dir / "cleaned_speeches.csv",
                output_dir,
                model_name=str(args.contextual_model),
                batch_size=max(int(args.contextual_batch_size), 1),
                max_length=max(int(args.contextual_max_length), 16),
                seed_words_file=args.seed_words_file,
            )
        except Exception as exc:
            if args.contextual_strict:
                raise
            print(f"Warning: contextual embedding stage skipped due to error: {exc}")
    if not args.skip_w2v:
        run_temporal_w2v(args.tokens_csv, output_dir)

    print("Embedding generation complete.")


if __name__ == "__main__":
    main()
