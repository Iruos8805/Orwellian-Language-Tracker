from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from preprocessing.utils import ensure_dir


def run_contextual_embeddings(
    sentences_csv: Path,
    output_dir: Path,
    model_name: str = "roberta-base",
    batch_size: int = 32,
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

    sent_df = pd.read_csv(sentences_csv, dtype={"speech_id": str, "sentence_id": str})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    speech_sums: dict[str, np.ndarray] = {}
    speech_counts: dict[str, int] = {}

    texts = sent_df["sentence_text"].fillna("").astype(str).tolist()
    speech_ids = sent_df["speech_id"].astype(str).tolist()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_speech = speech_ids[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        for sid, vec in zip(batch_speech, emb):
            if sid in speech_sums:
                speech_sums[sid] += vec
                speech_counts[sid] += 1
            else:
                speech_sums[sid] = vec.astype(np.float64)
                speech_counts[sid] = 1

    out_dir = ensure_dir(output_dir / "roberta_embeddings")
    index_rows = []
    for sid, vec_sum in speech_sums.items():
        arr = (vec_sum / max(speech_counts.get(sid, 1), 1)).astype(np.float32)
        out_path = out_dir / f"{sid}.npy"
        np.save(out_path, arr)
        index_rows.append({"speech_id": sid, "embedding_path": str(out_path)})

    pd.DataFrame(index_rows).to_csv(
        output_dir / "roberta_embedding_index.csv", index=False
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
    )[["speech_id", "decade"]]
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
    parser.add_argument("--contextual-batch-size", type=int, default=32)
    parser.add_argument(
        "--contextual-strict",
        action="store_true",
        help="Fail fast if contextual embedding stage errors; default is warn-and-continue.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    cleaned_df = pd.read_csv(args.cleaned_csv, dtype={"speech_id": str})
    cleaned_df.to_csv(output_dir / "cleaned_speeches.csv", index=False)

    if not args.skip_contextual:
        try:
            run_contextual_embeddings(
                args.sentences_csv,
                output_dir,
                batch_size=max(int(args.contextual_batch_size), 1),
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
