from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.utils import DECADE_ORDER, ensure_dir, safe_corr


def compute_ttr(tokens: list[str]) -> float:
    if not tokens:
        return math.nan
    return len(set(tokens)) / len(tokens)


def compute_mattr(tokens: list[str], window: int = 500) -> float:
    if not tokens:
        return math.nan
    if len(tokens) < window:
        return compute_ttr(tokens)
    vals = []
    for i in range(0, len(tokens) - window + 1):
        w = tokens[i : i + window]
        vals.append(len(set(w)) / window)
    return float(np.mean(vals))


def compute_yules_k(tokens: list[str]) -> float:
    if not tokens:
        return math.nan
    freqs = Counter(tokens)
    m1 = len(tokens)
    m2 = sum(f * f for f in freqs.values())
    return 10000 * (m2 - m1) / (m1 * m1)


def compute_herdan_c(tokens: list[str]) -> float:
    if not tokens:
        return math.nan
    v = len(set(tokens))
    n = len(tokens)
    if n <= 1 or v <= 1:
        return math.nan
    return math.log(v) / math.log(n)


def build_speech_scores(cleaned: pd.DataFrame, token_df: pd.DataFrame) -> pd.DataFrame:
    token_df = token_df[token_df["is_alpha"] == True].copy()
    token_df["lemma"] = token_df["lemma"].fillna("").astype(str)

    rows = []
    for speech_id, grp in token_df.groupby("speech_id"):
        tokens = grp["lemma"].tolist()
        row = {
            "speech_id": speech_id,
            "TTR": compute_ttr(tokens),
            "MATTR": compute_mattr(tokens, window=500),
            "Yule_K": compute_yules_k(tokens),
            "Herdan_C": compute_herdan_c(tokens),
            "hapax_ratio": grp["is_hapax"].mean() if len(grp) else np.nan,
        }
        adj = grp[grp["pos"] == "ADJ"]
        row["unique_adj_rate"] = (
            (adj["lemma"].nunique() / len(adj)) if len(adj) else np.nan
        )
        rows.append(row)

    speech_scores = pd.DataFrame(rows)
    speech_scores = speech_scores.merge(
        cleaned[["speech_id", "decade", "year"]], on="speech_id", how="left"
    )
    return speech_scores


def aggregate_by_decade(speech_scores: pd.DataFrame) -> pd.DataFrame:
    dec = speech_scores.groupby("decade", as_index=False).agg(
        TTR=("TTR", "mean"),
        MATTR=("MATTR", "mean"),
        Yule_K=("Yule_K", "mean"),
        Herdan_C=("Herdan_C", "mean"),
        hapax_ratio=("hapax_ratio", "mean"),
        unique_adj_rate=("unique_adj_rate", "mean"),
    )
    dec["decade"] = pd.Categorical(dec["decade"], categories=DECADE_ORDER, ordered=True)
    dec = dec.sort_values("decade").reset_index(drop=True)
    max_mattr = dec["MATTR"].max()
    dec["Diversity_loss"] = (max_mattr - dec["MATTR"]) / max_mattr
    return dec


def fallback_vocab_validation(hein_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(hein_dir.glob("byparty_2gram_*.txt")):
        congress = p.stem.split("_")[-1]
        try:
            df = pd.read_csv(p, sep="|", dtype={"count": float})
        except Exception:
            continue
        if "phrase" not in df.columns or "count" not in df.columns:
            continue
        counts = df.groupby("phrase", as_index=False)["count"].sum()
        total = counts["count"].sum()
        unique = counts.shape[0]
        ttr = unique / total if total else np.nan
        rows.append({"congress": congress, "vocab_like_ttr": ttr})

    val = pd.DataFrame(rows)
    if val.empty:
        return val

    val["congress"] = pd.to_numeric(val["congress"], errors="coerce")
    val = val.dropna(subset=["congress"])
    val["congress"] = val["congress"].astype(int)
    val["year"] = 1789 + (val["congress"] - 1) * 2
    val["decade"] = (val["year"] // 10 * 10).astype(str) + "s"
    return val.groupby("decade", as_index=False)["vocab_like_ttr"].mean()


def save_validation_plot(
    lexical: pd.DataFrame, validation: pd.DataFrame, output_dir: Path
):
    plt.figure(figsize=(12, 5))
    plt.plot(
        lexical["decade"].astype(str), lexical["MATTR"], marker="o", label="Hein MATTR"
    )
    if not validation.empty:
        merged = lexical.merge(validation, on="decade", how="left")
        plt.plot(
            merged["decade"].astype(str),
            merged["vocab_like_ttr"],
            marker="o",
            label="Vocabulary overlay",
        )
    plt.xticks(rotation=45)
    plt.ylabel("Score")
    plt.title("Lexical diversity trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mattr_validation_overlay.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute lexical diversity metrics by decade."
    )
    parser.add_argument("--cleaned-csv", type=Path, required=True)
    parser.add_argument("--tokens-csv", type=Path, required=True)
    parser.add_argument("--hein-dir", type=Path, required=True)
    parser.add_argument("--vocab-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("student1_lexical"))
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    cleaned = pd.read_csv(args.cleaned_csv, dtype={"speech_id": str})
    tokens = pd.read_csv(args.tokens_csv, dtype={"speech_id": str})

    speech_scores = build_speech_scores(cleaned, tokens)
    lexical_decade = aggregate_by_decade(speech_scores)
    lexical_decade.to_csv(output_dir / "diversity_scores.csv", index=False)

    validation = fallback_vocab_validation(args.hein_dir)
    validation.to_csv(output_dir / "vocabulary_validation.csv", index=False)

    merged = lexical_decade.merge(validation, on="decade", how="left")
    corr = (
        safe_corr(merged["MATTR"], merged["vocab_like_ttr"])
        if not validation.empty
        else math.nan
    )
    pd.DataFrame([{"correlation": corr}]).to_csv(
        output_dir / "validation_correlation.csv", index=False
    )
    save_validation_plot(lexical_decade, validation, output_dir)

    print(f"Saved lexical scores: {output_dir / 'diversity_scores.csv'}")
    print(f"Saved validation overlay: {output_dir / 'mattr_validation_overlay.png'}")
    print(f"Validation correlation: {corr}")


if __name__ == "__main__":
    main()
