from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.utils import DECADE_ORDER, ensure_dir


def proxy_lens_score(depth: float, nodes: float, sub_ratio: float) -> float:
    return float(
        max(0.0, min(1.0, 0.4 * (depth / 8.0) + 0.4 * (nodes / 60.0) + 0.2 * sub_ratio))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute syntactic complexity metrics by decade."
    )
    parser.add_argument("--amr-metrics", type=Path, required=True)
    parser.add_argument("--speech-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("student2_syntactic"))
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    amr = pd.read_csv(args.amr_metrics, dtype={"speech_id": str})
    stats = pd.read_csv(args.speech_stats, dtype={"speech_id": str})

    merged = amr.merge(stats, on="speech_id", how="inner", suffixes=("_amr", "_tok"))

    sent_col = "sentence_count_tok" if "sentence_count_tok" in merged.columns else "sentence_count"
    sub_col = (
        "subordinate_clause_count_tok"
        if "subordinate_clause_count_tok" in merged.columns
        else "subordinate_clause_count"
    )
    merged["subordination_ratio"] = merged[sub_col] / merged[sent_col].replace(0, np.nan)
    merged["mean_LENS"] = merged.apply(
        lambda r: proxy_lens_score(
            r["avg_amr_depth"],
            r["avg_amr_nodes"],
            r["subordination_ratio"] if pd.notna(r["subordination_ratio"]) else 0.0,
        ),
        axis=1,
    )

    out = (
        merged.groupby("decade", as_index=False)
        .agg(
            avg_amr_depth=("avg_amr_depth", "mean"),
            avg_amr_nodes=("avg_amr_nodes", "mean"),
            mean_LENS=("mean_LENS", "mean"),
            subordination_ratio=("subordination_ratio", "mean"),
            std_amr_depth=("avg_amr_depth", "std"),
            std_amr_nodes=("avg_amr_nodes", "std"),
        )
        .reset_index(drop=True)
    )
    out["decade"] = pd.Categorical(out["decade"], categories=DECADE_ORDER, ordered=True)
    out = out.sort_values("decade").reset_index(drop=True)

    max_depth = out["avg_amr_depth"].max()
    out["Logic_flat"] = (max_depth - out["avg_amr_depth"]) / max_depth

    out.to_csv(output_dir / "depth_scores.csv", index=False)

    plt.figure(figsize=(12, 5))
    x_labels = out["decade"].astype(str).to_numpy()
    x = np.arange(len(x_labels))
    plt.plot(x, out["avg_amr_depth"].to_numpy(), marker="o", label="AMR depth")
    plt.plot(x, out["subordination_ratio"].to_numpy(), marker="o", label="Subordination ratio")
    plt.xticks(x, x_labels, rotation=45)
    plt.ylabel("Score")
    plt.title("Syntactic depth trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "syntactic_trends.png", dpi=180)
    plt.close()

    print(f"Saved syntactic scores: {output_dir / 'depth_scores.csv'}")


if __name__ == "__main__":
    main()
