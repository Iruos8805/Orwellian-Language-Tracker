from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from preprocessing.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static project figures.")
    parser.add_argument("--lexical", type=Path, required=True)
    parser.add_argument("--syntactic", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--party", type=Path, required=True)
    parser.add_argument(
        "--semantic-words",
        type=Path,
        default=Path("student3_semantic/drift_scores.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    out = ensure_dir(args.output_dir)
    sns.set_style("whitegrid")

    lex = pd.read_csv(args.lexical)
    syn = pd.read_csv(args.syntactic)
    sem = pd.read_csv(args.semantic)
    idx = pd.read_csv(args.index)
    party = pd.read_csv(args.party)
    drift_words = (
        pd.read_csv(args.semantic_words)
        if args.semantic_words.exists()
        else pd.DataFrame()
    )

    if lex.empty or syn.empty or sem.empty or idx.empty:
        print("Warning: one or more input score tables are empty; skipping figure generation.")
        return

    x_lex_labels = lex["decade"].astype(str).to_numpy()
    x_lex = np.arange(len(x_lex_labels))

    plt.figure(figsize=(12, 5))
    plt.plot(x_lex, lex["MATTR"].to_numpy(), marker="o", label="MATTR")
    plt.xticks(x_lex, x_lex_labels, rotation=45)
    plt.ylabel("MATTR")
    plt.title("Hero graph: lexical diversity over time")
    plt.tight_layout()
    plt.savefig(out / "hero_mattr.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x_lex, lex["MATTR"].to_numpy(), marker="o", label="MATTR")
    axes[0].plot(x_lex, lex["Diversity_loss"].to_numpy(), marker="o", label="Diversity_loss")
    axes[0].legend()
    x_syn_labels = syn["decade"].astype(str).to_numpy()
    x_syn = np.arange(len(x_syn_labels))
    axes[1].plot(x_syn, syn["avg_amr_depth"].to_numpy(), marker="o", label="AMR depth")
    axes[1].plot(x_syn, syn["Logic_flat"].to_numpy(), marker="o", label="Logic_flat")
    axes[1].legend()
    x_sem_labels = sem["decade"].astype(str).to_numpy()
    x_sem = np.arange(len(x_sem_labels))
    axes[2].plot(x_sem, sem["Drift_rate"].to_numpy(), marker="o", label="Drift_rate")
    axes[2].legend()
    axes[2].set_xticks(x_sem)
    axes[2].set_xticklabels(x_sem_labels, rotation=45)
    fig.suptitle("Three-panel decay trends")
    fig.tight_layout()
    fig.savefig(out / "three_panel_trends.png", dpi=180)
    plt.close(fig)

    if not party.empty:
        pivot = party.pivot(index="party", columns="decade", values="Newspeak_Index")
        if not pivot.empty:
            plt.figure(figsize=(12, 4))
            sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".3f")
            plt.title("Newspeak index heatmap by party")
            plt.tight_layout()
            plt.savefig(out / "newspeak_heatmap_party.png", dpi=180)
            plt.close()

    plt.figure(figsize=(12, 5))
    x_idx_labels = idx["decade"].astype(str).to_numpy()
    x_idx = np.arange(len(x_idx_labels))
    plt.plot(x_idx, idx["Newspeak_Index"].to_numpy(), marker="o")
    plt.xticks(x_idx, x_idx_labels, rotation=45)
    plt.ylabel("Newspeak Index")
    plt.title("Unified Newspeak Index by decade")
    plt.tight_layout()
    plt.savefig(out / "newspeak_index_line.png", dpi=180)
    plt.close()

    if not drift_words.empty and {"decade_from", "decade", "word", "cosine_drift"}.issubset(
        drift_words.columns
    ):
        transitions = [
            ("1990s", "2000s"),
            ("2000s", "2010s"),
        ]
        drift_words["decade_from"] = drift_words["decade_from"].astype(str)
        drift_words["decade"] = drift_words["decade"].astype(str)
        mask = drift_words.apply(
            lambda r: (r["decade_from"], r["decade"]) in transitions, axis=1
        )
        window = drift_words[mask].copy()
        if not window.empty:
            top = (
                window.groupby("word", as_index=False)["cosine_drift"]
                .sum()
                .sort_values("cosine_drift", ascending=False)
                .head(12)
            )
            plt.figure(figsize=(10, 6))
            plt.barh(top["word"].astype(str), top["cosine_drift"].to_numpy())
            plt.gca().invert_yaxis()
            plt.xlabel("Cumulative drift (1990s→2010s)")
            plt.title("Top semantic shifts across 1990s–2010s")
            plt.tight_layout()
            plt.savefig(out / "semantic_shift_words.png", dpi=180)
            plt.close()

            top_words = top["word"].astype(str).tolist()[:6]
            trend = (
                window[window["word"].isin(top_words)]
                .groupby(["word", "decade"], as_index=False)["cosine_drift"]
                .mean()
            )
            plt.figure(figsize=(10, 6))
            for w in top_words:
                sub = trend[trend["word"] == w]
                x = np.arange(len(sub["decade"]))
                plt.plot(x, sub["cosine_drift"].to_numpy(), marker="o", label=w)
                plt.xticks(x, sub["decade"].astype(str).to_numpy(), rotation=45)
            plt.ylabel("Cosine drift")
            plt.title("Drift trajectories for top shifting words")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / "semantic_shift_trajectories.png", dpi=180)
            plt.close()

    print(f"Saved figures to: {out}")


if __name__ == "__main__":
    main()
