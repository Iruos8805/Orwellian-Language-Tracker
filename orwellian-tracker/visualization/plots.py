from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    out = ensure_dir(args.output_dir)
    sns.set_style("whitegrid")

    lex = pd.read_csv(args.lexical)
    syn = pd.read_csv(args.syntactic)
    sem = pd.read_csv(args.semantic)
    idx = pd.read_csv(args.index)
    party = pd.read_csv(args.party)

    plt.figure(figsize=(12, 5))
    plt.plot(lex["decade"], lex["MATTR"], marker="o", label="MATTR")
    plt.xticks(rotation=45)
    plt.ylabel("MATTR")
    plt.title("Hero graph: lexical diversity over time")
    plt.tight_layout()
    plt.savefig(out / "hero_mattr.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(lex["decade"], lex["MATTR"], marker="o", label="MATTR")
    axes[0].plot(
        lex["decade"], lex["Diversity_loss"], marker="o", label="Diversity_loss"
    )
    axes[0].legend()
    axes[1].plot(syn["decade"], syn["avg_amr_depth"], marker="o", label="AMR depth")
    axes[1].plot(syn["decade"], syn["Logic_flat"], marker="o", label="Logic_flat")
    axes[1].legend()
    axes[2].plot(sem["decade"], sem["Drift_rate"], marker="o", label="Drift_rate")
    axes[2].legend()
    axes[2].tick_params(axis="x", rotation=45)
    fig.suptitle("Three-panel decay trends")
    fig.tight_layout()
    fig.savefig(out / "three_panel_trends.png", dpi=180)
    plt.close(fig)

    pivot = party.pivot(index="party", columns="decade", values="Newspeak_Index")
    plt.figure(figsize=(12, 4))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".3f")
    plt.title("Newspeak index heatmap by party")
    plt.tight_layout()
    plt.savefig(out / "newspeak_heatmap_party.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(idx["decade"], idx["Newspeak_Index"], marker="o")
    plt.xticks(rotation=45)
    plt.ylabel("Newspeak Index")
    plt.title("Unified Newspeak Index by decade")
    plt.tight_layout()
    plt.savefig(out / "newspeak_index_line.png", dpi=180)
    plt.close()

    print(f"Saved figures to: {out}")


if __name__ == "__main__":
    main()
