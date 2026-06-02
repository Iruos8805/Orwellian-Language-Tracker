from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from preprocessing.utils import DECADE_ORDER, decade_to_index, ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Newspeak Index by decade.")
    parser.add_argument("--lexical", type=Path, required=True)
    parser.add_argument("--syntactic", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("integration"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--global-scaler-file", type=Path, default=Path("integration/global_scaler_ranges.csv"))
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    outputs_dir = ensure_dir(args.outputs_dir)

    lex = pd.read_csv(args.lexical)
    syn = pd.read_csv(args.syntactic)
    sem = pd.read_csv(args.semantic)

    if "Drift_rate" in sem.columns:
        sem = sem[["decade", "Drift_rate"]]
    elif "Drift_rate_normalized" in sem.columns:
        sem = sem[["decade", "Drift_rate_normalized"]].rename(
            columns={"Drift_rate_normalized": "Drift_rate"}
        )
    else:
        sem = sem[["decade"]]
        sem["Drift_rate"] = 0.0
    merged = (
        lex[["decade", "Diversity_loss"]]
        .merge(syn[["decade", "Logic_flat"]], on="decade", how="inner")
        .merge(sem, on="decade", how="inner")
    )

    if merged.empty:
        output_dir = ensure_dir(args.output_dir)
        outputs_dir = ensure_dir(args.outputs_dir)
        empty = pd.DataFrame(
            columns=[
                "decade",
                "Diversity_loss",
                "Logic_flat",
                "Drift_rate",
                "Diversity_loss_raw",
                "Logic_flat_raw",
                "Drift_rate_raw",
                "Time",
                "Newspeak_Index",
            ]
        )
        empty.to_csv(output_dir / "newspeak_index.csv", index=False)
        empty.to_csv(outputs_dir / "newspeak_index_full.csv", index=False)
        ensure_dir(args.global_scaler_file.parent)
        pd.DataFrame(columns=["metric", "min", "max"]).to_csv(
            args.global_scaler_file, index=False
        )
        with (outputs_dir / "decade_scores.json").open("w", encoding="utf-8") as fh:
            json.dump({}, fh, indent=2)
        print("Warning: empty merge for Newspeak index; wrote empty outputs.")
        return

    merged["Diversity_loss_raw"] = merged["Diversity_loss"]
    merged["Logic_flat_raw"] = merged["Logic_flat"]
    merged["Drift_rate_raw"] = merged["Drift_rate"]

    metric_cols = ["Diversity_loss", "Logic_flat", "Drift_rate"]
    ranges = pd.DataFrame(
        {
            "metric": metric_cols,
            "min": merged[metric_cols].min().to_numpy(),
            "max": merged[metric_cols].max().to_numpy(),
        }
    )
    ensure_dir(args.global_scaler_file.parent)
    ranges.to_csv(args.global_scaler_file, index=False)

    merged["Time"] = merged["decade"].map(decade_to_index)
    merged["Score_raw"] = (
        (merged["Diversity_loss"] + merged["Logic_flat"]) * merged["Drift_rate"]
    ) / merged["Time"]

    baseline_decade = "1870s"
    if baseline_decade in set(merged["decade"].astype(str)):
        baseline_score = float(
            merged.loc[merged["decade"].astype(str) == baseline_decade, "Score_raw"].iloc[0]
        )
    else:
        merged_sorted = merged.copy()
        merged_sorted["decade"] = pd.Categorical(
            merged_sorted["decade"], categories=DECADE_ORDER, ordered=True
        )
        merged_sorted = merged_sorted.sort_values("decade").reset_index(drop=True)
        baseline_score = float(merged_sorted["Score_raw"].iloc[0])

    if baseline_score == 0.0:
        baseline_score = 1e-9

    merged["Newspeak_Index"] = (merged["Score_raw"] - baseline_score) / baseline_score

    merged["decade"] = pd.Categorical(
        merged["decade"], categories=DECADE_ORDER, ordered=True
    )
    merged = merged.sort_values("decade").reset_index(drop=True)

    merged.to_csv(output_dir / "newspeak_index.csv", index=False)
    merged.to_csv(outputs_dir / "newspeak_index_full.csv", index=False)

    payload = {
        row["decade"]: {k: v for k, v in row.items() if k != "decade"}
        for row in merged.to_dict(orient="records")
    }
    with (outputs_dir / "decade_scores.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Saved decade index: {output_dir / 'newspeak_index.csv'}")
    print(f"Saved outputs JSON: {outputs_dir / 'decade_scores.json'}")
    print(f"Saved global scaler ranges: {args.global_scaler_file}")


if __name__ == "__main__":
    main()
