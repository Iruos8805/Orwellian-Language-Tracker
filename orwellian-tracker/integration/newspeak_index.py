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

    if "Drift_rate_normalized" in sem.columns:
        sem = sem[["decade", "Drift_rate_normalized"]].rename(
            columns={"Drift_rate_normalized": "Drift_rate"}
        )
    else:
        sem = sem[["decade", "Drift_rate"]]
    merged = (
        lex[["decade", "Diversity_loss"]]
        .merge(syn[["decade", "Logic_flat"]], on="decade", how="inner")
        .merge(sem, on="decade", how="inner")
    )

    merged["Diversity_loss_raw"] = merged["Diversity_loss"]
    merged["Logic_flat_raw"] = merged["Logic_flat"]
    merged["Drift_rate_raw"] = merged["Drift_rate"]

    scaler = MinMaxScaler()
    metric_cols = ["Diversity_loss", "Logic_flat", "Drift_rate"]
    scaled = scaler.fit_transform(merged[metric_cols])
    merged[metric_cols] = scaled

    ranges = pd.DataFrame(
        {
            "metric": metric_cols,
            "min": scaler.data_min_,
            "max": scaler.data_max_,
        }
    )
    ensure_dir(args.global_scaler_file.parent)
    ranges.to_csv(args.global_scaler_file, index=False)

    merged["Time"] = merged["decade"].map(decade_to_index)
    merged["Newspeak_Index"] = (
        (merged["Diversity_loss"] + merged["Logic_flat"]) * merged["Drift_rate"]
    ) / merged["Time"]

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
