from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from preprocessing.utils import ensure_dir, year_to_decade


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join speech-level records with decade Newspeak index and aggregate speaker breakdowns."
    )
    parser.add_argument("--cleaned-csv", type=Path, required=True)
    parser.add_argument("--newspeak-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("integration"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    outputs_dir = ensure_dir(args.outputs_dir)

    speeches = pd.read_csv(
        args.cleaned_csv, dtype={"speech_id": str, "speaker_id": str}
    )
    if "decade" not in speeches.columns and "year" in speeches.columns:
        speeches["decade"] = speeches["year"].map(lambda year: year_to_decade(int(year)))
    idx = pd.read_csv(args.newspeak_csv)

    speech_index = speeches.merge(
        idx[["decade", "Newspeak_Index"]], on="decade", how="left"
    )

    speaker_cong = (
        speech_index.groupby("speaker_id", as_index=False)["congress"]
        .nunique()
        .rename(columns={"congress": "distinct_congresses"})
    )
    speaker_cong["seniority"] = speaker_cong["distinct_congresses"].map(
        lambda n: "first-term" if n <= 1 else "veteran"
    )

    speech_index = speech_index.merge(
        speaker_cong[["speaker_id", "seniority"]], on="speaker_id", how="left"
    )
    speech_index.to_csv(output_dir / "index_by_speaker.csv", index=False)

    by_party = speech_index.groupby(["decade", "party"], as_index=False)[
        "Newspeak_Index"
    ].mean()
    by_chamber = speech_index.groupby(["decade", "chamber"], as_index=False)[
        "Newspeak_Index"
    ].mean()
    by_state = speech_index.groupby(["decade", "state"], as_index=False)[
        "Newspeak_Index"
    ].mean()
    by_seniority = speech_index.groupby(["decade", "seniority"], as_index=False)[
        "Newspeak_Index"
    ].mean()

    by_party.to_csv(output_dir / "index_by_party.csv", index=False)
    by_chamber.to_csv(output_dir / "index_by_chamber.csv", index=False)
    by_state.to_csv(output_dir / "index_by_state.csv", index=False)
    by_seniority.to_csv(output_dir / "index_by_seniority.csv", index=False)

    speech_index.to_csv(outputs_dir / "newspeak_index_full.csv", index=False)

    print(f"Saved speaker-level index: {output_dir / 'index_by_speaker.csv'}")
    print(f"Saved party/chamber/state/seniority slices in: {output_dir}")


if __name__ == "__main__":
    main()
