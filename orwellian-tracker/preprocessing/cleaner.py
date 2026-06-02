from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocessing.utils import ensure_dir, year_to_decade


COMMON_BOILERPLATE_PATTERNS = [
    r"\bmr\.?\s+speaker\b",
    r"\bmadam\s+speaker\b",
    r"\bi\s+yield\b",
    r"\bwithout\s+objection\b",
    r"\bthe\s+clerk\s+will\s+read\b",
    r"\bthe\s+chair\s+recognizes\b",
    r"\border\s+in\s+the\s+house\b",
]


def clean_text(text: str, procedural_phrases: list[str]) -> str:
    text = str(text)
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = text.replace("\x00", " ")
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text)

    lowered = text.lower()
    for pattern in COMMON_BOILERPLATE_PATTERNS:
        lowered = re.sub(pattern, " ", lowered)

    for phrase in procedural_phrases:
        if phrase:
            lowered = lowered.replace(f" {phrase} ", " ")

    lowered = re.sub(r"[^a-z0-9\s'\-]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def year_to_period_label(year: int, bin_edges: list[tuple[int, int]]) -> str:
    for start, end in bin_edges:
        if start <= year <= end:
            return f"{start}-{end}"
    return year_to_decade(year)


def parse_year_bins(spec: str) -> list[tuple[int, int]]:
    chunks = [c.strip() for c in str(spec).split(",") if c.strip()]
    out = []
    for ch in chunks:
        if "-" not in ch:
            continue
        left, right = ch.split("-", 1)
        if left.strip().isdigit() and right.strip().isdigit():
            a, b = int(left.strip()), int(right.strip())
            if a <= b:
                out.append((a, b))
    return out


def apply_bin_mode(df: pd.DataFrame, mode: str, bin_edges: list[tuple[int, int]]) -> pd.DataFrame:
    mode = (mode or "decade").strip().lower()
    if mode == "year":
        df["decade"] = df["year"].astype(int).astype(str)
        return df
    if mode == "custom":
        if not bin_edges:
            raise ValueError("--bin-mode custom requires --year-bins")
        df["decade"] = df["year"].map(lambda y: year_to_period_label(int(y), bin_edges))
        return df
    df["decade"] = df["year"].map(year_to_decade)
    return df


def word_count(text: str) -> int:
    return len(re.findall(r"\b[a-zA-Z][a-zA-Z'\-]*\b", text))


def load_procedural_phrases(path: Path | None, max_phrases: int = 1000) -> list[str]:
    if path is None or not path.exists():
        return []
    df = pd.read_csv(path, sep="|", dtype=str)
    if "phrase" not in df.columns:
        return []
    phrases = []
    for phrase in df["phrase"].fillna("").astype(str).tolist():
        p = phrase.strip().lower()
        if len(p.split()) >= 2 and p[0].isalpha():
            phrases.append(p)
        if len(phrases) >= max_phrases:
            break
    return phrases


def read_congress_bundle(hein_dir: Path, congress: str) -> pd.DataFrame:
    descr_path = hein_dir / f"descr_{congress}.txt"
    speeches_path = hein_dir / f"speeches_{congress}.txt"
    speaker_map_path = hein_dir / f"{congress}_SpeakerMap.txt"

    if not descr_path.exists() or not speeches_path.exists():
        return pd.DataFrame()

    descr = pd.read_csv(descr_path, sep="|", dtype=str, encoding="latin-1")
    speeches = read_speeches_file(speeches_path)

    merged = descr.merge(speeches, on="speech_id", how="left")
    merged["congress"] = int(congress)

    if speaker_map_path.exists():
        smap = pd.read_csv(speaker_map_path, sep="|", dtype=str, encoding="latin-1")
        smap = smap.rename(columns={"speakerid": "speaker_id"})
        keep_cols = [
            c
            for c in ["speech_id", "speaker_id", "party", "state", "chamber"]
            if c in smap.columns
        ]
        merged = merged.merge(
            smap[keep_cols], on="speech_id", how="left", suffixes=("", "_map")
        )
    else:
        merged["speaker_id"] = pd.NA

    merged["date"] = merged["date"].astype(str)
    merged["year"] = pd.to_numeric(merged["date"].str[:4], errors="coerce")
    merged = merged.dropna(subset=["year"]).copy()
    merged["year"] = merged["year"].astype(int)
    merged["decade"] = merged["year"].map(year_to_decade)
    return merged


def read_speeches_file(path: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="latin-1", errors="ignore") as fh:
        header = fh.readline()
        if "speech_id|speech" not in header:
            raise ValueError(f"Unexpected speeches header in {path}")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if "|" not in line:
                continue
            speech_id, speech = line.split("|", 1)
            rows.append({"speech_id": speech_id, "speech": speech})
    return pd.DataFrame(rows)


def build_cleaned_corpus(
    hein_dir: Path,
    procedural_phrases: list[str],
    limit: int = 0,
    sample_per_congress: int = 0,
    congress_filter: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    bin_edges: list[tuple[int, int]] | None = None,
    bin_mode: str = "decade",
) -> pd.DataFrame:
    descr_files = sorted(hein_dir.glob("descr_*.txt"))
    congresses = [f.stem.split("_")[-1] for f in descr_files]
    if congress_filter:
        wanted = set(congress_filter)
        congresses = [c for c in congresses if c in wanted]
    frames = []
    remaining = limit if limit and limit > 0 else None
    for congress in congresses:
        frame = read_congress_bundle(hein_dir, congress)
        if not frame.empty:
            if sample_per_congress and sample_per_congress > 0:
                frame = frame.head(sample_per_congress).copy()
            if remaining is not None:
                if remaining <= 0:
                    break
                frame = frame.head(remaining).copy()
                remaining -= len(frame)
            frames.append(frame)
        if remaining is not None and remaining <= 0:
            break
    if not frames:
        raise FileNotFoundError(
            "No valid descr/speeches bundles found in hein directory."
        )

    df = pd.concat(frames, ignore_index=True)
    df["raw_text"] = df["speech"].fillna("").astype(str)
    df["clean_text"] = df["raw_text"].map(lambda t: clean_text(t, procedural_phrases))
    df["clean_word_count"] = df["clean_text"].map(word_count)
    df["too_short"] = df["clean_word_count"] < 50

    if min_year is not None:
        df = df[df["year"] >= int(min_year)].copy()
    if max_year is not None:
        df = df[df["year"] <= int(max_year)].copy()
    df = apply_bin_mode(df, bin_mode, bin_edges or [])

    return df


def list_congresses(hein_dir: Path, congress_filter: list[str] | None = None) -> list[str]:
    descr_files = sorted(hein_dir.glob("descr_*.txt"))
    congresses = [f.stem.split("_")[-1] for f in descr_files]
    if congress_filter:
        wanted = set(congress_filter)
        congresses = [c for c in congresses if c in wanted]
    return congresses


def stream_cleaned_outputs(
    hein_dir: Path,
    procedural_phrases: list[str],
    output_dir: Path,
    limit: int = 0,
    sample_per_congress: int = 0,
    congress_filter: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    bin_edges: list[tuple[int, int]] | None = None,
    bin_mode: str = "decade",
    balance_decades: bool = False,
) -> tuple[int, int]:
    csv_kwargs = {
        "quoting": csv.QUOTE_MINIMAL,
        "quotechar": '"',
        "escapechar": "\\",
    }

    base_cols = [
        "speech_id",
        "year",
        "speaker_id",
        "raw_text",
        "clean_text",
        "decade",
        "congress",
        "party",
        "chamber",
        "state",
        "clean_word_count",
    ]

    cleaned_out = output_dir / "cleaned_speeches.csv"
    balanced_out = output_dir / "cleaned_speeches_balanced.csv"
    excluded_out = output_dir / "excluded_short_speeches.csv"
    for decade_path in output_dir.glob("decade_*.csv"):
        decade_path.unlink()
    for path in [cleaned_out, excluded_out]:
        if path.exists():
            path.unlink()
    if balanced_out.exists():
        balanced_out.unlink()

    decade_written: set[str] = set()
    congresses = list_congresses(hein_dir, congress_filter)
    remaining = limit if limit and limit > 0 else None
    included_total = 0
    excluded_total = 0

    for congress in congresses:
        frame = read_congress_bundle(hein_dir, congress)
        if frame.empty:
            continue

        if sample_per_congress and sample_per_congress > 0:
            frame = frame.head(sample_per_congress).copy()

        if remaining is not None:
            if remaining <= 0:
                break
            frame = frame.head(remaining).copy()
            remaining -= len(frame)

        frame["raw_text"] = frame["speech"].fillna("").astype(str)
        frame["clean_text"] = frame["raw_text"].map(
            lambda t: clean_text(t, procedural_phrases)
        )
        frame["clean_word_count"] = frame["clean_text"].map(word_count)
        frame["too_short"] = frame["clean_word_count"] < 50

        if min_year is not None:
            frame = frame[frame["year"] >= int(min_year)].copy()
        if max_year is not None:
            frame = frame[frame["year"] <= int(max_year)].copy()
        frame = apply_bin_mode(frame, bin_mode, bin_edges or [])

        included = frame[~frame["too_short"]].copy()
        excluded = frame[frame["too_short"]].copy()

        for col in base_cols:
            if col not in included.columns:
                included[col] = pd.NA
            if col not in excluded.columns:
                excluded[col] = pd.NA

        if not included.empty:
            included[base_cols].to_csv(
                cleaned_out,
                index=False,
                mode="a",
                header=not cleaned_out.exists(),
                **csv_kwargs,
            )
            included_total += len(included)

            for decade, sub in included.groupby("decade"):
                decade_file = output_dir / f"decade_{decade}.csv"
                sub[
                    [
                        "speech_id",
                        "year",
                        "speaker_id",
                        "raw_text",
                        "clean_text",
                        "decade",
                    ]
                ].to_csv(
                    decade_file,
                    index=False,
                    mode="a",
                    header=(decade not in decade_written and not decade_file.exists()),
                    **csv_kwargs,
                )
                decade_written.add(str(decade))

        if not excluded.empty:
            excluded[base_cols].to_csv(
                excluded_out,
                index=False,
                mode="a",
                header=not excluded_out.exists(),
                **csv_kwargs,
            )
            excluded_total += len(excluded)

        del frame, included, excluded

        if remaining is not None and remaining <= 0:
            break

    if included_total == 0 and excluded_total == 0:
        raise FileNotFoundError("No valid descr/speeches bundles found in hein directory.")

    if balance_decades:
        df = pd.read_csv(cleaned_out, dtype={"speech_id": str})
        if not df.empty and "decade" in df.columns:
            counts = df["decade"].value_counts()
            if not counts.empty:
                target = int(counts.min())
                balanced = (
                    df.groupby("decade", group_keys=False)
                    .apply(lambda g: g.sample(n=target, random_state=42))
                    .reset_index(drop=True)
                )
                balanced.to_csv(balanced_out, index=False)

    return included_total, excluded_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and decade-bin Hein Daily speeches."
    )
    parser.add_argument("--hein-dir", type=Path, required=True)
    parser.add_argument("--vocab-procedural", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-per-congress", type=int, default=0)
    parser.add_argument("--congress-list", default="")
    parser.add_argument("--min-year", type=int, default=0)
    parser.add_argument("--max-year", type=int, default=0)
    parser.add_argument("--year-bins", default="")
    parser.add_argument("--bin-mode", default="decade", choices=["decade", "year", "custom"])
    parser.add_argument("--balance-decades", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    procedural_phrases = load_procedural_phrases(args.vocab_procedural)

    congress_filter = [c.strip() for c in args.congress_list.split(",") if c.strip()]
    bin_edges = parse_year_bins(args.year_bins)
    included_count, excluded_count = stream_cleaned_outputs(
        args.hein_dir,
        procedural_phrases,
        output_dir=output_dir,
        limit=args.limit,
        sample_per_congress=args.sample_per_congress,
        congress_filter=congress_filter,
        min_year=(args.min_year if args.min_year > 0 else None),
        max_year=(args.max_year if args.max_year > 0 else None),
        bin_edges=bin_edges,
        bin_mode=args.bin_mode,
        balance_decades=args.balance_decades,
    )

    print(f"Saved cleaned corpus: {output_dir / 'cleaned_speeches.csv'}")
    if args.balance_decades:
        print(f"Saved balanced corpus: {output_dir / 'cleaned_speeches_balanced.csv'}")
    print(f"Included speeches: {included_count:,}")
    print(f"Excluded short speeches: {excluded_count:,}")


if __name__ == "__main__":
    main()
