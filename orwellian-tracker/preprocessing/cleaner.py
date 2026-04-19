from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

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
    return df


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
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    procedural_phrases = load_procedural_phrases(args.vocab_procedural)

    congress_filter = [c.strip() for c in args.congress_list.split(",") if c.strip()]
    df = build_cleaned_corpus(
        args.hein_dir,
        procedural_phrases,
        limit=args.limit,
        sample_per_congress=args.sample_per_congress,
        congress_filter=congress_filter,
    )

    excluded = df[df["too_short"]].copy()
    included = df[~df["too_short"]].copy()

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
    for col in base_cols:
        if col not in included.columns:
            included[col] = pd.NA

    included[base_cols].to_csv(output_dir / "cleaned_speeches.csv", index=False)
    excluded[[c for c in base_cols if c in excluded.columns]].to_csv(
        output_dir / "excluded_short_speeches.csv", index=False
    )

    for decade, sub in included.groupby("decade"):
        sub[
            ["speech_id", "year", "speaker_id", "raw_text", "clean_text", "decade"]
        ].to_csv(output_dir / f"decade_{decade}.csv", index=False)

    print(f"Saved cleaned corpus: {output_dir / 'cleaned_speeches.csv'}")
    print(f"Included speeches: {len(included):,}")
    print(f"Excluded short speeches: {len(excluded):,}")


if __name__ == "__main__":
    main()
