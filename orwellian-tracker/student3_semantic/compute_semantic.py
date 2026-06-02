from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler

from preprocessing.utils import DECADE_ORDER, ensure_dir


def load_targets(path: Path) -> list[str]:
    seen = set()
    ordered = []
    for line in path.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if not w or w in seen:
            continue
        seen.add(w)
        ordered.append(w)
    return ordered


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def model_for_decade(w2v_dir: Path, decade: str) -> Word2Vec | None:
    path = w2v_dir / f"w2v_{decade}.model"
    if not path.exists():
        return None
    return Word2Vec.load(str(path))


def write_neighbor_shifts(
    w2v_dir: Path,
    targets: list[str],
    decade_from: str,
    decade_to: str,
    topn: int,
    output_path: Path,
) -> None:
    m1 = model_for_decade(w2v_dir, decade_from)
    m2 = model_for_decade(w2v_dir, decade_to)
    if m1 is None or m2 is None:
        pd.DataFrame(
            columns=["word", "decade_from", "neighbors_from", "decade_to", "neighbors_to"]
        ).to_csv(output_path, index=False)
        return

    rows = []
    for word in targets:
        if word not in m1.wv.key_to_index or word not in m2.wv.key_to_index:
            continue
        try:
            n1 = [w for w, _ in m1.wv.most_similar(word, topn=topn)]
            n2 = [w for w, _ in m2.wv.most_similar(word, topn=topn)]
        except Exception:
            continue
        rows.append(
            {
                "word": word,
                "decade_from": decade_from,
                "neighbors_from": ", ".join(n1),
                "decade_to": decade_to,
                "neighbors_to": ", ".join(n2),
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def write_shift_summary(
    drift: pd.DataFrame,
    neighbor_path: Path,
    decade_from: str,
    decade_to: str,
    output_path: Path,
) -> None:
    if drift.empty:
        pd.DataFrame(
            columns=[
                "word",
                "decade_from",
                "decade_to",
                "shift_score",
                "neighbors_from",
                "neighbors_to",
            ]
        ).to_csv(output_path, index=False)
        return

    drift = drift.copy()
    drift["decade_from"] = drift["decade_from"].astype(str)
    drift["decade"] = drift["decade"].astype(str)
    transitions = [(decade_from, "2000s"), ("2000s", decade_to)]
    window = drift[drift.apply(lambda r: (r["decade_from"], r["decade"]) in transitions, axis=1)]

    if window.empty:
        pd.DataFrame(
            columns=[
                "word",
                "decade_from",
                "decade_to",
                "shift_score",
                "neighbors_from",
                "neighbors_to",
            ]
        ).to_csv(output_path, index=False)
        return

    scores = (
        window.groupby("word", as_index=False)["cosine_drift"]
        .sum()
        .rename(columns={"cosine_drift": "shift_score"})
    )

    if neighbor_path.exists():
        neighbors = pd.read_csv(neighbor_path)
        neighbors["decade_from"] = neighbors["decade_from"].astype(str)
        neighbors["decade_to"] = neighbors["decade_to"].astype(str)
        neighbors = neighbors[(neighbors["decade_from"] == decade_from) & (neighbors["decade_to"] == decade_to)]
        out = scores.merge(neighbors, on="word", how="left")
        out = out[["word", "decade_from", "decade_to", "shift_score", "neighbors_from", "neighbors_to"]]
    else:
        out = scores.copy()
        out["decade_from"] = decade_from
        out["decade_to"] = decade_to
        out["neighbors_from"] = ""
        out["neighbors_to"] = ""

    out = out.sort_values("shift_score", ascending=False)
    out.to_csv(output_path, index=False)


def orthogonal_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = source - source.mean(axis=0, keepdims=True)
    tgt = target - target.mean(axis=0, keepdims=True)
    m = src.T @ tgt
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt


def get_shared_vocab(m1: Word2Vec, m2: Word2Vec, min_count: int) -> list[str]:
    shared = set(m1.wv.key_to_index).intersection(set(m2.wv.key_to_index))
    kept = []
    for w in shared:
        c1 = m1.wv.get_vecattr(w, "count") if hasattr(m1.wv, "get_vecattr") else min_count
        c2 = m2.wv.get_vecattr(w, "count") if hasattr(m2.wv, "get_vecattr") else min_count
        if c1 >= min_count and c2 >= min_count:
            kept.append(w)
    return kept


def deberta_proxy(cleaned: pd.DataFrame, word: str, early_mask, late_mask) -> float:
    early_rate = (
        cleaned[early_mask]["clean_text"]
        .str.contains(rf"\b{word}\b", regex=True)
        .mean()
    )
    late_rate = (
        cleaned[late_mask]["clean_text"].str.contains(rf"\b{word}\b", regex=True).mean()
    )
    return float(abs((late_rate or 0.0) - (early_rate or 0.0)))


def deberta_seed_drift(index_path: Path) -> pd.DataFrame:
    if not index_path.exists():
        return pd.DataFrame(columns=["decade", "deberta_seed_drift"])
    index = pd.read_csv(index_path)
    if index.empty:
        return pd.DataFrame(columns=["decade", "deberta_seed_drift"])
    rows = []
    for r in index.itertuples(index=False):
        path = Path(r.embedding_path)
        if not path.exists():
            continue
        vec = np.load(path)
        if vec.ndim != 1:
            continue
        rows.append({"decade": str(r.decade), "word": str(r.word), "vec": vec})
    if not rows:
        return pd.DataFrame(columns=["decade", "deberta_seed_drift"])
    df = pd.DataFrame(rows)
    decades = sorted(df["decade"].unique(), key=lambda d: DECADE_ORDER.index(d) if d in DECADE_ORDER else 9999)
    drift_rows = []
    for i in range(1, len(decades)):
        d_prev = decades[i - 1]
        d_curr = decades[i]
        prev = df[df["decade"] == d_prev].set_index("word")["vec"]
        curr = df[df["decade"] == d_curr].set_index("word")["vec"]
        shared = prev.index.intersection(curr.index)
        if shared.empty:
            continue
        sims = []
        for w in shared:
            a = prev[w]
            b = curr[w]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                continue
            sims.append(float(np.dot(a, b) / (na * nb)))
        if sims:
            drift_rows.append({"decade": d_curr, "deberta_seed_drift": float(1.0 - np.mean(sims))})
    return pd.DataFrame(drift_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute semantic drift metrics by decade."
    )
    parser.add_argument("--cleaned-csv", type=Path, required=True)
    parser.add_argument("--w2v-dir", type=Path, required=True)
    parser.add_argument("--targets-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("student3_semantic"))
    parser.add_argument(
        "--deberta-seed-index",
        type=Path,
        default=Path("data/processed/deberta_embeddings/seed_word_index.csv"),
    )
    parser.add_argument("--min-shared-vocab", type=int, default=2000)
    parser.add_argument("--min-word-count", type=int, default=20)
    parser.add_argument("--neighbor-decade-from", default="1990s")
    parser.add_argument("--neighbor-decade-to", default="2010s")
    parser.add_argument("--neighbor-topn", type=int, default=5)
    parser.add_argument(
        "--neighbor-output",
        type=Path,
        default=Path("student3_semantic/word_shift_neighbors.csv"),
    )
    parser.add_argument(
        "--neighbor-summary-output",
        type=Path,
        default=Path("student3_semantic/word_shift_summary.csv"),
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    cleaned = pd.read_csv(args.cleaned_csv, dtype={"speech_id": str})
    targets = load_targets(args.targets_file)

    available_decades = sorted(
        {
            p.stem.replace("w2v_", "")
            for p in args.w2v_dir.glob("w2v_*.model")
            if p.stem.replace("w2v_", "") in DECADE_ORDER
        },
        key=lambda d: DECADE_ORDER.index(d),
    )

    rows = []
    coverage_rows = []
    for i in range(len(available_decades) - 1):
        d1, d2 = available_decades[i], available_decades[i + 1]
        m1 = model_for_decade(args.w2v_dir, d1)
        m2 = model_for_decade(args.w2v_dir, d2)
        if m1 is None or m2 is None:
            continue

        shared_vocab = get_shared_vocab(m1, m2, min_count=max(args.min_word_count, 1))
        if len(shared_vocab) < max(args.min_shared_vocab, 10):
            continue

        x = np.vstack([m1.wv[w] for w in shared_vocab])
        y = np.vstack([m2.wv[w] for w in shared_vocab])
        rot = orthogonal_align(y, x)

        for word in targets:
            present_d1 = word in m1.wv.key_to_index
            present_d2 = word in m2.wv.key_to_index
            coverage_rows.append(
                {
                    "decade_from": d1,
                    "decade_to": d2,
                    "word": word,
                    "present_from": present_d1,
                    "present_to": present_d2,
                    "shared_anchor_vocab": len(shared_vocab),
                }
            )
            if not present_d1 or not present_d2:
                continue

            c1 = m1.wv.get_vecattr(word, "count") if hasattr(m1.wv, "get_vecattr") else args.min_word_count
            c2 = m2.wv.get_vecattr(word, "count") if hasattr(m2.wv, "get_vecattr") else args.min_word_count
            if c1 < args.min_word_count or c2 < args.min_word_count:
                continue

            vec1 = m1.wv[word]
            vec2 = m2.wv[word] @ rot
            cos = cosine(vec1, vec2)
            drift = 1 - cos if pd.notna(cos) else np.nan
            rows.append(
                {
                    "decade_from": d1,
                    "decade": d2,
                    "word": word,
                    "cosine_drift": drift,
                }
            )

    drift = pd.DataFrame(rows)
    if drift.empty:
        pd.DataFrame(coverage_rows).to_csv(output_dir / "word_coverage.csv", index=False)
        pd.DataFrame(columns=["decade", "Drift_rate", "deberta_drift", "cluster_migration", "Drift_rate_normalized"]).to_csv(
            output_dir / "drift_rate_by_decade.csv", index=False
        )
        pd.DataFrame(columns=["decade_from", "decade", "word", "cosine_drift", "deberta_drift", "cluster_migration"]).to_csv(
            output_dir / "drift_scores.csv", index=False
        )
        print("No target words found across adjacent decade models after filtering.")
        print(f"Saved coverage report: {output_dir / 'word_coverage.csv'}")
        write_neighbor_shifts(
            args.w2v_dir,
            targets,
            str(args.neighbor_decade_from),
            str(args.neighbor_decade_to),
            max(args.neighbor_topn, 1),
            args.neighbor_output,
        )
        write_shift_summary(
            drift,
            args.neighbor_output,
            str(args.neighbor_decade_from),
            str(args.neighbor_decade_to),
            args.neighbor_summary_output,
        )
        return

    early_mask = cleaned["year"].between(1873, 1920)
    late_mask = cleaned["year"].between(1990, 2017)
    deberta_rows = []
    for w in targets:
        deberta_rows.append(
            {
                "word": w,
                "deberta_drift": deberta_proxy(cleaned, w, early_mask, late_mask),
            }
        )
    deberta_df = pd.DataFrame(deberta_rows)

    drift = drift.merge(deberta_df, on="word", how="left")
    drift["cluster_migration"] = (
        drift["cosine_drift"] * 0.7 + drift["deberta_drift"] * 0.3
    )
    drift.to_csv(output_dir / "drift_scores.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output_dir / "word_coverage.csv", index=False)

    decade = drift.groupby("decade", as_index=False).agg(
        Drift_rate=("cosine_drift", "mean"),
        deberta_drift=("deberta_drift", "mean"),
        cluster_migration=("cluster_migration", "mean"),
    )

    seed_drift = deberta_seed_drift(args.deberta_seed_index)
    if not seed_drift.empty:
        decade = decade.merge(seed_drift, on="decade", how="left")
    scaler = MinMaxScaler()
    decade["Drift_rate_normalized"] = scaler.fit_transform(decade[["Drift_rate"]])
    decade.to_csv(output_dir / "drift_rate_by_decade.csv", index=False)

    write_neighbor_shifts(
        args.w2v_dir,
        targets,
        str(args.neighbor_decade_from),
        str(args.neighbor_decade_to),
        max(args.neighbor_topn, 1),
        args.neighbor_output,
    )
    write_shift_summary(
        drift,
        args.neighbor_output,
        str(args.neighbor_decade_from),
        str(args.neighbor_decade_to),
        args.neighbor_summary_output,
    )

    plt.figure(figsize=(12, 5))
    decade_order = [d for d in DECADE_ORDER if d in set(drift["decade"].astype(str))]
    x_map = {d: i for i, d in enumerate(decade_order)}
    for w in targets[:10]:
        sub = drift[drift["word"] == w].copy()
        if sub.empty:
            continue
        sub["x"] = sub["decade"].astype(str).map(x_map)
        sub = sub.sort_values("x")
        plt.plot(sub["x"].to_numpy(), sub["cosine_drift"].to_numpy(), marker="o", alpha=0.6)
    plt.xticks(list(x_map.values()), list(x_map.keys()), rotation=45)
    plt.ylabel("Cosine drift")
    plt.title("Target-word semantic drift across decades")
    plt.tight_layout()
    plt.savefig(output_dir / "semantic_drift_lines.png", dpi=180)
    plt.close()

    print(f"Saved semantic drift word-level: {output_dir / 'drift_scores.csv'}")
    print(f"Saved semantic decade-level: {output_dir / 'drift_rate_by_decade.csv'}")
    print(f"Saved coverage report: {output_dir / 'word_coverage.csv'}")


if __name__ == "__main__":
    main()
