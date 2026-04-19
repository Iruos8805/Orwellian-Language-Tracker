from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import networkx as nx
import pandas as pd

from preprocessing.utils import ensure_dir


def _load_amr_model():
    try:
        import amrlib

        model = amrlib.load_stog_model()
        return model
    except Exception:
        return None


def _parse_depth(amr_text: str):
    try:
        import penman

        graph = penman.decode(amr_text)
        root = graph.top
        g = nx.DiGraph()
        for src, rel, tgt in graph.triples:
            if rel == ":instance":
                continue
            if isinstance(tgt, str) and tgt.startswith('"'):
                continue
            g.add_edge(src, tgt)
        if root is None or root not in g:
            nodes = len(
                {s for s, _, _ in graph.triples} | {t for _, _, t in graph.triples}
            )
            edges = len(graph.triples)
            return max(nodes, 1), max(edges, 0), 1

        max_depth = 1
        for node in nx.descendants(g, root):
            try:
                depth = nx.shortest_path_length(g, root, node) + 1
                if depth > max_depth:
                    max_depth = depth
            except nx.NetworkXNoPath:
                continue
        return max(g.number_of_nodes(), 1), g.number_of_edges(), max_depth
    except Exception:
        return 1, 0, 1


def _heuristic_metrics(sentence: str):
    words = sentence.split()
    n = max(len(words), 1)
    depth = 1 + min(6, n // 12)
    nodes = n
    edges = max(n - 1, 0)
    return nodes, edges, depth


def _worker_process(chunk: list[tuple[str, str, str]], amr_enabled: bool):
    model = _load_amr_model() if amr_enabled else None
    rows = []
    for speech_id, sentence_id, sentence_text in chunk:
        if model is not None:
            try:
                amr_text = model.parse_sents([sentence_text])[0]
                nodes, edges, depth = _parse_depth(amr_text)
            except Exception:
                amr_text = ""
                nodes, edges, depth = _heuristic_metrics(sentence_text)
        else:
            amr_text = ""
            nodes, edges, depth = _heuristic_metrics(sentence_text)

        rows.append(
            {
                "speech_id": speech_id,
                "sentence_id": sentence_id,
                "sentence_text": sentence_text,
                "amr_penman": amr_text,
                "amr_nodes": nodes,
                "amr_edges": edges,
                "amr_depth": depth,
            }
        )
    return rows


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse sentence AMR and aggregate speech-level AMR metrics."
    )
    parser.add_argument("--sentences-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--disable-amr", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    amr_dir = ensure_dir(output_dir / "amr_penman")

    sent_df = pd.read_csv(
        args.sentences_csv, dtype={"speech_id": str, "sentence_id": str}
    )
    if args.limit and args.limit > 0:
        sent_df = sent_df.head(args.limit)

    tuples = [
        (str(r.speech_id), str(r.sentence_id), str(r.sentence_text))
        for r in sent_df[["speech_id", "sentence_id", "sentence_text"]].itertuples(
            index=False
        )
    ]

    chunks = list(chunked(tuples, args.chunk_size))
    amr_enabled = not args.disable_amr

    with mp.Pool(processes=max(1, args.workers)) as pool:
        results = pool.starmap(_worker_process, [(c, amr_enabled) for c in chunks])

    rows = [item for sub in results for item in sub]
    amr_sentence_df = pd.DataFrame(rows)
    amr_sentence_df.to_csv(
        output_dir / "amr_sentence_metrics.csv.gz", index=False, compression="gzip"
    )

    for speech_id, sub in amr_sentence_df.groupby("speech_id"):
        penman_lines = [p for p in sub["amr_penman"].tolist() if p]
        if penman_lines:
            (amr_dir / f"{speech_id}.penman").write_text(
                "\n\n".join(penman_lines), encoding="utf-8"
            )

    agg = (
        amr_sentence_df.groupby("speech_id", as_index=False)
        .agg(
            avg_amr_nodes=("amr_nodes", "mean"),
            avg_amr_edges=("amr_edges", "mean"),
            avg_amr_depth=("amr_depth", "mean"),
            sentence_count=("sentence_id", "count"),
        )
        .reset_index(drop=True)
    )
    agg.to_csv(output_dir / "amr_speech_metrics.csv", index=False)

    print(f"Saved sentence AMR metrics: {output_dir / 'amr_sentence_metrics.csv.gz'}")
    print(f"Saved speech AMR metrics: {output_dir / 'amr_speech_metrics.csv'}")
    print(f"Saved penman directory: {amr_dir}")


if __name__ == "__main__":
    main()
