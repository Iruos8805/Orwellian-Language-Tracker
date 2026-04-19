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
    parser.add_argument("--max-sentences-per-speech", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=1000)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    amr_dir = ensure_dir(output_dir / "amr_penman")
    sentence_out = output_dir / "amr_sentence_metrics.csv"

    if sentence_out.exists():
        sentence_out.unlink()

    sent_df = pd.read_csv(
        args.sentences_csv, dtype={"speech_id": str, "sentence_id": str}
    )
    sent_df = sent_df.sort_values(["speech_id", "sentence_idx"], kind="stable")
    sent_df = sent_df.groupby("speech_id", as_index=False, group_keys=False).head(
        max(args.max_sentences_per_speech, 1)
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

    cuda_available = False
    if amr_enabled:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

    print(f"AMR enabled: {amr_enabled}")
    print(f"CUDA available: {cuda_available}")

    if amr_enabled and args.require_gpu:
        if not cuda_available:
            raise RuntimeError(
                "AMR GPU check failed: torch.cuda.is_available() is False. "
                "Use --disable-amr for CPU heuristic mode or rerun on a CUDA-ready environment."
            )

    speech_agg: dict[str, dict[str, float]] = {}
    penman_by_speech: dict[str, list[str]] = {}
    pending_rows: list[dict] = []

    with mp.Pool(processes=max(1, args.workers)) as pool:
        for result_chunk in pool.starmap(_worker_process, [(c, amr_enabled) for c in chunks]):
            pending_rows.extend(result_chunk)

            for row in result_chunk:
                sid = row["speech_id"]
                acc = speech_agg.setdefault(
                    sid,
                    {
                        "speech_id": sid,
                        "sum_nodes": 0.0,
                        "sum_edges": 0.0,
                        "sum_depth": 0.0,
                        "sentence_count": 0.0,
                    },
                )
                acc["sum_nodes"] += float(row["amr_nodes"])
                acc["sum_edges"] += float(row["amr_edges"])
                acc["sum_depth"] += float(row["amr_depth"])
                acc["sentence_count"] += 1.0

                if row["amr_penman"]:
                    penman_by_speech.setdefault(sid, []).append(row["amr_penman"])

            if len(pending_rows) >= max(args.flush_every, 1):
                pd.DataFrame(pending_rows).to_csv(
                    sentence_out,
                    index=False,
                    mode="a",
                    header=not sentence_out.exists(),
                )
                pending_rows.clear()

    if pending_rows:
        pd.DataFrame(pending_rows).to_csv(
            sentence_out,
            index=False,
            mode="a",
            header=not sentence_out.exists(),
        )

    for speech_id, penmans in penman_by_speech.items():
        (amr_dir / f"{speech_id}.penman").write_text("\n\n".join(penmans), encoding="utf-8")

    agg_rows = []
    for sid, acc in speech_agg.items():
        n = max(acc["sentence_count"], 1.0)
        agg_rows.append(
            {
                "speech_id": sid,
                "avg_amr_nodes": acc["sum_nodes"] / n,
                "avg_amr_edges": acc["sum_edges"] / n,
                "avg_amr_depth": acc["sum_depth"] / n,
                "sentence_count": int(acc["sentence_count"]),
            }
        )
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(output_dir / "amr_speech_metrics.csv", index=False)

    print(f"Saved sentence AMR metrics: {sentence_out}")
    print(f"Saved speech AMR metrics: {output_dir / 'amr_speech_metrics.csv'}")
    print(f"Saved penman directory: {amr_dir}")


if __name__ == "__main__":
    main()
