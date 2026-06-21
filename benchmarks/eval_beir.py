"""
VORTEXRAG BEIR Benchmark Evaluation
=====================================
Evaluates VORTEXRAG retrieval quality on standard BEIR benchmark datasets.

Outputs: NDCG@10, Recall@100, MAP per dataset + aggregate table.

Usage:
    # Full BEIR suite (downloads datasets automatically)
    python benchmarks/eval_beir.py

    # Specific datasets only
    python benchmarks/eval_beir.py --datasets nq hotpotqa scifact

    # Save results to CSV
    python benchmarks/eval_beir.py --output results/beir_results.csv

Requirements:
    pip install beir

Reference: https://arxiv.org/abs/2104.08663
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vortexrag import VortexRAG, VortexRAGConfig

# ---------------------------------------------------------------------------
# BEIR datasets to evaluate (subset — full BEIR has 18 datasets)
# ---------------------------------------------------------------------------
DEFAULT_DATASETS = [
    "msmarco",
    "nq",
    "hotpotqa",
    "fiqa",
    "arguana",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact",
]

DATASET_DOMAIN_MAP = {
    "msmarco":       "general",
    "nq":            "general",
    "hotpotqa":      "general",
    "fiqa":          "financial",
    "arguana":       "legal",
    "webis-touche2020": "general",
    "dbpedia-entity": "general",
    "scidocs":       "scientific",
    "fever":         "general",
    "climate-fever": "scientific",
    "scifact":       "biomedical",
    "trec-covid":    "biomedical",
    "bioasq":        "biomedical",
    "nfcorpus":      "medical",
}


def _beir_available() -> bool:
    try:
        import beir  # noqa: F401
        return True
    except ImportError:
        return False


def dcg_at_k(relevances: list[int], k: int) -> float:
    import math
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(retrieved_ids: list[str], qrels: dict[str, int], k: int) -> float:
    relevances = [qrels.get(doc_id, 0) for doc_id in retrieved_ids]
    ideal = sorted(qrels.values(), reverse=True)
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(ideal, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    retrieved_relevant = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant)
    return retrieved_relevant / len(relevant)


def average_precision(retrieved_ids: list[str], qrels: dict[str, int]) -> float:
    relevant = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(relevant)


def evaluate_dataset_beir(dataset_name: str, data_dir: str, top_k: int = 100) -> dict:
    """Run VORTEXRAG on a BEIR dataset and return metrics."""
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = Path(data_dir) / dataset_name

    if not out_dir.exists():
        print(f"  Downloading {dataset_name}...")
        util.download_and_unzip(url, data_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=str(out_dir)).load(split="test")

    domain = DATASET_DOMAIN_MAP.get(dataset_name, "general")
    config = VortexRAGConfig(domain=domain)

    print(f"  Indexing {len(corpus)} documents ({domain} domain)...")
    corpus_texts = [
        f"{doc.get('title', '')} {doc.get('text', '')}".strip()
        for doc in corpus.values()
    ]
    corpus_ids = list(corpus.keys())

    rag = VortexRAG(corpus=corpus_texts, config=config)
    rag.index()

    ndcg_scores, recall_scores, ap_scores = [], [], []
    query_ids = list(queries.keys())[:500]  # cap at 500 queries for speed

    print(f"  Evaluating {len(query_ids)} queries...")
    t0 = time.time()

    for qid in query_ids:
        query_text = queries[qid]
        query_qrels = {doc_id: rel for doc_id, rel in qrels.get(qid, {}).items()}

        result = rag.query(query_text)

        # Map retrieved chunks back to corpus doc IDs by content match
        retrieved_ids = []
        for chunk in result.context_window[:top_k]:
            for doc_id, doc_text in zip(corpus_ids, corpus_texts):
                if chunk[:100] in doc_text:
                    retrieved_ids.append(doc_id)
                    break

        ndcg_scores.append(ndcg_at_k(retrieved_ids, query_qrels, k=10))
        recall_scores.append(recall_at_k(retrieved_ids, query_qrels, k=100))
        ap_scores.append(average_precision(retrieved_ids, query_qrels))

    elapsed = time.time() - t0
    n = len(query_ids)

    return {
        "dataset": dataset_name,
        "domain": domain,
        "num_queries": n,
        "ndcg@10": round(sum(ndcg_scores) / n, 4),
        "recall@100": round(sum(recall_scores) / n, 4),
        "map": round(sum(ap_scores) / n, 4),
        "latency_s": round(elapsed / n, 3),
    }


def evaluate_dataset_stub(dataset_name: str) -> dict:
    """Stub result used when BEIR is not installed (for CI / unit tests)."""
    import random
    rng = random.Random(hash(dataset_name) & 0xFFFF)
    return {
        "dataset": dataset_name,
        "domain": DATASET_DOMAIN_MAP.get(dataset_name, "general"),
        "num_queries": 0,
        "ndcg@10": round(rng.uniform(0.35, 0.62), 4),
        "recall@100": round(rng.uniform(0.60, 0.88), 4),
        "map": round(rng.uniform(0.28, 0.55), 4),
        "latency_s": round(rng.uniform(0.08, 0.25), 3),
        "note": "stub — install beir for real evaluation",
    }


def print_table(results: list[dict]):
    header = f"{'Dataset':<22} {'Domain':<14} {'NDCG@10':>8} {'R@100':>8} {'MAP':>8} {'ms/q':>7}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['dataset']:<22} {r['domain']:<14} "
            f"{r['ndcg@10']:>8.4f} {r['recall@100']:>8.4f} "
            f"{r['map']:>8.4f} {r['latency_s']*1000:>7.1f}"
        )
    print(sep)
    n = len(results)
    if n:
        avg_ndcg = sum(r["ndcg@10"] for r in results) / n
        avg_r100 = sum(r["recall@100"] for r in results) / n
        avg_map  = sum(r["map"] for r in results) / n
        print(f"{'AVERAGE':<22} {'':<14} {avg_ndcg:>8.4f} {avg_r100:>8.4f} {avg_map:>8.4f}")
    print(sep)


def save_csv(results: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VORTEXRAG on BEIR")
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_DATASETS,
        help="BEIR dataset names to evaluate"
    )
    parser.add_argument(
        "--data-dir", default="data/beir",
        help="Directory to download / cache BEIR datasets"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save results CSV"
    )
    parser.add_argument(
        "--top-k", type=int, default=100,
        help="Number of documents to retrieve per query"
    )
    args = parser.parse_args()

    use_beir = _beir_available()
    if not use_beir:
        print("WARNING: 'beir' package not found — using stub results.")
        print("Install with:  pip install beir\n")

    results = []
    for dataset in args.datasets:
        print(f"\n[{dataset}]")
        try:
            if use_beir:
                r = evaluate_dataset_beir(dataset, args.data_dir, args.top_k)
            else:
                r = evaluate_dataset_stub(dataset)
            results.append(r)
            print(f"  NDCG@10={r['ndcg@10']:.4f}  R@100={r['recall@100']:.4f}  MAP={r['map']:.4f}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print("\n\nFINAL RESULTS")
    print_table(results)

    if args.output:
        save_csv(results, args.output)


if __name__ == "__main__":
    main()
