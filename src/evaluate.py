"""
Evaluation Module
=================
Responsible for:
1. Creating a synthetic evaluation set (title → document matching)
2. Computing retrieval metrics: Precision@K, MRR, NDCG@K
3. Running both TF-IDF and FAISS engines on the same queries
4. Producing a comparative report proving which approach is superior

Evaluation Strategy:
- We sample N documents from the corpus
- Use each document's TITLE as the query
- The source document is the KNOWN relevant result (ground truth)
- We check if each engine retrieves that document in its top-K results

Known Limitations (document in writeup):
- Title-as-query introduces lexical bias favoring TF-IDF (titles share words with body)
- Binary relevance only (relevant/not relevant) — no graded relevance
- Single relevant document per query (real search often has multiple relevant docs)
- Despite these limitations, this is a standard evaluation approach for demo/research projects

Engineering Decisions:
- We use a fixed random seed for reproducibility (same eval set every run)
- Sample size of 200 queries balances statistical significance with runtime
- Results are printed in a formatted comparison table
"""

import os
import sys
import time
import logging
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def precision_at_k(retrieved_indices: List[int], relevant_index: int, k: int) -> float:
    """
    Compute Precision@K: fraction of top-K results that are relevant.

    Parameters
    ----------
    retrieved_indices : List[int]
        Ordered list of document indices returned by the search engine.
    relevant_index : int
        The index of the single known-relevant document.
    k : int
        Cutoff position.

    Returns
    -------
    float
        Precision@K value (0.0 or 1/K for single-relevant-doc case).

    Math:
        Precision@K = |{relevant docs in top-K}| / K

    Since we have exactly ONE relevant doc per query:
        - If it's in top-K → Precision@K = 1/K
        - If it's NOT in top-K → Precision@K = 0.0

    Note: With single relevance, this simplifies to a hit/miss indicator
    divided by K. Hit@K (binary) would be more natural, but we implement
    the standard formula for correctness.
    """
    top_k = retrieved_indices[:k]
    if relevant_index in top_k:
        return 1.0 / k  # Standard Precision@K with 1 relevant doc
    return 0.0


def hit_at_k(retrieved_indices: List[int], relevant_index: int, k: int) -> float:
    """
    Compute Hit@K: whether the relevant document appears in top-K at all.

    More intuitive than Precision@K for single-relevant-doc evaluation.
    Returns 1.0 if found, 0.0 if not.
    """
    return 1.0 if relevant_index in retrieved_indices[:k] else 0.0


def reciprocal_rank(retrieved_indices: List[int], relevant_index: int) -> float:
    """
    Compute the Reciprocal Rank (RR) for a single query.

    RR = 1 / rank_of_first_relevant_document

    Examples:
        - Relevant doc at position 1 → RR = 1/1 = 1.0
        - Relevant doc at position 3 → RR = 1/3 = 0.33
        - Relevant doc not found     → RR = 0.0

    MRR (Mean Reciprocal Rank) is the average of RR across all queries.
    """
    try:
        # .index() returns 0-based position, rank is 1-based
        rank = retrieved_indices.index(relevant_index) + 1
        return 1.0 / rank
    except ValueError:
        # Relevant document not in retrieved list
        return 0.0


def ndcg_at_k(retrieved_indices: List[int], relevant_index: int, k: int) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).

    With binary relevance and a single relevant document:
        - DCG@K = 1/log2(rank+1) if relevant doc is in top-K, else 0
        - IDCG@K = 1/log2(2) = 1.0 (perfect ranking puts it at position 1)
        - NDCG@K = DCG@K / IDCG@K = 1/log2(rank+1)

    Parameters
    ----------
    retrieved_indices : List[int]
        Ordered list of retrieved document indices.
    relevant_index : int
        Index of the known-relevant document.
    k : int
        Cutoff position.

    Returns
    -------
    float
        NDCG@K score between 0.0 and 1.0.

    Math walkthrough:
        If relevant doc is at rank r (1-indexed) within top-K:
            DCG@K = 1 / log2(r + 1)
            IDCG@K = 1 / log2(1 + 1) = 1 / log2(2) = 1.0
            NDCG@K = DCG@K / IDCG@K = 1 / log2(r + 1)

        Examples:
            rank 1 → NDCG = 1/log2(2) = 1.0
            rank 2 → NDCG = 1/log2(3) = 0.63
            rank 5 → NDCG = 1/log2(6) = 0.39
            rank 10 → NDCG = 1/log2(11) = 0.29
            not found → NDCG = 0.0
    """
    top_k = retrieved_indices[:k]
    if relevant_index not in top_k:
        return 0.0

    # Find 1-based rank within top-K
    rank = top_k.index(relevant_index) + 1

    # DCG for binary relevance with single relevant doc
    dcg = 1.0 / np.log2(rank + 1)

    # IDCG: best possible = relevant doc at rank 1
    idcg = 1.0 / np.log2(2)  # = 1.0

    return dcg / idcg


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════

def create_eval_set(
    df: pd.DataFrame,
    n_queries: int = 200,
    min_title_length: int = 15,
    seed: int = 42,
) -> List[Dict]:
    """
    Create a synthetic evaluation set from the corpus.

    Strategy:
    - Sample N documents with titles long enough to be meaningful queries
    - Use title as query, document index as ground truth

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned corpus.
    n_queries : int
        Number of evaluation queries. 200 gives statistical significance
        while keeping runtime reasonable.
    min_title_length : int
        Minimum title character length. Filters out titles like "Bug fix"
        that are too generic to test retrieval quality.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    List[Dict]
        Each dict has: {"query": str, "relevant_idx": int, "title": str}
    """
    random.seed(seed)
    np.random.seed(seed)

    # Filter for documents with informative titles
    candidates = df[df["title"].str.len() >= min_title_length].copy()
    logger.info(f"Eval candidates (title >= {min_title_length} chars): {len(candidates)}")

    # Sample
    n_queries = min(n_queries, len(candidates))
    sample_indices = random.sample(range(len(candidates)), n_queries)

    eval_set = []
    for idx in sample_indices:
        row = candidates.iloc[idx]
        # We need the index in the ORIGINAL DataFrame (not the filtered one)
        original_idx = candidates.index[idx]
        eval_set.append({
            "query": row["title"],
            "relevant_idx": original_idx,
            "title": row["title"],
        })

    logger.info(f"Created eval set with {len(eval_set)} queries")
    return eval_set


def evaluate_engine(
    engine,
    eval_set: List[Dict],
    engine_name: str,
    k_values: List[int] = config.EVAL_K_VALUES,
    top_k: int = None,
) -> Dict[str, float]:
    """
    Evaluate a search engine on the evaluation set.

    Parameters
    ----------
    engine : TFIDFSearchEngine or FAISSSearchEngine
        Must have a .search(query, top_k) method returning DataFrame with results.
    eval_set : List[Dict]
        Evaluation queries with ground truth.
    engine_name : str
        Name for logging/display.
    k_values : List[int]
        K values for Precision@K and NDCG@K.
    top_k : int
        Max results to retrieve per query.

    Returns
    -------
    Dict[str, float]
        All computed metrics.
    """
    if top_k is None:
        top_k = max(k_values)

    logger.info(f"Evaluating {engine_name} on {len(eval_set)} queries (top_k={top_k})...")
    start_time = time.time()

    # Storage for per-query metrics
    reciprocal_ranks = []
    hits = {k: [] for k in k_values}
    precisions = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}

    for i, eval_item in enumerate(eval_set):
        query = eval_item["query"]
        relevant_idx = eval_item["relevant_idx"]

        # Run search
        results = engine.search(query, top_k=top_k)

        # Get retrieved document indices from the corpus
        # We need to map back from the results DataFrame to corpus indices
        # The results contain rows from corpus_df, so we can use the original index
        retrieved_indices = list(engine.corpus_df.index[
            engine.corpus_df["text"].isin(results["text"])
        ])

        # If that approach is too slow or inaccurate, use position-based matching:
        # For FAISS and TF-IDF engines, the results are from corpus_df.iloc[indices]
        # Let's use a more reliable approach: match by title + text
        retrieved_indices = []
        for _, row in results.iterrows():
            # Find the index in corpus_df that matches this result
            mask = engine.corpus_df["text"] == row["text"]
            matching = engine.corpus_df.index[mask].tolist()
            if matching:
                retrieved_indices.append(matching[0])

        # Compute metrics
        reciprocal_ranks.append(reciprocal_rank(retrieved_indices, relevant_idx))

        for k in k_values:
            hits[k].append(hit_at_k(retrieved_indices, relevant_idx, k))
            precisions[k].append(precision_at_k(retrieved_indices, relevant_idx, k))
            ndcgs[k].append(ndcg_at_k(retrieved_indices, relevant_idx, k))

    elapsed = time.time() - start_time

    # Aggregate metrics (mean across all queries)
    metrics = {
        "MRR": np.mean(reciprocal_ranks),
        "Query Time (s)": elapsed / len(eval_set),
    }

    for k in k_values:
        metrics[f"Hit@{k}"] = np.mean(hits[k])
        metrics[f"Precision@{k}"] = np.mean(precisions[k])
        metrics[f"NDCG@{k}"] = np.mean(ndcgs[k])

    logger.info(f"{engine_name} evaluation complete in {elapsed:.1f}s")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison_report(
    tfidf_metrics: Dict[str, float],
    faiss_metrics: Dict[str, float],
) -> None:
    """
    Print a formatted comparison table showing both engines' performance.
    """
    print("\n" + "=" * 75)
    print("EVALUATION REPORT: TF-IDF Baseline vs FAISS Semantic Search")
    print("=" * 75)
    print(f"{'Metric':<20} {'TF-IDF':>12} {'FAISS (Semantic)':>18} {'Δ Improvement':>15}")
    print("-" * 75)

    for metric in sorted(tfidf_metrics.keys()):
        tfidf_val = tfidf_metrics[metric]
        faiss_val = faiss_metrics[metric]

        if "Time" in metric:
            # For time, lower is better
            delta = f"{((tfidf_val - faiss_val) / tfidf_val * 100):+.1f}%"
            print(f"{metric:<20} {tfidf_val:>12.4f} {faiss_val:>18.4f} {delta:>15}")
        else:
            # For all other metrics, higher is better
            if tfidf_val > 0:
                pct_improvement = ((faiss_val - tfidf_val) / tfidf_val) * 100
                delta = f"{pct_improvement:+.1f}%"
            else:
                delta = "N/A"
            print(f"{metric:<20} {tfidf_val:>12.4f} {faiss_val:>18.4f} {delta:>15}")

    print("=" * 75)
    print("\nKey Takeaways:")
    print(f"  • MRR: {faiss_metrics['MRR']:.3f} vs {tfidf_metrics['MRR']:.3f} "
          f"(Semantic finds relevant docs {faiss_metrics['MRR']/max(tfidf_metrics['MRR'], 0.001):.1f}x faster)")
    print(f"  • Hit@1: {faiss_metrics['Hit@1']*100:.1f}% vs {tfidf_metrics['Hit@1']*100:.1f}% "
          f"(First result is correct)")
    print(f"  • NDCG@10: {faiss_metrics['NDCG@10']:.3f} vs {tfidf_metrics['NDCG@10']:.3f} "
          f"(Overall ranking quality)")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Run the full evaluation comparing TF-IDF vs FAISS:
        python src/evaluate.py

    Prerequisites:
        - data/processed_issues.parquet (run data_loader.py)
        - data/corpus_embeddings.npy (run embed.py)
        - indexes/tfidf_engine.pkl (run tfidf_search.py)
        - indexes/faiss_index.bin (run faiss_index.py)
    """
    from data_loader import load_processed_data
    from tfidf_search import TFIDFSearchEngine
    from faiss_index import FAISSSearchEngine

    # ─── Load Data ────────────────────────────────────────────────────────
    df = load_processed_data()

    # ─── Load Engines ─────────────────────────────────────────────────────
    logger.info("Loading TF-IDF engine...")
    tfidf_engine = TFIDFSearchEngine.load()

    logger.info("Loading FAISS engine...")
    faiss_engine = FAISSSearchEngine()
    faiss_engine.load_index(df)

    # ─── Create Evaluation Set ────────────────────────────────────────────
    eval_set = create_eval_set(df, n_queries=200, seed=42)

    # Show sample queries
    print("\nSample evaluation queries:")
    for item in eval_set[:5]:
        print(f"  Query: '{item['query'][:60]}...' → Target idx: {item['relevant_idx']}")
    print()

    # ─── Evaluate Both Engines ────────────────────────────────────────────
    tfidf_metrics = evaluate_engine(
        tfidf_engine, eval_set, "TF-IDF Baseline"
    )

    faiss_metrics = evaluate_engine(
        faiss_engine, eval_set, "FAISS Semantic"
    )

    # ─── Print Report ─────────────────────────────────────────────────────
    print_comparison_report(tfidf_metrics, faiss_metrics)

    # ─── Additional Analysis: Per-Query Breakdown ─────────────────────────
    print("\n" + "=" * 75)
    print("FAILURE ANALYSIS: Queries where TF-IDF beat Semantic Search")
    print("=" * 75)
    print("(These highlight the lexical bias in title-as-query evaluation)")
    print()

    # Find queries where TF-IDF found the answer but FAISS didn't
    tfidf_wins = 0
    faiss_wins = 0
    both_miss = 0

    for eval_item in eval_set[:50]:  # Analyze first 50
        query = eval_item["query"]
        relevant_idx = eval_item["relevant_idx"]

        tfidf_results = tfidf_engine.search(query, top_k=10)
        faiss_results = faiss_engine.search(query, top_k=10)

        # Check if relevant doc is in results
        tfidf_found = any(
            df.iloc[relevant_idx]["text"] == row["text"]
            for _, row in tfidf_results.iterrows()
        )
        faiss_found = any(
            df.iloc[relevant_idx]["text"] == row["text"]
            for _, row in faiss_results.iterrows()
        )

        if tfidf_found and not faiss_found:
            tfidf_wins += 1
            if tfidf_wins <= 3:
                print(f"  TF-IDF wins: '{query[:70]}...'")
        elif faiss_found and not tfidf_found:
            faiss_wins += 1
        elif not tfidf_found and not faiss_found:
            both_miss += 1

    print(f"\n  Summary (first 50 queries):")
    print(f"    TF-IDF finds, FAISS misses: {tfidf_wins}")
    print(f"    FAISS finds, TF-IDF misses: {faiss_wins}")
    print(f"    Both miss: {both_miss}")
    print("=" * 75)
