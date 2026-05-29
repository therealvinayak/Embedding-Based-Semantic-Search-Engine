"""
TF-IDF Baseline Search Module
==============================
This module implements a lexical search engine using TF-IDF + Cosine Similarity.
It serves as the BASELINE against which we'll measure our semantic (embedding) approach.

Architecture:
    Query → TF-IDF Vectorize → Cosine Similarity vs Corpus Vectors → Rank → Top-K

Key Engineering Decisions:
- We use Scikit-learn's TfidfVectorizer which handles TF, IDF, and L2 normalization
  in one step. The L2 normalization means dot product = cosine similarity (optimization).
- We cap vocabulary at MAX_FEATURES to control memory usage. In production with millions
  of documents, an uncapped vocabulary could consume GBs of RAM.
- Bigrams (ngram_range=(1,2)) capture short phrases like "import error" or "memory leak"
  that unigrams would miss.
"""

import os
import time
import logging
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class TFIDFSearchEngine:
    """
    A production-style TF-IDF search engine with fit/query/save/load semantics.

    Why a class and not just functions?
    - The vectorizer and matrix are stateful (fitted on the corpus).
    - Encapsulation makes it easy to swap this for the FAISS engine later
      with an identical interface (Strategy Pattern).
    - Serialization (save/load) is cleaner when state is bundled.
    """

    def __init__(
        self,
        max_features: int = config.TFIDF_MAX_FEATURES,
        ngram_range: tuple = config.TFIDF_NGRAM_RANGE,
    ):
        """
        Initialize the TF-IDF engine.

        Parameters
        ----------
        max_features : int
            Maximum vocabulary size. Acts as a dimensionality cap.
            Higher = more expressive but more memory. 10K is a good default for ~3K docs.

        ngram_range : tuple
            (min_n, max_n) for n-gram extraction.
            (1,1) = only unigrams ("error", "import")
            (1,2) = unigrams + bigrams ("error", "import", "import error")
            (1,3) = up to trigrams — rarely worth the memory cost for small corpora.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",  # Remove "the", "is", "at", etc.
            sublinear_tf=True,     # Apply log(1 + tf) instead of raw tf — dampens
                                   # the effect of very frequent terms in long docs.
            dtype=np.float32,      # float32 saves 50% memory vs float64, negligible
                                   # precision loss for similarity ranking.
        )
        self.tfidf_matrix = None  # Sparse matrix: (n_docs, max_features)
        self.corpus_df = None     # Reference to document metadata for result display

    def fit(self, df: pd.DataFrame, text_column: str = "text") -> None:
        """
        Fit the TF-IDF vectorizer on the corpus and transform documents to vectors.

        This is the "indexing" step — analogous to building a FAISS index in Phase 3.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned corpus with a text column.
        text_column : str
            Name of the column containing searchable text.
        """
        logger.info(f"Fitting TF-IDF on {len(df)} documents...")
        start_time = time.time()

        self.corpus_df = df.copy()
        self.tfidf_matrix = self.vectorizer.fit_transform(df[text_column])

        elapsed = time.time() - start_time
        logger.info(
            f"TF-IDF matrix shape: {self.tfidf_matrix.shape} "
            f"(docs × vocab) | Built in {elapsed:.2f}s"
        )
        logger.info(
            f"Matrix density: {self.tfidf_matrix.nnz / np.prod(self.tfidf_matrix.shape):.4%} "
            f"(sparse — most entries are zero)"
        )

    def search(self, query: str, top_k: int = config.TOP_K) -> pd.DataFrame:
        """
        Search the corpus for documents most similar to the query.

        How it works:
        1. Transform the query into the same TF-IDF vector space as the corpus.
           (Uses the FITTED vocabulary — unknown query words are ignored!)
        2. Compute cosine similarity between query vector and ALL document vectors.
        3. Sort by similarity score (descending) and return top-K.

        Parameters
        ----------
        query : str
            User's search query (natural language).
        top_k : int
            Number of results to return.

        Returns
        -------
        pd.DataFrame
            Top-K results with columns: [rank, score, title, text]
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Engine not fitted. Call .fit() first.")

        # Transform query into the same vector space
        query_vector = self.vectorizer.transform([query])

        # Compute cosine similarity against all documents
        # Since TfidfVectorizer uses L2 normalization by default,
        # cosine_similarity is equivalent to a dot product here.
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top-K indices (argsort is ascending, so we negate or use [::-1])
        # np.argpartition is O(n) vs O(n log n) for full sort — but for 3K docs
        # the difference is negligible. We use argsort for clarity.
        top_indices = similarities.argsort()[::-1][:top_k]
        top_scores = similarities[top_indices]

        # Build results DataFrame
        results = self.corpus_df.iloc[top_indices][["title", "text"]].copy()
        results.insert(0, "rank", range(1, top_k + 1))
        results.insert(1, "score", top_scores)
        results = results.reset_index(drop=True)

        return results

    def batch_search(
        self, queries: List[str], top_k: int = config.TOP_K
    ) -> List[pd.DataFrame]:
        """
        Search multiple queries efficiently.

        Why batch search?
        - Evaluation requires running many queries (Phase 4).
        - Matrix multiplication is faster than looping (vectorized computation).
        - This mirrors production systems that process queries in batches.
        """
        results = []
        for query in queries:
            results.append(self.search(query, top_k=top_k))
        return results

    def save(self, filepath: str = None) -> str:
        """
        Serialize the fitted engine to disk.

        Why pickle the whole object?
        - The vectorizer's vocabulary and IDF weights must be preserved together.
        - Scikit-learn objects are pickle-compatible by design.
        - In production, you'd version these artifacts (MLflow, DVC, etc.).
        """
        if filepath is None:
            os.makedirs(config.INDEX_DIR, exist_ok=True)
            filepath = os.path.join(config.INDEX_DIR, "tfidf_engine.pkl")

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                    "corpus_df": self.corpus_df,
                },
                f,
            )
        logger.info(f"TF-IDF engine saved to: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath: str = None) -> "TFIDFSearchEngine":
        """
        Load a previously fitted engine from disk.

        Why a classmethod?
        - Factory pattern: returns a fully initialized instance without calling fit().
        - Clean API: TFIDFSearchEngine.load("path") reads naturally.
        """
        if filepath is None:
            filepath = os.path.join(config.INDEX_DIR, "tfidf_engine.pkl")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved engine at {filepath}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        engine = cls()
        engine.vectorizer = data["vectorizer"]
        engine.tfidf_matrix = data["tfidf_matrix"]
        engine.corpus_df = data["corpus_df"]
        logger.info(f"TF-IDF engine loaded from: {filepath}")
        return engine


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Demo: Load processed data, build TF-IDF index, and run sample queries.

    Usage:
        python src/tfidf_search.py
    """
    from data_loader import run_pipeline

    # Load or prepare data
    df = run_pipeline()

    # Build the TF-IDF search engine
    engine = TFIDFSearchEngine()
    engine.fit(df)

    # Run sample queries to verify it works
    sample_queries = [
        "dataset download error",
        "how to fine-tune a model",
        "memory leak during training",
        "tokenizer not working",
    ]

    print("\n" + "=" * 70)
    print("TF-IDF BASELINE SEARCH DEMO")
    print("=" * 70)

    for query in sample_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        results = engine.search(query, top_k=5)
        for _, row in results.iterrows():
            # Truncate text for display
            snippet = row["text"][:100].replace("\n", " ")
            print(f"  [{row['rank']}] Score: {row['score']:.4f} | {snippet}...")
        print()

    # Save the engine for later use
    engine.save()
    print("✅ TF-IDF engine saved to disk.")
