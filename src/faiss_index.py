"""
FAISS Vector Index Module
=========================
Responsible for:
1. Building a FAISS index from pre-computed embeddings
2. Querying the index for top-K nearest neighbors
3. Saving/loading the index to/from disk
4. Providing a clean search interface consistent with TFIDFSearchEngine

Engineering Decisions:
- We use IndexFlatIP (Inner Product) because our embeddings are L2-normalized.
  On normalized vectors: inner_product(a, b) = cosine_similarity(a, b).
  This gives us intuitive scores (higher = more similar) without extra computation.

- We wrap FAISS in a class with the same .search() interface as TFIDFSearchEngine.
  This is the Strategy Pattern — the app layer doesn't care which engine is behind it.

- Index is saved in FAISS's native binary format (.index), which is:
  * Portable across machines (same endianness)
  * Fast to load (memory-mapped internally)
  * Supports all index types (Flat, IVF, HNSW, PQ, etc.)

- For our corpus of ~2,734 docs, IndexFlatIP is the correct choice:
  * Exact search (100% recall guaranteed)
  * Sub-millisecond query time at this scale
  * No training step required
  * Zero approximation error
"""

import os
import time
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import faiss

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


class FAISSSearchEngine:
    """
    Semantic search engine powered by FAISS vector similarity search.

    Architecture:
        Query → Encode (sentence-transformer) → FAISS Index Search → Top-K Results

    This class handles the INDEX side. It expects pre-computed embeddings
    and uses FAISS for fast retrieval. The encoding step is delegated to
    the EmbeddingEngine (separation of concerns).
    """

    def __init__(self, embedding_dim: int = config.EMBEDDING_DIM):
        """
        Initialize the FAISS search engine.

        Parameters
        ----------
        embedding_dim : int
            Dimensionality of the embedding vectors. Must match the model output.
            all-MiniLM-L6-v2 → 384 dimensions.
        """
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.Index] = None
        self.corpus_df: Optional[pd.DataFrame] = None
        self.model = None  # Lazy-loaded sentence-transformer for query encoding

    def build_index(
        self,
        embeddings: np.ndarray,
        df: pd.DataFrame,
        index_type: str = "flat_ip",
    ) -> None:
        """
        Build the FAISS index from pre-computed embeddings.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape (n_docs, embedding_dim). Must be float32 and L2-normalized.
        df : pd.DataFrame
            Corpus metadata (title, text, etc.) for result display.
        index_type : str
            Type of FAISS index to build:
            - "flat_ip": Exact search via inner product (default, best for <100K docs)
            - "flat_l2": Exact search via L2 distance
            - "ivf": Approximate search with inverted file index (for >100K docs)
            - "hnsw": Approximate search with HNSW graph (best recall/speed tradeoff)

        Why we support multiple index types:
        - Demonstrates understanding of tradeoffs (great for resume/interviews)
        - Makes the engine configurable for different corpus sizes
        - In production, you'd A/B test index types for your specific workload
        """
        # Validate input
        assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"
        assert embeddings.shape[1] == self.embedding_dim, (
            f"Embedding dim mismatch: got {embeddings.shape[1]}, expected {self.embedding_dim}"
        )
        assert embeddings.dtype == np.float32, (
            f"FAISS requires float32, got {embeddings.dtype}"
        )

        n_docs = embeddings.shape[0]
        self.corpus_df = df.copy()

        logger.info(f"Building FAISS index: type={index_type}, n_docs={n_docs}, dim={self.embedding_dim}")
        start = time.time()

        if index_type == "flat_ip":
            # ─── Exact Inner Product Search ──────────────────────────────
            # Best for: small-medium corpora (<100K), when 100% recall is required.
            # How it works: stores all vectors, computes IP against every one at query time.
            # Complexity: O(n × d) per query.
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        elif index_type == "flat_l2":
            # ─── Exact L2 (Euclidean) Search ─────────────────────────────
            # Same as flat_ip but uses L2 distance. Lower = more similar.
            # Use when vectors are NOT normalized.
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        elif index_type == "ivf":
            # ─── Inverted File Index (Approximate) ───────────────────────
            # Best for: 100K-10M documents when some recall loss is acceptable.
            # How it works:
            #   1. Clusters vectors into nlist groups via K-means (training step)
            #   2. At query time, only searches nprobe closest clusters
            # Tradeoff: nprobe ↑ = better recall, slower
            nlist = min(100, n_docs // 10)  # Rule of thumb: sqrt(n) to n/10 clusters
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            # IVF requires training on representative data
            logger.info(f"Training IVF index with nlist={nlist}...")
            self.index.train(embeddings)
            self.index.nprobe = min(10, nlist)  # Search 10 clusters by default
            logger.info(f"IVF trained. nprobe={self.index.nprobe}")

        elif index_type == "hnsw":
            # ─── Hierarchical Navigable Small World (Approximate) ────────
            # Best for: high recall + low latency at any scale.
            # How it works: builds a multi-layer graph connecting similar vectors.
            # Search = greedy graph traversal (like navigating a social network).
            # M = number of connections per node (higher = better recall, more memory)
            M = 32  # Default HNSW connectivity
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            self.index.hnsw.efConstruction = 200  # Build quality (higher = slower build, better graph)
            self.index.hnsw.efSearch = 50  # Search quality (higher = better recall, slower query)

        else:
            raise ValueError(f"Unknown index type: {index_type}. Choose from: flat_ip, flat_l2, ivf, hnsw")

        # Add vectors to the index
        self.index.add(embeddings)

        elapsed = time.time() - start
        logger.info(
            f"Index built in {elapsed:.3f}s | "
            f"Total vectors: {self.index.ntotal} | "
            f"Index size in memory: ~{self.index.ntotal * self.embedding_dim * 4 / 1024 / 1024:.2f} MB"
        )

    def _get_model(self):
        """
        Lazy-load the sentence-transformer model for query encoding.

        Why lazy loading?
        - Building/loading an index doesn't require the model.
        - The model is only needed at QUERY TIME.
        - Saves ~2s startup time when you're just loading an index for inspection.
        """
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading query encoder: {config.EMBEDDING_MODEL_NAME}")
            self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        return self.model

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query string into the embedding space.

        Returns shape (1, embedding_dim) as float32 — ready for FAISS.
        """
        model = self._get_model()
        embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    def search(self, query: str, top_k: int = config.TOP_K) -> pd.DataFrame:
        """
        Search the FAISS index for documents most similar to the query.

        Pipeline:
        1. Encode query → 384-d normalized vector
        2. FAISS searches index → returns (distances, indices) arrays
        3. Map indices back to document metadata → formatted results

        Parameters
        ----------
        query : str
            Natural language search query.
        top_k : int
            Number of results to return.

        Returns
        -------
        pd.DataFrame
            Top-K results with columns: [rank, score, title, text]

        Performance:
            Query encoding: ~5-10ms (transformer forward pass)
            FAISS search: <1ms for IndexFlatIP at 3K docs
            Total: ~10-15ms per query (well within interactive latency budget)
        """
        if self.index is None:
            raise RuntimeError("No index built. Call build_index() or load_index() first.")

        # Step 1: Encode query
        query_embedding = self.encode_query(query)  # (1, 384)

        # Step 2: FAISS search
        # Returns: distances (similarity scores), indices (positions in corpus)
        # Both have shape (n_queries, top_k) = (1, top_k)
        distances, indices = self.index.search(query_embedding, top_k)

        # Flatten from (1, top_k) to (top_k,)
        scores = distances.flatten()
        doc_indices = indices.flatten()

        # Step 3: Build results DataFrame
        results = self.corpus_df.iloc[doc_indices][["title", "text"]].copy()
        results.insert(0, "rank", range(1, top_k + 1))
        results.insert(1, "score", scores)
        results = results.reset_index(drop=True)

        return results

    def batch_search(
        self, queries: List[str], top_k: int = config.TOP_K
    ) -> List[pd.DataFrame]:
        """
        Efficient batch search for evaluation.

        FAISS natively supports batch queries — we encode all queries at once
        and search in a single call for maximum throughput.
        """
        if self.index is None:
            raise RuntimeError("No index built. Call build_index() or load_index() first.")

        model = self._get_model()

        # Batch encode all queries
        query_embeddings = model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=32,
        ).astype(np.float32)

        # Batch FAISS search
        distances, indices = self.index.search(query_embeddings, top_k)

        # Build results for each query
        results = []
        for i in range(len(queries)):
            scores = distances[i]
            doc_indices = indices[i]
            result_df = self.corpus_df.iloc[doc_indices][["title", "text"]].copy()
            result_df.insert(0, "rank", range(1, top_k + 1))
            result_df.insert(1, "score", scores)
            result_df = result_df.reset_index(drop=True)
            results.append(result_df)

        return results

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save_index(self, filename: str = "faiss_index.bin") -> str:
        """
        Save the FAISS index to disk in native binary format.

        FAISS's format is:
        - Self-describing (includes index type metadata)
        - Portable across platforms (with same endianness)
        - Supports memory-mapping for huge indexes (mmap)
        - Much faster to load than re-building from embeddings
        """
        if self.index is None:
            raise RuntimeError("No index to save. Build or load an index first.")

        os.makedirs(config.INDEX_DIR, exist_ok=True)
        filepath = os.path.join(config.INDEX_DIR, filename)
        faiss.write_index(self.index, filepath)
        logger.info(f"FAISS index saved to: {filepath} ({self.index.ntotal} vectors)")
        return filepath

    def load_index(
        self,
        df: pd.DataFrame,
        filename: str = "faiss_index.bin",
    ) -> None:
        """
        Load a previously built FAISS index from disk.

        Parameters
        ----------
        df : pd.DataFrame
            Corpus metadata — must be in the same order as when the index was built!
            FAISS stores only vectors (not metadata), so we need the DataFrame
            to map indices back to document content.
        filename : str
            Name of the index file in INDEX_DIR.

        Why pass df separately?
        - FAISS indices are pure vector stores — no metadata.
        - This separation is intentional: the index is lightweight and portable,
          while metadata can live in any database (Postgres, Redis, etc.).
        - In production, you'd have: FAISS for vector search → get IDs → lookup metadata in DB.
        """
        filepath = os.path.join(config.INDEX_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No FAISS index at {filepath}")

        self.index = faiss.read_index(filepath)
        self.corpus_df = df.copy()
        logger.info(
            f"FAISS index loaded: {self.index.ntotal} vectors, dim={self.index.d} "
            f"from {filepath}"
        )


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Build the FAISS index and run sample queries:
        python src/faiss_index.py

    Prerequisites:
        - Run src/data_loader.py first (creates processed_issues.parquet)
        - Run src/embed.py first (creates corpus_embeddings.npy)
    """
    from data_loader import load_processed_data
    from embed import EmbeddingEngine

    # Load data and embeddings
    df = load_processed_data()
    embeddings = np.load(os.path.join(config.DATA_DIR, "corpus_embeddings.npy"))
    logger.info(f"Loaded embeddings: {embeddings.shape}")

    # Build FAISS index
    engine = FAISSSearchEngine()
    engine.build_index(embeddings, df, index_type="flat_ip")

    # Save index for later use
    engine.save_index()

    # Run the same queries for comparison
    sample_queries = [
        "dataset download error",
        "how to fine-tune a model",
        "memory leak during training",
        "tokenizer not working",
    ]

    print("\n" + "=" * 70)
    print("FAISS SEMANTIC SEARCH DEMO (IndexFlatIP)")
    print("=" * 70)

    for query in sample_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        results = engine.search(query, top_k=5)
        for _, row in results.iterrows():
            snippet = row["text"][:100].replace("\n", " ")
            print(f"  [{row['rank']}] Score: {row['score']:.4f} | {snippet}...")
        print()

    # ─── Benchmark: FAISS vs NumPy brute-force ────────────────────────────
    print("\n" + "=" * 70)
    print("BENCHMARK: FAISS vs NumPy Brute-Force")
    print("=" * 70)

    # Encode a test query
    query_vec = engine.encode_query("memory leak during training")

    # FAISS timing
    import timeit
    faiss_time = timeit.timeit(
        lambda: engine.index.search(query_vec, 10), number=1000
    )

    # NumPy timing
    numpy_time = timeit.timeit(
        lambda: np.argsort(-(query_vec @ embeddings.T).flatten())[:10], number=1000
    )

    print(f"  FAISS IndexFlatIP:  {faiss_time*1000:.2f}ms for 1000 queries ({faiss_time:.4f}ms/query)")
    print(f"  NumPy dot product:  {numpy_time*1000:.2f}ms for 1000 queries ({numpy_time:.4f}ms/query)")
    print(f"  Speedup: {numpy_time/faiss_time:.1f}x")
    print("=" * 70)
