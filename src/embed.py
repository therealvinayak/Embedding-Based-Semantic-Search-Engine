"""
Dense Embedding Module
======================
Author: Vinayak Vinod

Responsible for:
1. Loading the sentence-transformer model (all-MiniLM-L6-v2)
2. Encoding the corpus into dense 384-d vectors (with batching)
3. Saving/loading embeddings to disk (NumPy format)

Engineering Decisions:
- We use sentence-transformers (not raw HuggingFace transformers) because it handles:
    * Proper mean pooling (not [CLS])
    * L2 normalization (optional, but good for cosine similarity)
    * Batching with progress bars
    * GPU/CPU fallback automatically
- Embeddings are saved as .npy files (NumPy binary format):
    * Faster to load than CSV/Parquet for dense arrays
    * Preserves float32 precision exactly
    * Memory-mapped loading possible for very large corpora
- We normalize embeddings to unit length. Why?
    * After normalization: cosine_similarity(a, b) = dot(a, b)
    * Dot product is cheaper to compute than full cosine (no division)
    * FAISS IndexFlatIP (inner product) on normalized vectors = cosine search
"""

import os
import time
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

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


class EmbeddingEngine:
    """
    Encodes text documents into dense semantic vectors using a pre-trained
    sentence-transformer model.

    Why a class?
    - Model loading is expensive (~2-5 seconds). We load once and reuse.
    - Encapsulates the model + embeddings + metadata together.
    - Consistent interface with TFIDFSearchEngine (Strategy Pattern).
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        """
        Load the sentence-transformer model.

        Parameters
        ----------
        model_name : str
            HuggingFace model identifier. Default: all-MiniLM-L6-v2
            - 384 dimensions
            - 6 transformer layers (fast)
            - Trained on 1B+ sentence pairs for semantic similarity
            - ~80MB model size

        The model downloads on first use and is cached in ~/.cache/huggingface/
        """
        logger.info(f"Loading embedding model: {model_name}")
        start = time.time()
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded in {time.time() - start:.2f}s")
        logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        self.embeddings: Optional[np.ndarray] = None
        self.corpus_df: Optional[pd.DataFrame] = None

    def encode_corpus(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode all documents in the corpus into dense vectors.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned corpus DataFrame.
        text_column : str
            Column containing the text to encode.
        batch_size : int
            Number of documents to encode at once.

            Why batching matters:
            - GPU memory is finite. 64 docs × 512 tokens × 384-d fits in ~2GB VRAM.
            - On CPU, batching still helps due to parallel SIMD operations.
            - Too large → OOM. Too small → underutilizes hardware.
            - 64 is a safe default for most machines.
        normalize : bool
            Whether to L2-normalize vectors to unit length.

            Why normalize?
            - cosine_similarity(a, b) = dot(a, b) when ||a|| = ||b|| = 1
            - FAISS IndexFlatIP on normalized vectors = exact cosine search
            - Saves computation at query time (no division needed)

        Returns
        -------
        np.ndarray
            Shape: (n_documents, embedding_dim) — e.g., (2734, 384)
            dtype: float32
        """
        logger.info(f"Encoding {len(df)} documents (batch_size={batch_size})...")
        self.corpus_df = df.copy()

        texts = df[text_column].tolist()
        start = time.time()

        # sentence-transformers handles batching internally with show_progress_bar
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,  # Returns np.ndarray (not torch.Tensor)
            normalize_embeddings=normalize,  # L2 normalization
        )

        elapsed = time.time() - start
        docs_per_sec = len(texts) / elapsed

        logger.info(
            f"Encoding complete: {self.embeddings.shape} in {elapsed:.1f}s "
            f"({docs_per_sec:.0f} docs/sec)"
        )
        logger.info(
            f"Memory usage: {self.embeddings.nbytes / 1024 / 1024:.2f} MB "
            f"(float32, {self.embeddings.dtype})"
        )

        # Verify normalization (sanity check)
        if normalize:
            norms = np.linalg.norm(self.embeddings[:5], axis=1)
            logger.info(f"Sample norms (should be ~1.0): {norms.round(4)}")

        return self.embeddings

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query into the same vector space as the corpus.

        Returns shape (1, 384) for compatibility with similarity computations.

        Why a separate method?
        - Queries are short (5-15 words), documents can be 1000+ words.
        - In production, query encoding must be FAST (user is waiting).
        - Single query encoding takes ~5-10ms on CPU — well within latency budget.
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embedding  # Shape: (1, 384)

    def search(self, query: str, top_k: int = config.TOP_K) -> pd.DataFrame:
        """
        Brute-force semantic search using dot product (= cosine sim on normalized vectors).

        This is the NAIVE approach — works perfectly for <10K documents.
        Phase 3 replaces this with FAISS for scalability.

        Parameters
        ----------
        query : str
            Natural language search query.
        top_k : int
            Number of results to return.

        Returns
        -------
        pd.DataFrame
            Top-K results with [rank, score, title, text]
        """
        if self.embeddings is None:
            raise RuntimeError("No embeddings computed. Call encode_corpus() first.")

        # Encode the query
        query_embedding = self.encode_query(query)  # (1, 384)

        # Compute dot product similarity (= cosine since vectors are normalized)
        # Matrix multiplication: (1, 384) × (384, n_docs) → (1, n_docs)
        similarities = (query_embedding @ self.embeddings.T).flatten()

        # Get top-K
        top_indices = similarities.argsort()[::-1][:top_k]
        top_scores = similarities[top_indices]

        # Build results
        results = self.corpus_df.iloc[top_indices][["title", "text"]].copy()
        results.insert(0, "rank", range(1, top_k + 1))
        results.insert(1, "score", top_scores)
        results = results.reset_index(drop=True)

        return results

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save_embeddings(self, filename: str = "corpus_embeddings.npy") -> str:
        """
        Save computed embeddings to disk as a NumPy binary file.

        Why .npy over .csv or .parquet?
        - Binary format: no parsing overhead, instant load
        - Exact precision preservation (no float→string→float conversion)
        - Supports memory-mapping (np.load with mmap_mode) for huge files
        - 2734 × 384 × 4 bytes = ~4 MB (tiny!)
        """
        if self.embeddings is None:
            raise RuntimeError("No embeddings to save. Call encode_corpus() first.")

        os.makedirs(config.DATA_DIR, exist_ok=True)
        filepath = os.path.join(config.DATA_DIR, filename)
        np.save(filepath, self.embeddings)
        logger.info(f"Embeddings saved to: {filepath} ({self.embeddings.shape})")
        return filepath

    def load_embeddings(self, filename: str = "corpus_embeddings.npy") -> np.ndarray:
        """
        Load pre-computed embeddings from disk.

        Why cache embeddings?
        - Encoding 2734 docs takes ~30-60s on CPU. You don't want to redo this
          every time you restart the app or tweak the search logic.
        - In production, embeddings are computed once (or on a schedule) and
          served from disk/memory.
        """
        filepath = os.path.join(config.DATA_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"No embeddings at {filepath}. Run encode_corpus() first."
            )
        self.embeddings = np.load(filepath)
        logger.info(f"Loaded embeddings: {self.embeddings.shape} from {filepath}")
        return self.embeddings


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this to encode the corpus and test semantic search:
        python src/embed.py

    Compare results with TF-IDF to see the semantic improvement!
    """
    from data_loader import load_processed_data

    # Load cleaned data
    df = load_processed_data()

    # Initialize embedding engine
    engine = EmbeddingEngine()

    # Encode the corpus
    embeddings = engine.encode_corpus(df)

    # Save embeddings for reuse
    engine.save_embeddings()

    # Run the SAME queries as TF-IDF for comparison
    sample_queries = [
        "dataset download error",
        "how to fine-tune a model",
        "memory leak during training",
        "tokenizer not working",
    ]

    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH (Dense Embeddings) DEMO")
    print("=" * 70)

    for query in sample_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        results = engine.search(query, top_k=5)
        for _, row in results.iterrows():
            snippet = row["text"][:100].replace("\n", " ")
            print(f"  [{row['rank']}] Score: {row['score']:.4f} | {snippet}...")
        print()

    print("=" * 70)
    print("✅ Compare these results with the TF-IDF output above!")
    print("   Notice how 'memory leak during training' now finds ACTUAL bugs,")
    print("   not just documents containing the word 'memory'.")
    print("=" * 70)
