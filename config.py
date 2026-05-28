"""
Central Configuration for Embedding-Based Semantic Search Engine
================================================================
Author: Vinayak Vinod

Why a central config?
- Single source of truth for all hyperparameters and paths
- Easy to tweak experiments without hunting through multiple files
- Production systems externalize config (env vars, YAML); this is the stepping stone
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INDEX_DIR = os.path.join(PROJECT_ROOT, "indexes")

# ─── Dataset ─────────────────────────────────────────────────────────────────
DATASET_NAME = "lewtun/github-issues"
DATASET_SPLIT = "train"

# Minimum character length for a document to be considered valid after cleaning.
# Why 30? Anything shorter (e.g., "fixed" or "see above") carries no semantic value.
MIN_TEXT_LENGTH = 30

# ─── TF-IDF Baseline ────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 10_000  # Vocabulary cap — prevents memory blowup on large corpora
TFIDF_NGRAM_RANGE = (1, 2)   # Unigrams + bigrams capture short phrases like "import error"

# ─── Embedding Model (Phase 2) ──────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Output dimensionality of all-MiniLM-L6-v2
EMBEDDING_BATCH_SIZE = 64  # Batch size for encoding — tune based on GPU/RAM

# ─── Retrieval ───────────────────────────────────────────────────────────────
TOP_K = 10  # Number of results to return per query

# ─── Evaluation (Phase 4) ────────────────────────────────────────────────────
EVAL_K_VALUES = [1, 3, 5, 10]  # Precision@K evaluated at these cutoffs
