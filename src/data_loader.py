"""
Data Loader Module
==================
Responsible for:
1. Fetching the lewtun/github-issues dataset from Hugging Face
2. Cleaning raw text (HTML stripping, whitespace normalization)
3. Combining title + body into a single searchable document field
4. Filtering out low-quality rows (too short, empty, duplicates)
5. Persisting cleaned data to disk for reproducibility

Engineering Decisions:
- We use the `datasets` library (not a manual download) for reproducibility and caching.
- Cleaning is intentionally conservative: we strip HTML and normalize whitespace,
  but we KEEP code snippets. Why? In a GitHub issues dataset, code IS the content.
  A user searching "ImportError pandas" expects to find issues containing that traceback.
- We persist to Parquet (not CSV) because it preserves dtypes, is columnar (fast reads),
  and compresses well. This is a production best practice.
"""

import os
import re
import logging

import pandas as pd
from bs4 import BeautifulSoup
from datasets import load_dataset

# Allow imports from project root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Cleaning Functions ──────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """
    Remove HTML tags from text using BeautifulSoup.

    Why BeautifulSoup instead of regex?
    - Regex is fragile with nested/malformed HTML (common in GitHub issues).
    - BS4 handles edge cases like <br/>, unclosed tags, and entities.
    - Production code should never parse HTML with regex (famous StackOverflow answer).
    """
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple spaces/newlines into single spaces and strip edges.

    Why? GitHub issues often have:
    - Triple newlines between paragraphs
    - Indented code blocks leaving irregular spacing
    - Copy-paste artifacts

    We normalize for consistent tokenization downstream.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a single text field.
    Order matters: strip HTML first (exposes raw text), then normalize spacing.
    """
    text = strip_html(text)
    text = normalize_whitespace(text)
    return text


# ─── Core Data Loading ───────────────────────────────────────────────────────

def fetch_dataset() -> pd.DataFrame:
    """
    Load the lewtun/github-issues dataset from Hugging Face Hub.

    Returns a Pandas DataFrame for easier manipulation.

    Why convert to Pandas?
    - The HF `datasets` library is great for streaming/large data, but for ~3K rows
      Pandas gives us richer manipulation (groupby, string methods, easy null handling).
    - Pandas is also what most ML engineers know, making the code more readable.
    """
    logger.info(f"Loading dataset: {config.DATASET_NAME} (split: {config.DATASET_SPLIT})")
    dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)
    df = dataset.to_pandas()
    logger.info(f"Raw dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def prepare_documents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw dataset into cleaned, searchable documents.

    Steps:
    1. Combine title + body into a single 'text' field
    2. Clean the combined text
    3. Filter out documents below minimum length threshold
    4. Drop exact duplicates (common in issue trackers — bots, templates)
    5. Reset index for clean sequential access

    Why combine title + body?
    - Users search with short queries that might match either field.
    - A unified text field simplifies vectorization (one vector per document).
    - The title often summarizes; the body provides detail. Both are valuable.
    """
    logger.info("Preparing documents...")

    # Step 1: Combine title and body (handle NaN gracefully)
    # fillna("") ensures we don't get "NaN" as literal text
    df = df.copy()
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")
    df["text"] = df["title"].str.strip() + " " + df["body"].str.strip()

    # Step 2: Apply cleaning pipeline
    logger.info("Cleaning text...")
    df["text"] = df["text"].apply(clean_text)

    # Step 3: Filter short documents
    initial_count = len(df)
    df = df[df["text"].str.len() >= config.MIN_TEXT_LENGTH]
    filtered_count = initial_count - len(df)
    logger.info(f"Filtered {filtered_count} docs below {config.MIN_TEXT_LENGTH} chars")

    # Step 4: Drop duplicates on the cleaned text
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"])
    dedup_count = before_dedup - len(df)
    logger.info(f"Removed {dedup_count} duplicate documents")

    # Step 5: Reset index
    df = df.reset_index(drop=True)
    logger.info(f"Final corpus size: {len(df)} documents")

    return df


# ─── Persistence ─────────────────────────────────────────────────────────────

def save_processed_data(df: pd.DataFrame, filename: str = "processed_issues.parquet") -> str:
    """
    Save cleaned DataFrame to Parquet format.

    Why Parquet over CSV?
    - Binary format → 3-5x smaller file size
    - Preserves column types (no int→float conversion on NaN)
    - Columnar storage → fast reads when you only need certain columns
    - Industry standard for data pipelines (Spark, BigQuery, DuckDB all use it)
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    filepath = os.path.join(config.DATA_DIR, filename)
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved processed data to: {filepath}")
    return filepath


def load_processed_data(filename: str = "processed_issues.parquet") -> pd.DataFrame:
    """
    Load previously processed data from disk.

    This avoids re-downloading and re-cleaning every time you run an experiment.
    Idempotent data pipelines are a production must-have.
    """
    filepath = os.path.join(config.DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Processed data not found at {filepath}. "
            "Run the data preparation pipeline first."
        )
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded processed data: {df.shape[0]} documents from {filepath}")
    return df


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    """
    End-to-end data preparation pipeline.

    Checks if processed data already exists (caching for fast iteration).
    If not, fetches from HuggingFace, cleans, and persists.

    Returns the cleaned DataFrame ready for vectorization.
    """
    processed_path = os.path.join(config.DATA_DIR, "processed_issues.parquet")

    if os.path.exists(processed_path):
        logger.info("Found cached processed data. Loading from disk...")
        return load_processed_data()

    # Full pipeline
    raw_df = fetch_dataset()
    clean_df = prepare_documents(raw_df)
    save_processed_data(clean_df)

    return clean_df


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this script directly to execute the full data preparation pipeline:
        python src/data_loader.py

    This is a common pattern: modules are importable AND runnable.
    """
    df = run_pipeline()

    # Print summary statistics for verification
    print("\n" + "=" * 60)
    print("DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Total documents: {len(df)}")
    print(f"Avg text length: {df['text'].str.len().mean():.0f} chars")
    print(f"Min text length: {df['text'].str.len().min()} chars")
    print(f"Max text length: {df['text'].str.len().max()} chars")
    print(f"\nSample document (first 200 chars):")
    print(f"  '{df['text'].iloc[0][:200]}...'")
    print("=" * 60)
