"""
Streamlit Search Application
============================
Author: Vinayak Vinod

A production-style web interface for the Embedding-Based Semantic Search Engine.
Users can type natural language queries and see retrieved GitHub issues ranked
by relevance.

Features:
- Side-by-side comparison: TF-IDF vs Semantic Search
- Adjustable top-K slider
- Score visualization
- Query latency display
- Expandable result cards showing full text

Architecture:
    User Query → [Streamlit UI] → [Search Engine Backend] → [Formatted Results]

The app loads pre-built indexes at startup (cached) for instant queries.

Usage:
    streamlit run src/app.py

Engineering Decisions:
- @st.cache_resource for model/index loading (loaded once, shared across sessions)
- Results displayed as expandable cards (not a raw table) for better UX
- Both engines available for A/B comparison (educational + impressive for demo)
- Responsive layout using Streamlit columns
"""

import os
import sys
import time

import streamlit as st
import pandas as pd

# ─── Path Setup ──────────────────────────────────────────────────────────────
# Ensure imports work regardless of where streamlit is launched from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_corpus():
    """
    Load the processed corpus DataFrame.

    @st.cache_resource ensures this is loaded ONCE and shared across all users/sessions.
    Without caching, every page reload would re-read the parquet file.
    """
    from data_loader import load_processed_data
    return load_processed_data()


@st.cache_resource
def load_tfidf_engine():
    """Load the pre-built TF-IDF search engine."""
    from tfidf_search import TFIDFSearchEngine
    return TFIDFSearchEngine.load()


@st.cache_resource
def load_faiss_engine(_df):
    """
    Load the pre-built FAISS index and associate it with corpus metadata.

    Note: _df prefix with underscore tells Streamlit not to hash this parameter
    (DataFrames are expensive to hash).
    """
    from faiss_index import FAISSSearchEngine
    engine = FAISSSearchEngine()
    engine.load_index(_df)
    return engine


# ═══════════════════════════════════════════════════════════════════════════════
# UI HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def display_results(results: pd.DataFrame, engine_name: str, query_time: float):
    """
    Display search results in a clean, readable format.

    Each result is shown as an expandable card with:
    - Rank and similarity score
    - Title (bold)
    - Truncated text preview
    - Full text in expander
    """
    st.markdown(f"**{engine_name}** — {len(results)} results in {query_time*1000:.1f}ms")

    if results.empty:
        st.warning("No results found.")
        return

    for _, row in results.iterrows():
        rank = int(row["rank"])
        score = float(row["score"])
        title = row["title"] if row["title"] else "Untitled"
        text = row["text"]

        # Score-based color indicator
        if score >= 0.7:
            score_color = "🟢"
        elif score >= 0.4:
            score_color = "🟡"
        else:
            score_color = "🔴"

        # Result card
        with st.container():
            st.markdown(
                f"**{rank}.** {score_color} Score: `{score:.4f}` — **{title[:80]}**"
            )
            # Show first 200 chars as preview
            preview = text[:200].replace("\n", " ")
            st.caption(f"{preview}...")

            # Expandable full text
            with st.expander(f"View full text (#{rank})"):
                st.text(text[:2000])  # Cap at 2000 chars for performance

        st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main Streamlit application."""

    # ─── Page Configuration ───────────────────────────────────────────────
    st.set_page_config(
        page_title="Semantic Search Engine",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ─── Header ───────────────────────────────────────────────────────────
    st.title("🔍 Embedding-Based Semantic Search Engine")
    st.markdown(
        """
        Search through **2,734 GitHub issues** using either lexical (TF-IDF) or
        semantic (transformer + FAISS) search. Type a natural language query below
        to find relevant issues.

        *Built by Vinayak Vinod | Powered by all-MiniLM-L6-v2 + FAISS*
        """
    )

    # ─── Sidebar Controls ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("Settings")

        search_mode = st.radio(
            "Search Mode",
            options=["Semantic (FAISS)", "TF-IDF (Baseline)", "Compare Both"],
            index=0,
            help="Choose which search engine to use, or compare both side by side."
        )

        top_k = st.slider(
            "Number of results (Top-K)",
            min_value=1,
            max_value=20,
            value=5,
            help="How many results to retrieve per query."
        )

        st.divider()
        st.markdown("### About")
        st.markdown(
            """
            **Dataset**: lewtun/github-issues (HuggingFace)

            **Models**:
            - TF-IDF: Scikit-learn (10K vocab, bigrams)
            - Semantic: all-MiniLM-L6-v2 (384-d)
            - Index: FAISS IndexFlatIP

            **Metrics** (200 eval queries):
            | Metric | TF-IDF | Semantic |
            |--------|--------|----------|
            | MRR | 0.722 | **0.831** |
            | Hit@1 | 61.5% | **74.5%** |
            | NDCG@10 | 0.772 | **0.862** |
            """
        )

    # ─── Load Resources ───────────────────────────────────────────────────
    with st.spinner("Loading search engines..."):
        df = load_corpus()
        tfidf_engine = load_tfidf_engine()
        faiss_engine = load_faiss_engine(df)

    # ─── Search Input ─────────────────────────────────────────────────────
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., dataset download error, memory leak during training, tokenizer not working",
        help="Type a natural language query describing the issue you're looking for."
    )

    # ─── Execute Search ───────────────────────────────────────────────────
    if query:
        st.markdown(f"---")
        st.markdown(f"### Results for: *\"{query}\"*")

        if search_mode == "Compare Both":
            # Side-by-side comparison
            col1, col2 = st.columns(2)

            with col1:
                start = time.time()
                tfidf_results = tfidf_engine.search(query, top_k=top_k)
                tfidf_time = time.time() - start
                display_results(tfidf_results, "TF-IDF (Lexical)", tfidf_time)

            with col2:
                start = time.time()
                faiss_results = faiss_engine.search(query, top_k=top_k)
                faiss_time = time.time() - start
                display_results(faiss_results, "Semantic (FAISS)", faiss_time)

        elif search_mode == "Semantic (FAISS)":
            start = time.time()
            results = faiss_engine.search(query, top_k=top_k)
            query_time = time.time() - start
            display_results(results, "Semantic Search (FAISS)", query_time)

        else:  # TF-IDF
            start = time.time()
            results = tfidf_engine.search(query, top_k=top_k)
            query_time = time.time() - start
            display_results(results, "TF-IDF Baseline", query_time)

    else:
        # Show example queries when no input
        st.markdown("---")
        st.markdown("### Try these example queries:")
        example_queries = [
            "dataset download error",
            "how to fine-tune a model",
            "memory leak during training",
            "tokenizer not working",
            "loading data from local files",
            "GPU out of memory",
        ]

        cols = st.columns(3)
        for i, example in enumerate(example_queries):
            with cols[i % 3]:
                st.code(example, language=None)


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
