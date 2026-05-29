# Embedding-Based Semantic Search Engine

A full-stack semantic search engine that retrieves relevant GitHub issues using dense vector embeddings and FAISS, significantly outperforming traditional lexical (TF-IDF) search.

---

## Results

| Metric | TF-IDF Baseline | Semantic (FAISS) | Improvement |
|--------|:-:|:-:|:-:|
| **MRR** | 0.722 | **0.831** | +15.1% |
| **Hit@1** | 61.5% | **74.5%** | +21.1% |
| **NDCG@10** | 0.772 | **0.862** | +11.6% |
| **Hit@10** | 93.0% | **95.5%** | +2.7% |

Evaluated on 200 synthetic queries (title-as-query methodology) with a fixed random seed for reproducibility.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Sentence Transformer │  ← all-MiniLM-L6-v2 (384-d)
│  Query Encoding       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  FAISS IndexFlatIP   │  ← Inner Product on normalized vectors = Cosine Similarity
│  Vector Search       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Top-K Results       │  ← Ranked by semantic similarity
│  + Metadata Lookup   │
└─────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Dataset | HuggingFace `lewtun/github-issues` | 3,019 raw GitHub issues |
| Text Processing | Pandas, BeautifulSoup | Cleaning, deduplication, filtering |
| Baseline Search | Scikit-learn TF-IDF | Lexical retrieval (10K vocab, bigrams) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | 384-d dense semantic vectors |
| Vector Index | FAISS (`IndexFlatIP`) | Exact nearest neighbor search |
| Evaluation | Custom implementation | Precision@K, MRR, NDCG@K |
| Web UI | Streamlit | Interactive search interface |

---

## Project Structure

```
├── config.py                 # Central configuration (model, paths, hyperparams)
├── requirements.txt          # Python dependencies
├── src/
│   ├── data_loader.py       # ETL: fetch, clean, persist corpus (3,019 → 2,734 docs)
│   ├── tfidf_search.py      # TF-IDF baseline search engine
│   ├── embed.py             # Dense embedding encoding with batching
│   ├── faiss_index.py       # FAISS index building, querying, persistence
│   ├── evaluate.py          # Formal evaluation framework
│   └── app.py              # Streamlit web application
├── data/                    # Generated: processed corpus + embeddings
│   ├── processed_issues.parquet
│   └── corpus_embeddings.npy
├── indexes/                 # Generated: serialized search indexes
│   ├── tfidf_engine.pkl
│   └── faiss_index.bin
└── notebooks/               # Exploration (optional)
```

---

## Setup & Usage

### Prerequisites

- Python 3.9+
- ~500 MB disk space (model weights + data artifacts)

### Installation

```bash
git clone https://github.com/therealvinayak/Embedding-Based-Semantic-Search-Engine.git
cd Embedding-Based-Semantic-Search-Engine
pip install -r requirements.txt
```

### Build the Pipeline

Run each module in sequence to build all artifacts:

```bash
# Step 1: Fetch and clean the dataset
python src/data_loader.py

# Step 2: Generate dense embeddings (~30s on CPU)
python src/embed.py

# Step 3: Build TF-IDF baseline index
python src/tfidf_search.py

# Step 4: Build FAISS vector index
python src/faiss_index.py

# Step 5: Run evaluation
python src/evaluate.py
```

### Launch the Web App

```bash
streamlit run src/app.py
```

Open `http://localhost:8501` in your browser. Features:
- Natural language query input
- Toggle between Semantic, TF-IDF, or side-by-side comparison
- Adjustable top-K results slider
- Score-based relevance indicators
- Expandable full-text result cards

---

## How It Works

### Phase 1: Data Preparation & TF-IDF Baseline

- Fetches `lewtun/github-issues` dataset (3,019 issues) from HuggingFace
- Combines title + body, strips HTML, normalizes whitespace
- Filters short docs (<30 chars) and deduplicates → 2,734 clean documents
- Builds a TF-IDF sparse matrix (2,734 x 10,000) with L2 normalization
- Serves as the baseline for comparison

### Phase 2: Dense Embeddings

- Encodes all documents using `all-MiniLM-L6-v2` (6-layer distilled transformer)
- Produces 384-dimensional vectors via mean pooling over token embeddings
- L2-normalizes vectors so dot product = cosine similarity
- Total memory: ~4 MB for entire corpus

### Phase 3: FAISS Indexing

- Builds an `IndexFlatIP` (exact inner product search) over normalized embeddings
- Supports configurable index types: Flat, IVF, HNSW for different scale requirements
- Index build time: <10ms for 2,734 vectors
- Query time: <1ms (excluding model encoding)

### Phase 4: Evaluation

- Creates synthetic evaluation set: document titles as queries, source document as ground truth
- Computes Precision@K, Hit@K, MRR (Mean Reciprocal Rank), and NDCG@K
- Known limitation: title-as-query introduces lexical bias favoring TF-IDF. Despite this, semantic search wins across all metrics.

### Phase 5: Streamlit Application

- Cached resource loading (model + indexes loaded once)
- Side-by-side engine comparison mode
- Sub-20ms query latency

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Parquet over CSV | Binary, columnar, preserves dtypes, 3-5x smaller |
| `max_features=10000` | Caps vocabulary to prevent memory explosion with diminishing returns |
| `sublinear_tf=True` | Dampens effect of very frequent terms in long documents |
| Mean pooling over [CLS] | sentence-transformers are trained with mean pooling objective |
| L2 normalization | Enables dot product = cosine similarity (cheaper computation) |
| `IndexFlatIP` over `IndexFlatL2` | Higher score = better match (intuitive ranking) |
| Float32 embeddings | 50% memory savings vs float64, negligible precision loss |
| Pickle for TF-IDF | Scikit-learn objects are pickle-compatible by design |
| `.npy` for embeddings | Binary, instant load, supports memory mapping |

---

## Potential Extensions

- **Hybrid search**: Weighted combination of TF-IDF + semantic scores
- **Cross-encoder re-ranking**: Two-stage retrieval (fast recall → accurate re-rank)
- **Query expansion**: Add synonyms before encoding to improve recall
- **Approximate NN**: Switch to HNSW or IVF for >100K document scale
- **FastAPI backend**: REST API for integration with other services
- **Fine-tuning**: Train the embedding model on domain-specific data

---

## References

- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
- [HuggingFace Semantic Search Tutorial](https://huggingface.co/docs/datasets/faiss_es)
- [lewtun/github-issues Dataset](https://huggingface.co/datasets/lewtun/github-issues)

---

## License

MIT
