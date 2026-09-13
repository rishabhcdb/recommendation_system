# Hybrid Recommendation System

A production-ready, two-stage hybrid recommender system trained on the **Amazon Electronics Reviews** dataset (~1.7 M interactions). The system mirrors the candidate-generation + ranking architecture used by large consumer platforms.

---

## Architecture

```
User ID
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 1 – Candidate Generation (ALS)                      │
│  Implicit-feedback matrix factorisation retrieves the     │
│  top-N personalised candidates from collaborative signals  │
└─────────────────────────┬────────────────────────────────┘
                          │  top-20 candidates
                          ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 2 – Content Expansion (FAISS)                        │
│  For each candidate, a FAISS inner-product index finds     │
│  semantically similar items using sentence-transformer     │
│  embeddings of product title + brand + description         │
└─────────────────────────┬────────────────────────────────┘
                          │  ~120 candidates (20 × 5 neighbours + originals)
                          ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 3 – Weighted Ranking                                 │
│  final_score = 0.6 × ALS_score                            │
│              + 0.3 × content_similarity                   │
│              + 0.1 × popularity_prior                     │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
                    Top-K results
```

> **Cold-start handling** – users with no interaction history automatically receive the global popularity-ranked list.

---

## Real-world Applicability

The architecture is domain-agnostic — it generalizes to any product with implicit user-item interactions and item metadata (e-commerce, streaming, food delivery, job boards, marketplaces):

| This project | Any recommendation domain |
|---|---|
| User reviews (ratings) | Purchases, orders, clicks, watch time — any implicit signal |
| Product ASINs (IDs) | Items, listings, restaurants, titles, jobs |
| Product embeddings (title + description) | Item embeddings from any text/tag metadata |
| ALS collaborative signals | "Users who engaged with X also engaged with Y" |
| FAISS content expansion | Semantically similar items, at scale |
| Cold-start fallback | New-user onboarding via trending/popular items |

---

## Project Structure

```
.
├── config.yaml                  # All hyperparameters and paths in one place
├── requirements.txt             # Pinned Python dependencies
├── Dockerfile                   # Containerises the API for deployment
├── app.py                       # FastAPI REST API (serve the model)
│
├── src/
│   ├── config.py                # Loads config.yaml → cfg object
│   ├── features.py              # Metadata cleaning + embedding generation
│   ├── build_index.py           # Builds FAISS vector index
│   ├── train.py                 # Full pipeline CLI entry-point
│   ├── evaluate.py              # Offline metrics: Precision, Recall, NDCG @K
│   ├── inference.py             # HybridRecommender class (scoring/blending)
│   └── models/
│       └── als.py               # ALSModel wrapper (fit / recommend / save / load)
│
├── notebooks/
│   ├── 01_data_understanding.ipynb      # EDA – distributions, sparsity
│   ├── 02_popularity_baseline.ipynb     # Baseline + train/test split
│   ├── 03_item_collaborative_filtering.ipynb  # Item-KNN (exploratory)
│   ├── 04_metadata_processing.ipynb    # Metadata cleaning
│   ├── 05_embeddings.ipynb             # Sentence-transformer embeddings
│   ├── 06_ALS_matrix_factorization.ipynb     # ALS training + eval
│   ├── 07_FAISS.ipynb                  # FAISS index building + latency test
│   └── 08_hybrid_recommender.ipynb     # Full hybrid pipeline + explainability
│
└── data/                        # Generated artifacts (not committed to git)
    ├── reviews/
    ├── metadata/
    ├── catalog.parquet
    ├── interactions.parquet
    ├── train.parquet / test.parquet / train_cf.parquet
    ├── product_embeddings.npy
    ├── faiss_index.bin
    └── als_model.pkl
```

---

## Offline Evaluation Results

Evaluation uses a **leave-one-out** split (last interaction per user held out for test). Metrics computed over 1,000 randomly sampled users.

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity baseline | — | — | — |
| ALS (collaborative) | — | — | — |
| **Hybrid (ALS + Content)** | **—** | **—** | **—** |

> Run `python -m src.evaluate` after training to populate this table.

---

## Quickstart

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Place raw data

```
data/reviews/Elec_5C.parquet
data/metadata/metadata_filtered.parquet
```

### 3 · Run the full training pipeline

```bash
# Full run (features + FAISS index + ALS + evaluation)
python -m src.train

# Skip expensive embedding step if catalog + embeddings already exist
python -m src.train --skip-features

# Skip FAISS rebuild too
python -m src.train --skip-features --skip-index
```

### 4 · Start the API locally

```bash
uvicorn app:app --reload
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

### 5 · Try an API call

```bash
# Get recommendations for a user
curl "http://localhost:8000/recommend/AO94DHGC771SJ"

# Explain a specific recommendation
curl "http://localhost:8000/explain/AO94DHGC771SJ/0528881469"
```

---

## Docker Deployment

```bash
# Build the image
docker build -t hybrid-recommender .

# Run the container
docker run -p 8000:8000 hybrid-recommender

# The API is now available at http://localhost:8000
```

---

## Configuration

All tunable parameters live in [`config.yaml`](config.yaml). No hardcoded values in source code.

```yaml
hybrid:
  weight_als:        0.6   # ← tweak blend weights here
  weight_content:    0.3
  weight_popularity: 0.1
  als_candidates:    20    # candidates fetched from ALS
  content_neighbors: 5     # FAISS neighbours per candidate
```

---

## Tech Stack

| Component | Library |
|---|---|
| Collaborative filtering | `implicit` (ALS) |
| Product embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector search | `faiss-cpu` |
| Data processing | `pandas`, `numpy`, `scipy` |
| API | `FastAPI` + `uvicorn` |
| Containerisation | `Docker` |

---

## Dataset

**Amazon Product Reviews – Electronics (5-core)**  
J. McAuley et al., *Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class Collaborative Filtering*, WWW 2016.