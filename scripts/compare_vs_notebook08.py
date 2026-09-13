import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import faiss
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix

# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────

print("=" * 70)
print("Loading artifacts…")

catalog = pd.read_parquet(ROOT / "data/catalog.parquet")
embeddings = np.load(ROOT / "data/product_embeddings.npy")
index = faiss.read_index(str(ROOT / "data/faiss_index.bin"))
train_cf = pd.read_parquet(ROOT / "data/train_cf.parquet")

user_ids = train_cf["reviewerID"].unique()
item_ids = train_cf["asin"].unique()
user_to_idx = {user: idx for idx, user in enumerate(user_ids)}
item_to_idx = {item: idx for idx, item in enumerate(item_ids)}
idx_to_item = {v: k for k, v in item_to_idx.items()}

rows = train_cf["reviewerID"].map(user_to_idx)
cols = train_cf["asin"].map(item_to_idx)
data = np.ones(len(train_cf))
interaction_matrix = csr_matrix(
    (data, (rows, cols)),
    shape=(len(user_ids), len(item_ids)),
)

with open(ROOT / "data/als_model.pkl", "rb") as f:
    nb_model_raw = pickle.load(f)

user_history = train_cf.groupby("reviewerID")["asin"].apply(set).to_dict()
item_popularity = train_cf.groupby("asin").size().to_dict()
max_popularity = max(item_popularity.values())
popularity_top10 = [
    asin for asin, _ in sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)[:10]
]

asin_to_idx = {asin: i for i, asin in enumerate(catalog["asin"])}

print(f"  catalog rows   : {len(catalog):,}")
print(f"  train_cf rows  : {len(train_cf):,}")
print(f"  known users    : {len(user_to_idx):,}")
print(f"  index.ntotal   : {index.ntotal:,}")
print(f"  Nb pkl type    : {type(nb_model_raw).__name__}")

# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────

def nb_als_candidates(user_id, n=50):
    if user_id not in user_to_idx:
        return []
    user_idx = user_to_idx[user_id]

    if hasattr(nb_model_raw, "_model"):

        ids, scores = nb_model_raw._model.recommend(
            nb_model_raw.user_to_idx.get(user_id, user_to_idx[user_id]),
            nb_model_raw._interaction_matrix[nb_model_raw.user_to_idx.get(user_id, user_idx)],
            N=n,
        )
        idx_to_item_nb = nb_model_raw.idx_to_item
    else:
        # Raw implicit model
        ids, scores = nb_model_raw.recommend(user_idx, interaction_matrix[user_idx], N=n)
        idx_to_item_nb = idx_to_item

    results = []
    for item_idx, score in zip(ids, scores):
        asin = idx_to_item_nb[item_idx]
        results.append((asin, float(score)))
    return results

def nb_content_neighbors(asin, n=10):
    
    if asin not in asin_to_idx:
        return []
    idx = asin_to_idx[asin]
    query = embeddings[idx].reshape(1, -1)
    scores, indices = index.search(query, n + 1)
    results = []
    for sim, i in zip(scores[0][1:], indices[0][1:]):
        results.append((catalog.iloc[i]["asin"], float(sim)))
    return results

def nb_popularity_score(asin):
    return item_popularity.get(asin, 0) / max_popularity

def nb_hybrid_recommend(user_id, top_k=10):
    if user_id not in user_history:
        return catalog[catalog["asin"].isin(popularity_top10)][["asin", "title"]]

    raw_als = nb_als_candidates(user_id, n=20)
    als_results = []
    if raw_als:
        als_scores = [s for _, s in raw_als]
        lo, hi = min(als_scores), max(als_scores)
        for asin, score in raw_als:
            scaled = (score - lo) / (hi - lo) if hi > lo else 1.0
            als_results.append((asin, scaled))

    candidate_scores = {}
    for asin, score in als_results:
        candidate_scores[asin] = {
            "als": score, "content": 0, "popularity": nb_popularity_score(asin)
        }

    for asin, _ in als_results:
        neighbors = nb_content_neighbors(asin, n=5)
        for neighbor_asin, sim in neighbors:
            if neighbor_asin not in candidate_scores:
                candidate_scores[neighbor_asin] = {
                    "als": 0, "content": sim, "popularity": nb_popularity_score(neighbor_asin)
                }
            else:
                candidate_scores[neighbor_asin]["content"] = max(
                    candidate_scores[neighbor_asin]["content"], sim
                )

    seen = user_history[user_id]
    for item in list(candidate_scores):
        if item in seen:
            del candidate_scores[item]

    ranked = []
    for asin, s in candidate_scores.items():
        final_score = 0.6 * s["als"] + 0.3 * s["content"] + 0.1 * s["popularity"]
        ranked.append((asin, final_score))
    ranked.sort(key=lambda x: x[1], reverse=True)

    results = []
    for asin, score in ranked[:top_k]:
        title = catalog.loc[catalog["asin"] == asin, "title"]
        if len(title):
            results.append({"asin": asin, "title": title.iloc[0], "hybrid_score": score})
    return pd.DataFrame(results)

# ──────────────────────────────────────────────────────────
# 3. Instantiate HybridRecommender
# ──────────────────────────────────────────────────────────

print("\nInstantiating HybridRecommender…")
from src.inference import HybridRecommender
rec = HybridRecommender()
print("  Ready.")

# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────

test_users = [
    train_cf["reviewerID"].iloc[500],
    train_cf["reviewerID"].iloc[0],
    train_cf["reviewerID"].iloc[1000],
]

print("\n" + "=" * 70)
print("COMPARISON: HybridRecommender vs Notebook 08 inline logic")
print("=" * 70)

all_match = True

for user_id in test_users:
    print(f"\n{'-'*70}")
    print(f"User: {user_id}")

    nb_df  = nb_hybrid_recommend(user_id, top_k=10)
    src_df = rec.recommend(user_id, top_k=10)

    nb_asins  = nb_df["asin"].tolist()  if "asin" in nb_df.columns  else []
    src_asins = src_df["asin"].tolist() if "asin" in src_df.columns else []

    print(f"\n  {'Rank':<5} {'Notebook 08 ASIN':<15} {'HybridRecommender ASIN':<15} Match?")
    print(f"  {'────':<5} {'────────────────':<15} {'──────────────────────':<15} ──────")

    max_len = max(len(nb_asins), len(src_asins))
    user_ok = True
    for i in range(max_len):
        nb_a  = nb_asins[i]  if i < len(nb_asins)  else "—"
        src_a = src_asins[i] if i < len(src_asins) else "—"
        match = "✓" if nb_a == src_a else "✗ DIFF"
        if nb_a != src_a:
            user_ok = False
            all_match = False
        print(f"  {i+1:<5} {nb_a:<15} {src_a:<15} {match}")

    if not user_ok:

        print("\n  Score comparison (first 5):")
        nb_scores  = dict(zip(nb_asins,  nb_df["hybrid_score"].tolist()))  if "hybrid_score" in nb_df.columns  else {}
        src_scores = dict(zip(src_asins, src_df["hybrid_score"].tolist())) if "hybrid_score" in src_df.columns else {}
        for asin in src_asins[:5]:
            nb_s  = nb_scores.get(asin,  "—")
            src_s = src_scores.get(asin, "—")
            print(f"    {asin}  nb={nb_s}  src={src_s}")

print(f"\n{'='*70}")
if all_match:
    print("RESULT: ALL 3 USERS MATCH EXACTLY ✓")
else:
    print("RESULT: DISCREPANCIES FOUND – see rows marked ✗ above")
print("=" * 70)

# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────

print("\n── explain() output for first user's top recommendation ──")
sample_user = test_users[0]
sample_recs = rec.recommend(sample_user, top_k=1)
if not sample_recs.empty:
    sample_asin = sample_recs.iloc[0]["asin"]
    for line in rec.explain(sample_user, sample_asin):
        print(line)
