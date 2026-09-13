import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    hits = len(set(recommended[:k]) & relevant)
    return hits / k

def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / len(relevant)

def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )
    # ideal
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def _avg_metrics(recs_fn, ground_truth: dict, k: int, sample_users: list) -> dict:
    
    p_list, r_list, n_list = [], [], []
    for user in tqdm(sample_users, leave=False):
        recs    = recs_fn(user)
        relevant = ground_truth.get(user, set())
        p_list.append(precision_at_k(recs, relevant, k))
        r_list.append(recall_at_k(recs, relevant, k))
        n_list.append(ndcg_at_k(recs, relevant, k))
    return {
        f"Precision@{k}": np.mean(p_list),
        f"Recall@{k}":    np.mean(r_list),
        f"NDCG@{k}":      np.mean(n_list),
    }

def _make_popularity_fn(train_cf: pd.DataFrame, k: int):
    popularity = train_cf.groupby("asin").size().sort_values(ascending=False)
    top_k_asins = popularity.index[:k].tolist()

    def fn(user_id: str) -> list:
        return top_k_asins

    return fn

def _make_als_fn(als_model, n: int):
    def fn(user_id: str) -> list:
        return [asin for asin, _ in als_model.recommend(user_id, n=n)]

    return fn

def _make_hybrid_fn(recommender, top_k: int):
    def fn(user_id: str) -> list:
        df = recommender.recommend(user_id, top_k=top_k)
        return df["asin"].tolist()

    return fn

def run_evaluation(seed: int | None = None) -> pd.DataFrame:
    
    eval_cfg = cfg.evaluation
    k           = eval_cfg.k
    sample_size = eval_cfg.sample_size
    seed        = seed or eval_cfg.seeds[0]

    test_path = ROOT / cfg.data.test
    log.info("Loading test set from %s", test_path)
    test = pd.read_parquet(test_path)
    ground_truth: dict[str, set] = (
        test.groupby("reviewerID")["asin"].apply(set).to_dict()
    )

    np.random.seed(seed)
    all_users    = list(ground_truth.keys())
    sample_users = np.random.choice(all_users, size=min(sample_size, len(all_users)), replace=False).tolist()
    log.info("Evaluating on %d sampled users (seed=%d)", len(sample_users), seed)

    train_cf = pd.read_parquet(ROOT / cfg.data.train_cf)

    from src.inference import HybridRecommender
    hybrid = HybridRecommender()
    als_model = hybrid._als

    log.info("Evaluating Popularity baseline…")
    pop_metrics = _avg_metrics(_make_popularity_fn(train_cf, k), ground_truth, k, sample_users)

    log.info("Evaluating ALS…")
    als_metrics = _avg_metrics(_make_als_fn(als_model, k), ground_truth, k, sample_users)

    log.info("Evaluating Hybrid…")
    hyb_metrics = _avg_metrics(_make_hybrid_fn(hybrid, k), ground_truth, k, sample_users)

    results = pd.DataFrame(
        [pop_metrics, als_metrics, hyb_metrics],
        index=["Popularity baseline", "ALS", "Hybrid (ALS + Content)"],
    )
    results.index.name = "Model"

    log.info("\n\n%s\n", results.to_string(float_format="{:.4f}".format))
    return results

if __name__ == "__main__":
    run_evaluation()
