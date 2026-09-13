import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.config import cfg

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

class HybridRecommender:
    

    def __init__(self):
        log.info("Loading HybridRecommender artifacts…")
        self._load_catalog()
        self._load_als()
        self._load_faiss()
        self._build_popularity_lookup()
        log.info("HybridRecommender ready.")

    def _load_catalog(self) -> None:
        path = ROOT / cfg.data.catalog
        self.catalog = pd.read_parquet(path)
        # lookup
        self.asin_to_idx: dict[str, int] = {
            asin: i for i, asin in enumerate(self.catalog["asin"])
        }
        log.info("Catalog loaded: %d products", len(self.catalog))

    def _load_als(self) -> None:
        
        import pickle as _pickle
        from src.models.als import ALSModel
        als_path = ROOT / cfg.data.als_model

        with open(als_path, "rb") as f:
            raw = _pickle.load(f)

        train_cf = pd.read_parquet(ROOT / cfg.data.train_cf)

        if isinstance(raw, ALSModel):
            # new format
            self._als = raw
        else:
            # legacy compat
            log.info("Legacy als_model.pkl detected (raw AlternatingLeastSquares). "
                     "Rebuilding encoder dicts from train_cf…")
            wrapper = ALSModel.__new__(ALSModel)
            wrapper._model = raw

            user_ids = train_cf["reviewerID"].unique()
            item_ids = train_cf["asin"].unique()
            wrapper.user_to_idx = {u: i for i, u in enumerate(user_ids)}
            wrapper.item_to_idx = {it: i for i, it in enumerate(item_ids)}
            wrapper.idx_to_item = {i: it for it, i in wrapper.item_to_idx.items()}

            from scipy.sparse import csr_matrix as _csr
            rows = train_cf["reviewerID"].map(wrapper.user_to_idx)
            cols = train_cf["asin"].map(wrapper.item_to_idx)
            wrapper._interaction_matrix = _csr(
                (np.ones(len(train_cf)), (rows, cols)),
                shape=(len(user_ids), len(item_ids)),
            )
            self._als = wrapper

        # history
        self._user_history: dict[str, set] = (
            train_cf.groupby("reviewerID")["asin"].apply(set).to_dict()
        )
        log.info("ALS model loaded. Known users: %d", len(self._als.user_to_idx))

    def _load_faiss(self) -> None:
        index_path     = ROOT / cfg.data.faiss_index
        embeddings_path = ROOT / cfg.data.embeddings
        self._index      = faiss.read_index(str(index_path))
        self._embeddings = np.load(embeddings_path).astype("float32")
        log.info("FAISS index loaded: %d vectors", self._index.ntotal)

    def _build_popularity_lookup(self) -> None:
        train_cf = pd.read_parquet(ROOT / cfg.data.train_cf)
        popularity = train_cf.groupby("asin").size().to_dict()
        max_pop    = max(popularity.values())
        self._popularity_score: dict[str, float] = {
            asin: count / max_pop for asin, count in popularity.items()
        }
        # fallback
        self._popularity_top_k: list[str] = [
            asin
            for asin, _ in sorted(popularity.items(), key=lambda x: x[1], reverse=True)[: cfg.api.top_k]
        ]
        log.info("Popularity lookup built (%d items).", len(self._popularity_score))

    def _als_candidates(self, user_id: str, n: int) -> list[tuple[str, float]]:
        
        raw = self._als.recommend(user_id, n)
        if not raw:
            return []
        scores = [s for _, s in raw]
        lo, hi = min(scores), max(scores)
        if hi > lo:
            return [(asin, (s - lo) / (hi - lo)) for asin, s in raw]
        return [(asin, 1.0) for asin, _ in raw]

    def _content_neighbors(self, asin: str, n: int) -> list[tuple[str, float]]:
        
        if asin not in self.asin_to_idx:
            return []
        idx   = self.asin_to_idx[asin]
        query = self._embeddings[idx].reshape(1, -1)
        scores, indices = self._index.search(query, n + 1)
        return [
            (self.catalog.iloc[i]["asin"], float(sim))
            for sim, i in zip(scores[0][1:], indices[0][1:])
        ]

    def _pop_score(self, asin: str) -> float:
        return self._popularity_score.get(asin, 0.0)

    def recommend(self, user_id: str, top_k: int = 10) -> pd.DataFrame:
        
        h = cfg.hybrid

        if user_id not in self._user_history:
            cold = self.catalog[self.catalog["asin"].isin(self._popularity_top_k)][
                ["asin", "title"]
            ].copy().reset_index(drop=True)
            cold["hybrid_score"] = None          
            return cold.head(top_k)

        als_results = self._als_candidates(user_id, n=h.als_candidates)

        candidate_scores: dict[str, dict] = {}

        for asin, als_score in als_results:
            candidate_scores[asin] = {
                "als":        als_score,
                "content":    0.0,
                "popularity": self._pop_score(asin),
            }

        for asin, _ in als_results:
            for neighbor_asin, sim in self._content_neighbors(asin, n=h.content_neighbors):
                if neighbor_asin not in candidate_scores:
                    candidate_scores[neighbor_asin] = {
                        "als":        0.0,
                        "content":    sim,
                        "popularity": self._pop_score(neighbor_asin),
                    }
                else:
                    candidate_scores[neighbor_asin]["content"] = max(
                        candidate_scores[neighbor_asin]["content"], sim
                    )

        seen = self._user_history[user_id]
        for item in list(candidate_scores):
            if item in seen:
                del candidate_scores[item]

        ranked = sorted(
            [
                (
                    asin,
                    h.weight_als        * s["als"]
                    + h.weight_content  * s["content"]
                    + h.weight_popularity * s["popularity"],
                )
                for asin, s in candidate_scores.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for asin, score in ranked[:top_k]:
            title = self.catalog.loc[self.catalog["asin"] == asin, "title"]
            if len(title):
                results.append({"asin": asin, "title": title.iloc[0], "hybrid_score": round(score, 6)})

        return pd.DataFrame(results)

    def explain(self, user_id: str, asin: str) -> list[str]:
        
        lines: list[str] = [
            f"Recommended ASIN : {asin}",
            "Signals used     :",
        ]

        if user_id not in self._user_history:
            lines.append("  • Popularity prior (Cold-Start fallback – no history found)")
            return lines

        als_recs = [item for item, _ in self._als_candidates(user_id, n=cfg.hybrid.als_candidates)]
        if asin in als_recs:
            lines.append("  • Collaborative filtering (ALS) – appeared in your personalised candidates")
        else:
            lines.append("  • Content similarity (FAISS) – semantically similar to one of your candidates")

        pop = self._pop_score(asin)
        lines.append(f"  • Popularity prior  (normalised score: {pop:.4f})")
        lines.append(
            f"  Blend weights: "
            f"ALS={cfg.hybrid.weight_als}  "
            f"Content={cfg.hybrid.weight_content}  "
            f"Popularity={cfg.hybrid.weight_popularity}"
        )
        return lines
