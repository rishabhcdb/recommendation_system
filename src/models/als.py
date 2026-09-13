import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from src.config import cfg

log = logging.getLogger(__name__)

class ALSModel:
    

    def __init__(self):
        als_cfg = cfg.als
        self._model = AlternatingLeastSquares(
            factors=als_cfg.factors,
            regularization=als_cfg.regularization,
            iterations=als_cfg.iterations,
            random_state=als_cfg.random_state,
        )
        self.user_to_idx: dict[str, int] = {}
        self.item_to_idx: dict[str, int] = {}
        self.idx_to_item: dict[int, str] = {}
        self._interaction_matrix: csr_matrix | None = None

    def fit(self, train_cf: pd.DataFrame) -> "ALSModel":
        
        log.info("Building user/item encoders…")
        user_ids = train_cf["reviewerID"].unique()
        item_ids = train_cf["asin"].unique()

        self.user_to_idx = {u: i for i, u in enumerate(user_ids)}
        self.item_to_idx = {it: i for i, it in enumerate(item_ids)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}

        rows = train_cf["reviewerID"].map(self.user_to_idx)
        cols = train_cf["asin"].map(self.item_to_idx)
        data = np.ones(len(train_cf))

        self._interaction_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(user_ids), len(item_ids)),
        )
        log.info(
            "Interaction matrix: %s  (%.4f%% dense)",
            self._interaction_matrix.shape,
            100 * self._interaction_matrix.nnz / np.prod(self._interaction_matrix.shape),
        )

        log.info("Training ALS (factors=%d, iter=%d)…", cfg.als.factors, cfg.als.iterations)
        self._model.fit(self._interaction_matrix)
        log.info("ALS training complete.")
        return self

    def recommend(self, user_id: str, n: int = 10) -> list[tuple[str, float]]:
        
        if user_id not in self.user_to_idx:
            return []

        user_idx = self.user_to_idx[user_id]
        ids, scores = self._model.recommend(
            user_idx,
            self._interaction_matrix[user_idx],
            N=n,
        )
        return [(self.idx_to_item[i], float(s)) for i, s in zip(ids, scores)]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Saved ALS model → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ALSModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info("Loaded ALS model from %s", path)
        return obj
