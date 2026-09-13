import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import cfg
from src.features import build_catalog, build_embeddings
from src.build_index import build_faiss_index
from src.models.als import ALSModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# splits

def make_splits(interactions_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    log.info("Loading interactions from %s", interactions_path)
    df = pd.read_parquet(interactions_path)
    df = df.sort_values(["reviewerID", "unixReviewTime"])

    test  = df.groupby("reviewerID").tail(1)
    train = df.drop(test.index)

    threshold = cfg.features.positive_rating_threshold
    train_cf  = train[train["overall"] >= threshold].copy()
    train_cf["interaction"] = np.int8(1)

    log.info(
        "Split → train=%d  test=%d  train_cf=%d  (threshold≥%d)",
        len(train), len(test), len(train_cf), threshold,
    )
    return train, test, train_cf

def run(skip_features: bool = False, skip_index: bool = False) -> None:
    if skip_features:
        log.info("--skip-features: skipping catalog + embedding generation.")
    else:
        catalog = build_catalog()
        build_embeddings(catalog)

    if skip_index:
        log.info("--skip-index: skipping FAISS index build.")
    else:
        build_faiss_index()

    interactions_path = ROOT / cfg.data.interactions
    train, test, train_cf = make_splits(interactions_path)

    train.to_parquet(ROOT / cfg.data.train, index=False)
    test.to_parquet(ROOT / cfg.data.test, index=False)
    train_cf.to_parquet(ROOT / cfg.data.train_cf, index=False)
    log.info("Saved train / test / train_cf parquet files.")

    als_model = ALSModel()
    als_model.fit(train_cf)
    als_model.save(ROOT / cfg.data.als_model)

    log.info("Running offline evaluation…")
    from src.evaluate import run_evaluation
    run_evaluation()

    log.info("Training pipeline complete. All artifacts saved to data/.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the hybrid recommender pipeline.")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip catalog building and embedding generation.")
    parser.add_argument("--skip-index",    action="store_true",
                        help="Skip FAISS index building.")
    args = parser.parse_args()

    run(skip_features=args.skip_features, skip_index=args.skip_index)
