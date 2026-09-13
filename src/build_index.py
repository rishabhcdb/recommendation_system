import logging
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

def build_faiss_index() -> faiss.Index:
    
    emb_path = ROOT / cfg.data.embeddings
    log.info("Loading embeddings from %s", emb_path)
    embeddings = np.load(emb_path).astype("float32")
    log.info("Embeddings shape: %s", embeddings.shape)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    log.info("Indexed %d vectors (dim=%d)", index.ntotal, dimension)

    out_path = ROOT / cfg.data.faiss_index
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    log.info("Saved FAISS index → %s", out_path)

    return index

def _smoke_test(index: faiss.Index, embeddings: np.ndarray, catalog: pd.DataFrame) -> None:
    
    query = embeddings[0].reshape(1, -1)
    start = time.perf_counter()
    scores, indices = index.search(query, 11)
    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info("Smoke-test latency: %.2f ms", elapsed_ms)
    neighbours = catalog.iloc[indices[0][1:]][["asin", "title"]].head(5)
    log.info("Top-5 neighbours of %s:\n%s", catalog.iloc[0]["asin"], neighbours.to_string(index=False))

if __name__ == "__main__":
    idx = build_faiss_index()

    catalog_path = ROOT / cfg.data.catalog
    emb_path     = ROOT / cfg.data.embeddings
    catalog   = pd.read_parquet(catalog_path)
    embeddings = np.load(emb_path).astype("float32")

    _smoke_test(idx, embeddings, catalog)
