import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

def _flatten_categories(x) -> str:
    
    try:
        return " ".join(str(item) for sublist in x for item in sublist)
    except Exception:
        return ""

def build_catalog() -> pd.DataFrame:
    
    meta_path = ROOT / cfg.data.metadata
    log.info("Loading metadata from %s", meta_path)
    meta = pd.read_parquet(meta_path)
    log.info("Raw metadata shape: %s", meta.shape)

    meta_clean = meta.copy()

    for col in ("title", "description", "brand"):
        if col in meta_clean.columns:
            meta_clean[col] = meta_clean[col].fillna("")

    if "categories" in meta_clean.columns:
        meta_clean["category_text"] = meta_clean["categories"].apply(_flatten_categories)
    else:
        meta_clean["category_text"] = ""

    meta_clean["product_text"] = (
        meta_clean.get("title", "")
        + " "
        + meta_clean.get("brand", "")
        + " "
        + meta_clean["category_text"]
        + " "
        + meta_clean.get("description", "")
    )

    keep_cols = [c for c in ["asin", "title", "brand", "category_text", "description", "product_text"]
                 if c in meta_clean.columns]
    catalog = meta_clean[keep_cols].copy()

    out_path = ROOT / cfg.data.catalog
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out_path, index=False)
    log.info("Saved catalog → %s  (%d rows)", out_path, len(catalog))

    return catalog

def build_embeddings(catalog: pd.DataFrame | None = None) -> np.ndarray:
    
    if catalog is None:
        catalog_path = ROOT / cfg.data.catalog
        log.info("Loading catalog from %s", catalog_path)
        catalog = pd.read_parquet(catalog_path)

    model_name = cfg.features.embedding_model
    max_chars   = cfg.features.max_text_chars
    batch_size  = cfg.features.embedding_batch_size

    log.info("Loading SentenceTransformer: %s", model_name)
    model = SentenceTransformer(model_name)

    texts = [str(t)[:max_chars] for t in catalog["product_text"].tolist()]
    log.info("Encoding %d products (batch_size=%d)…", len(texts), batch_size)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    out_path = ROOT / cfg.data.embeddings
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    log.info("Saved embeddings → %s  shape=%s", out_path, embeddings.shape)

    return embeddings

if __name__ == "__main__":
    catalog = build_catalog()
    build_embeddings(catalog)
