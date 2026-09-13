import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import cfg
from src.inference import HybridRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

recommender: Optional[HybridRecommender] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    log.info("Loading HybridRecommender…")
    recommender = HybridRecommender()
    log.info("HybridRecommender ready. API is live.")
    yield
    log.info("Shutting down.")

app = FastAPI(
    title="Hybrid Recommender API",
    description=(
        "Two-stage retrieval and ranking recommender system. "
        "Stage 1: ALS collaborative filtering. "
        "Stage 2: FAISS content expansion. "
        "Stage 3: Weighted blend (ALS + content + popularity)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

class RecommendedItem(BaseModel):
    asin:         str
    title:        str
    hybrid_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    user_id:         str
    top_k:           int
    recommendations: list[RecommendedItem]

class ExplainResponse(BaseModel):
    user_id:     str
    asin:        str
    explanation: list[str]

@app.get("/health")
def health():
    
    return {"status": "ok"}

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: str, top_k: int = cfg.api.top_k):
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    df = recommender.recommend(user_id, top_k=top_k)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No recommendations for user '{user_id}'.")

    items = []
    for _, row in df.iterrows():
        score = row.get("hybrid_score") if hasattr(row, "get") else row["hybrid_score"]
        # JSON compat
        if score is not None:
            import math
            try:
                if math.isnan(float(score)):
                    score = None
            except (TypeError, ValueError):
                score = None
        items.append(RecommendedItem(asin=row["asin"], title=row["title"], hybrid_score=score))

    return RecommendationResponse(user_id=user_id, top_k=top_k, recommendations=items)

@app.get("/explain/{user_id}/{asin}", response_model=ExplainResponse)
def explain(user_id: str, asin: str):
    
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    lines = recommender.explain(user_id, asin)
    return ExplainResponse(user_id=user_id, asin=asin, explanation=lines)
