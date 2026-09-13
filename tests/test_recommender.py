import pytest
import pandas as pd
from src.inference import HybridRecommender

@pytest.fixture(scope="module")
def recommender():
    
    return HybridRecommender()

def test_cold_start_user(recommender):
    
    user_id = "NON_EXISTENT_USER_123"
    recs = recommender.recommend(user_id, top_k=5)
    
    assert len(recs) == 5
    assert "asin" in recs.columns
    assert "hybrid_score" in recs.columns

    assert recs["hybrid_score"].iloc[0] is None

def test_known_user(recommender):
    

    user_id = list(recommender._user_history.keys())[0]
    recs = recommender.recommend(user_id, top_k=5)
    
    assert len(recs) == 5
    assert "hybrid_score" in recs.columns

    assert pd.notna(recs["hybrid_score"].iloc[0])
    
def test_seen_items_excluded(recommender):
    
    user_id = list(recommender._user_history.keys())[0]
    seen_items = recommender._user_history[user_id]
    

    recs = recommender.recommend(user_id, top_k=20)
    recommended_asins = set(recs["asin"].tolist())
    

    assert len(recommended_asins.intersection(seen_items)) == 0
