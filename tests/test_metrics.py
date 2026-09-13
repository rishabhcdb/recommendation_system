import pytest
from src.evaluate import precision_at_k, recall_at_k, ndcg_at_k

def test_precision_at_k():
    recommended = ["a", "b", "c", "d"]
    relevant = {"b", "d", "e"}
    

    assert precision_at_k(recommended, relevant, k=4) == 0.5
    

    assert precision_at_k(recommended, relevant, k=2) == 0.5

    assert precision_at_k(recommended, relevant, k=1) == 0.0

def test_recall_at_k():
    recommended = ["a", "b", "c", "d"]
    relevant = {"b", "d", "e"}
    

    assert recall_at_k(recommended, relevant, k=4) == 2.0 / 3.0
    

    assert recall_at_k(recommended, set(), k=4) == 0.0

def test_ndcg_at_k():

    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0
    
    # Completely wrong ranking
    assert ndcg_at_k(["c", "d", "e"], {"a", "b"}, k=3) == 0.0

    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0
