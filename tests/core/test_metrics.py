import pytest
from src.core.metrics import calculate_metrics

def test_calculate_metrics_perfect_match():
    """
    予測が正解と完全に一致する場合のテスト。
    """
    predictions = [[1, 2, 3], [5, 4, 6]]
    ground_truths = [[1], [5]]
    k = 3
    metrics = calculate_metrics(predictions, ground_truths, k)
    assert metrics[f"recall@{k}"] == 1.0
    assert metrics[f"ndcg@{k}"] == 1.0
    assert metrics[f"hit_ratio@{k}"] == 1.0

def test_calculate_metrics_no_match():
    """
    予測が正解と全く一致しない場合のテスト。
    """
    predictions = [[1, 2, 3], [4, 5, 6]]
    ground_truths = [[10], [20]]
    k = 3
    metrics = calculate_metrics(predictions, ground_truths, k)
    assert metrics[f"recall@{k}"] == 0.0
    assert metrics[f"ndcg@{k}"] == 0.0
    assert metrics[f"hit_ratio@{k}"] == 0.0

def test_calculate_metrics_partial_match():
    """
    予測が部分的に一致する場合のテスト。
    """
    predictions = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14]]
    ground_truths = [[3, 6], [15, 11]]
    k = 5
    metrics = calculate_metrics(predictions, ground_truths, k)
    
    # User 1: recall = 1/2 = 0.5
    # User 2: recall = 1/2 = 0.5
    # Total recall = (0.5 + 0.5) / 2 = 0.5
    assert pytest.approx(metrics[f"recall@{k}"], 0.001) == 0.5

    # User 1: hit = 1
    # User 2: hit = 1
    # Total hit_ratio = (1 + 1) / 2 = 1.0
    assert metrics[f"hit_ratio@{k}"] == 1.0

    # User 1: dcg = 1/log2(3+1) = 0.5
    # User 2: dcg = 1/log2(2+1) = 1/log2(3)
    # idcg1 = 1/log2(1+1) + 1/log2(2+1) = 1 + 1/log2(3)
    # idcg2 = 1/log2(1+1) + 1/log2(2+1) = 1 + 1/log2(3)
    # ndcg1 = 0.5 / (1 + 1/log2(3))
    # ndcg2 = (1/log2(3)) / (1 + 1/log2(3))
    # total_ndcg = (ndcg1 + ndcg2) / 2
    import numpy as np
    log2_3 = np.log2(3)
    log2_4 = np.log2(4)
    dcg1 = 1 / log2_4
    dcg2 = 1 / log2_3
    idcg = 1 / np.log2(2) + 1 / np.log2(3)
    ndcg1 = dcg1 / idcg
    ndcg2 = dcg2 / idcg
    expected_ndcg = (ndcg1 + ndcg2) / 2
    assert pytest.approx(metrics[f"ndcg@{k}"], 0.001) == expected_ndcg

def test_calculate_metrics_empty_input():
    """
    入力が空の場合のテスト。
    """
    predictions = []
    ground_truths = []
    k = 5
    metrics = calculate_metrics(predictions, ground_truths, k)
    assert metrics[f"recall@{k}"] == 0.0
    assert metrics[f"ndcg@{k}"] == 0.0
    assert metrics[f"hit_ratio@{k}"] == 0.0

def test_calculate_metrics_k_value():
    """
    kの値がメトリクスに影響を与えるかのテスト。
    """
    predictions = [[1, 2, 3, 4, 5]]
    ground_truths = [[5]]
    
    # k=3ではヒットしない
    metrics_k3 = calculate_metrics(predictions, ground_truths, 3)
    assert metrics_k3["hit_ratio@3"] == 0.0
    
    # k=5ではヒットする
    metrics_k5 = calculate_metrics(predictions, ground_truths, 5)
    assert metrics_k5["hit_ratio@5"] == 1.0
