from typing import List, Dict
import numpy as np

def calculate_metrics(
    predictions: List[List[int]],
    ground_truths: List[List[int]],
    k: int
) -> Dict[str, float]:
    """
    推薦リストの評価指標（Recall@K, NDCG@K, HitRatio@K）を計算します。

    Args:
        predictions (List[List[int]]): 各ユーザーの推薦アイテムIDのリストのリスト。
        ground_truths (List[List[int]]): 各ユーザーの正解アイテムIDのリストのリスト。
        k (int): 評価対象とする推薦リストのトップK。

    Returns:
        Dict[str, float]: 計算された評価指標を含む辞書。
    """
    if not predictions or not ground_truths:
        return {f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, f"hit_ratio@{k}": 0.0}

    total_recall = 0.0
    total_ndcg = 0.0
    total_hit = 0.0
    num_users = len(predictions)

    for user_preds, user_truths in zip(predictions, ground_truths):
        # トップKの予測
        top_k_preds = user_preds[:k]

        # HitRatio@K
        hit = any(item in user_truths for item in top_k_preds)
        total_hit += 1 if hit else 0

        # Recall@K
        if user_truths:
            num_hits = len(set(top_k_preds) & set(user_truths))
            total_recall += num_hits / len(user_truths)

        # NDCG@K
        dcg = 0.0
        idcg = 0.0
        for i, pred_item in enumerate(top_k_preds):
            if pred_item in user_truths:
                dcg += 1.0 / np.log2(i + 2) # +2 because log2(1) is 0
        
        # IDCG (理想的なDCG) は、正解アイテムを関連度順に並べた場合のDCG
        # ここでは関連度を1として、正解アイテムの数だけ計算
        for i in range(min(len(user_truths), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg > 0:
            total_ndcg += dcg / idcg

    return {
        f"recall@{k}": total_recall / num_users,
        f"ndcg@{k}": total_ndcg / num_users,
        f"hit_ratio@{k}": total_hit / num_users,
    }

if __name__ == "__main__":
    # テストデータ
    predictions_1 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
    ground_truths_1 = [[1, 100], [7, 200], [13, 300]]
    k_1 = 3
    metrics_1 = calculate_metrics(predictions_1, ground_truths_1, k_1)
    print(f"Test 1 (k={k_1}): {metrics_1}")
    # 期待値: recall@3: (1/2 + 1/2 + 1/2)/3 = 0.5, hit_ratio@3: 1.0, ndcg@3: (1/log2(2) + 1/log2(2) + 1/log2(2))/3 = 1.0

    predictions_2 = [[1, 2, 3], [4, 5, 6]]
    ground_truths_2 = [[1, 4], [5, 7]]
    k_2 = 1
    metrics_2 = calculate_metrics(predictions_2, ground_truths_2, k_2)
    print(f"Test 2 (k={k_2}): {metrics_2}")
    # 期待値: recall@1: (1/2 + 1/2)/2 = 0.5, hit_ratio@1: 1.0, ndcg@1: (1/log2(2) + 1/log2(2))/2 = 1.0

    predictions_3 = [[1, 2, 3]]
    ground_truths_3 = [[4, 5, 6]]
    k_3 = 3
    metrics_3 = calculate_metrics(predictions_3, ground_truths_3, k_3)
    print(f"Test 3 (k={k_3}): {metrics_3}")
    # 期待値: recall@3: 0.0, hit_ratio@3: 0.0, ndcg@3: 0.0

    predictions_4 = []
    ground_truths_4 = []
    k_4 = 5
    metrics_4 = calculate_metrics(predictions_4, ground_truths_4, k_4)
    print(f"Test 4 (empty lists, k={k_4}): {metrics_4}")
    # 期待値: recall@5: 0.0, hit_ratio@5: 0.0, ndcg@5: 0.0
