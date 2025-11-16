import torch
from typing import List, Dict


def recall_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Calculates Recall@K.
    :param predictions: Tensor of predicted item scores/rankings (batch_size, num_items)
    :param targets: Tensor of true item IDs (batch_size,)
    :param k: The 'K' for Recall@K
    :return: Recall@K score
    """
    _, top_k_indices = torch.topk(predictions, k, dim=1)  # Get top K item indices
    # Check if the target item is in the top K predictions
    # targets needs to be expanded to (batch_size, 1) for comparison
    targets_expanded = targets.unsqueeze(1)
    hits = (top_k_indices == targets_expanded).any(dim=1).float()
    return hits.mean().item()


def ndcg_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Calculates NDCG@K.
    :param predictions: Tensor of predicted item scores/rankings (batch_size, num_items)
    :param targets: Tensor of true item IDs (batch_size,)
    :param k: The 'K' for NDCG@K
    :return: NDCG@K score
    """
    batch_size, num_items = predictions.shape
    _, top_k_indices = torch.topk(predictions, k, dim=1)

    # Create a relevance matrix: 1 if target is in top-k, 0 otherwise
    relevance = torch.zeros_like(top_k_indices, dtype=torch.float)
    targets_expanded = targets.unsqueeze(1)
    for i in range(batch_size):
        # Find where the target item is in the top_k_indices for the current batch item
        target_pos = (top_k_indices[i] == targets_expanded[i]).nonzero(as_tuple=True)
        if len(target_pos[0]) > 0:
            # If target is found, set relevance at that position to 1
            relevance[i, target_pos[0][0]] = 1.0

    # Calculate DCG
    # Denominators for DCG: log2(position + 1)
    # positions are 0-indexed, so log2(1), log2(2), ..., log2(k)
    denominators = torch.log2(
        torch.arange(1, k + 1, dtype=torch.float, device=predictions.device) + 1
    )
    dcg = (relevance / denominators).sum(dim=1)

    # Calculate IDCG (Ideal DCG)
    # For a single relevant item, IDCG is always 1 / log2(1+1) = 1
    idcg = 1.0 / torch.log2(
        torch.tensor(2.0, device=predictions.device)
    )  # For the first position

    # Handle cases where target is not in top-k (dcg will be 0)
    ndcg = torch.where(
        dcg > 0, dcg / idcg, torch.tensor(0.0, device=predictions.device)
    )

    return ndcg.mean().item()


def hit_ratio_at_k(predictions: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    Calculates Hit Ratio@K. This is identical to Recall@K for a single target item.
    :param predictions: Tensor of predicted item scores/rankings (batch_size, num_items)
    :param targets: Tensor of true item IDs (batch_size,)
    :param k: The 'K' for Hit Ratio@K
    :return: Hit Ratio@K score
    """
    return recall_at_k(predictions, targets, k)


def evaluate_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, ks: List[int]
) -> Dict[str, float]:
    """
    Evaluates multiple recommendation metrics at different K values.
    :param predictions: Tensor of predicted item scores/rankings (batch_size, num_items)
    :param targets: Tensor of true item IDs (batch_size,)
    :param ks: List of K values to evaluate (e.g., [1, 5, 10, 20])
    :return: Dictionary of metric names to scores
    """
    results = {}
    for k in ks:
        results[f"Recall@{k}"] = recall_at_k(predictions, targets, k)
        results[f"NDCG@{k}"] = ndcg_at_k(predictions, targets, k)
        results[f"HitRatio@{k}"] = hit_ratio_at_k(
            predictions, targets, k
        )  # Redundant but kept for clarity
    return results


# Example usage
if __name__ == "__main__":
    # Dummy predictions and targets
    # Batch size = 3, num_items = 10
    # Predictions are scores for each item
    predictions = torch.tensor(
        [
            [0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6, 0.5, 0.0],  # Target 3 (score 0.9)
            [0.9, 0.1, 0.2, 0.3, 0.8, 0.4, 0.5, 0.6, 0.7, 0.0],  # Target 0 (score 0.9)
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Target 9 (score 0.9)
        ]
    )
    targets = torch.tensor([3, 0, 9])  # True item IDs

    ks_to_evaluate = [1, 5, 10]

    print("Evaluating metrics:")
    results = evaluate_metrics(predictions, targets, ks_to_evaluate)
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

    # Expected output for Recall@1:
    # Batch 1: Target 3 (score 0.9) is rank 1. Hit.
    # Batch 2: Target 0 (score 0.9) is rank 1. Hit.
    # Batch 3: Target 9 (score 0.9) is rank 1. Hit.
    # Recall@1 = 3/3 = 1.0

    # Expected output for NDCG@1:
    # Batch 1: Target 3 is rank 1. DCG = 1/log2(1+1) = 1. IDCG = 1. NDCG = 1.
    # Batch 2: Target 0 is rank 1. DCG = 1/log2(1+1) = 1. IDCG = 1. NDCG = 1.
    # Batch 3: Target 9 is rank 1. DCG = 1/log2(1+1) = 1. IDCG = 1. NDCG = 1.
    # NDCG@1 = 1.0

    # Expected output for Recall@5:
    # Batch 1: Target 3 (rank 1) is in top 5. Hit.
    # Batch 2: Target 0 (rank 1) is in top 5. Hit.
    # Batch 3: Target 9 (rank 1) is in top 5. Hit.
    # Recall@5 = 1.0
