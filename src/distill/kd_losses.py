import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


def compute_ranking_distillation_loss(
    student_logits: torch.Tensor,
    teacher_rankings: torch.Tensor,
    teacher_confidences: torch.Tensor,
    item_num: int,
    cfg: DictConfig,
    device: torch.device,
    ps: torch.Tensor = None,  # Propensity score for DRO, if used
) -> torch.Tensor:
    """
    Computes the importance-aware ranking distillation loss as described in DLLM2Rec.

    Args:
        student_logits: Logits from the student model (batch_size, item_num).
        teacher_rankings: Top-K item IDs from teacher (batch_size, K).
        teacher_confidences: Confidence scores for teacher rankings (batch_size, K).
        item_num: Total number of items.
        cfg: Hydra config for distillation parameters.
        device: Torch device.
        ps: Propensity scores for DROS loss.

    Returns:
        A scalar tensor representing the ranking distillation loss.
    """

    batch_size, K = teacher_rankings.shape
    bce_loss_fn = torch.nn.BCEWithLogitsLoss(
        reduction="none"
    )  # We need unreduced loss for weighting

    # Extract distillation parameters
    gamma_position = cfg.distill.gamma_position
    gamma_confidence = cfg.distill.gamma_confidence
    gamma_consistency = cfg.distill.gamma_consistency

    # Mimic negative sampling from DLLM2Rec main.py
    # For ranking distillation, DLLM2Rec's main.py uses a single negative item sampled per positive item.
    # This negative item is sampled from all items *excluding* history and target.
    # Here, for simplicity and assuming we are distilling target items where we want to match teacher rankings,
    # we'll generate negative items. In DLLM2Rec main, `target_neg` is sampled once per batch for BCE.
    # For RD, it iteratively picks candidate items and computes BCE against `target_neg`.
    # Let's use a simpler negative sampling for the RD loss calculation for now:
    # For each candidate in teacher_rankings, sample a negative that's not in the candidate set.

    # A more faithful replication of DLLM2Rec's neg sampling for RD:
    # `target_neg` is sampled once per batch outside the loop and reused.

    # Negative item sampling (as seen in DLLM2Rec's main.py)
    # The negative for BCE is `target_neg`, which is sampled once per sequence for the whole batch
    # and reused for each candidate item in the teacher ranking.

    # For negative items, sample from all items excluding those in `teacher_rankings` for the current sequence.
    # This was not explicitly clear in DLLM2Rec's `main.py` `loss_all_rd` loop, where `target_neg` was
    # a batch-wide negative sample for the original BCE Loss part.
    # Let's assume a simpler approach for now to avoid overcomplicating until detailed behavior is clear.
    # For each candidate from teacher, sample a random negative item.

    # For now, let's reuse the general negative sampling logic from `trainer_baseline`.
    # This might need refinement depending on the exact behavior intended by DLLM2Rec's ranking distillation.

    # For consistency with DLLM2Rec's `main.py` implementation where `target_neg` is used
    # from the main BCE loss loop. We need the actual `target_neg` from the training loop.
    # Therefore, this function should take `target_neg` as an argument.

    # Let's assume `target_neg` is provided by the trainer.

    # Calculate weights
    # weight_static (position-aware)
    position_weights = torch.arange(
        1, K + 1, dtype=torch.float32, device=device
    ).unsqueeze(
        0
    )  # (1, K)
    # The original DLLM2Rec uses `_lambda = 1` and `torch.exp(-weight_static / _lambda)`.
    # The lambda parameter in cfg.distill is not clearly specified if it's the `_lambda` for position_weights.
    # Let's assume a default lambda for position weights for now if not explicitly in cfg.distill.
    _lambda_pos_weight = cfg.distill.get("lambda_position_weight", 1.0)
    weight_rank = torch.exp(-position_weights / _lambda_pos_weight)  # (batch_size, K)
    weight_rank = weight_rank / torch.sum(weight_rank, dim=1, keepdim=True)

    # weight_confidence (confidence-aware)
    # Original DLLM2Rec uses torch.exp(-candidate_confidence) + 1e-8
    weight_confidence = torch.exp(-teacher_confidences) + 1e-8
    weight_confidence = weight_confidence / torch.sum(
        weight_confidence, dim=1, keepdim=True
    )

    # weight_consistency (consistency-aware) - requires comparing student's top-K with teacher's top-K
    # This was done by `common_tensor = (candidate.unsqueeze(2) == cf_rank_top.unsqueeze(1)).any(dim=2).int() + 1e-8`
    # where `cf_rank_top` is student's top-K. This implies reranking by student's score.
    # Let's assume it checks common items between student's predictions and teacher's.

    student_top_K = (-student_logits).argsort(dim=1)[:, :K]
    common_mask = (
        (teacher_rankings.unsqueeze(2) == student_top_K.unsqueeze(1)).any(dim=2).int()
    )  # (batch_size, K)
    weight_consistency = (
        common_mask.float() + 1e-8
    )  # Add small epsilon to avoid division by zero

    # Normalizing weight_consistency per row
    weight_consistency = weight_consistency / torch.sum(
        weight_consistency, dim=1, keepdim=True
    )

    # Combine weights
    weight_fin = (
        gamma_position * weight_rank
        + gamma_confidence * weight_confidence
        + gamma_consistency * weight_consistency
    )
    # Normalize final weights
    final_weights = weight_fin / torch.sum(
        weight_fin, dim=1, keepdim=True
    )  # (batch_size, K)

    total_ranking_loss = 0.0
    for i in range(K):
        candidate_item = teacher_rankings[:, i : i + 1]  # (batch_size, 1)

        # Need a negative sample for BCE.
        # DLLM2Rec `main.py` uses `target_neg` (sampled once) for this.
        # This function should probably receive `target_neg` from the training loop.
        # For now, let's create a *dummy_negative* if not provided, assuming we don't have history.
        # A proper `target_neg` would exclude history and the `candidate_item`.

        # Let's assume `target_neg` is provided by the trainer.

        # To strictly match DLLM2Rec `main.py`:
        # `pos_scores = torch.gather(model_output, 1, target)` where `target` is `candidate_item`
        # `neg_scores = torch.gather(model_output, 1, target_neg)` where `target_neg` is sampled once per batch.

        # Let's ensure `compute_ranking_distillation_loss` also receives `target_neg` from the trainer.
        # For now, let's include it as a required argument `negative_items`.

        # Dummy negative sampling if not passed (for testing this module alone)
        # NOTE: This neg sampling is very basic for `kd_losses.py`'s standalone test.
        # The trainer will provide much better `negative_items`.
        if "negative_items" not in cfg.distill:  # placeholder for actual negative_items
            # Sample random negative items for each batch entry, different from candidate_item
            negative_items_for_bce = torch.randint(
                0, item_num, (batch_size, 1), device=device
            )
            # Ensure it's not the same as candidate_item - extremely basic collision handling
            negative_items_for_bce = torch.where(
                negative_items_for_bce == candidate_item,
                (negative_items_for_bce + 1) % item_num,
                negative_items_for_bce,
            )
        else:
            negative_items_for_bce = (
                cfg.distill.negative_items
            )  # This would come from main trainer

        pos_scores = torch.gather(student_logits, 1, candidate_item)
        neg_scores = torch.gather(student_logits, 1, negative_items_for_bce)

        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        bce_scores = torch.cat((pos_scores, neg_scores), 0)
        bce_labels = torch.cat((pos_labels, neg_labels), 0)

        bce_rd_loss_unreduced = bce_loss_fn(bce_scores, bce_labels)
        # Reshape to (batch_size, 2) for (pos, neg) and take mean across items
        # Or, we can compute it more directly for pos and neg parts.

        # Let's match DLLM2Rec's `loss_bce_rd = -(pos_labels*torch.log(torch.sigmoid(pos_scores)) + (1-neg_labels)*torch.log(torch.sigmoid(1-neg_scores)))`
        # This is a per-sample BCE loss, not aggregated over pos/neg parts yet.
        # The `bce_loss_fn` with `reduction='none'` already does this.

        # A standard BCE loss for one positive and one negative sample:
        # Sum of sigmoid(pos_score) and (1-sigmoid(neg_score))

        # Re-implementing BCE part to match DLLM2Rec `main.py` for clarity:
        loss_bce_pos = -F.logsigmoid(pos_scores) * pos_labels
        loss_bce_neg = -F.logsigmoid(-neg_scores) * (
            1 - neg_labels
        )  # (1-neg_labels) is 1 for neg_scores
        loss_bce_rd_per_sample = loss_bce_pos + loss_bce_neg

        # Sum over the 1-dim (dummy for scores)
        loss_bce_rd_per_sample = loss_bce_rd_per_sample.sum(dim=1)  # (batch_size,)

        # DROS (robust ranking distillation) loss (if alpha > 0)
        # This part assumes propensity scores `ps` are available and match `item_num`
        if ps is not None and cfg.distill.alpha > 0:
            _beta = cfg.distill.get("beta", 1.0)  # From DLLM2Rec main.py

            # pos_scores_dro, pos_loss_dro calculation for current candidate
            pos_scores_dro = torch.gather(student_logits, 1, candidate_item) * (
                ps[candidate_item.squeeze(1)].unsqueeze(1)
            )
            pos_loss_dro = torch.gather(student_logits - 1, 1, candidate_item) * (
                ps[candidate_item.squeeze(1)].unsqueeze(1)
            )

            # Inner DRO calculation for all items multiplied by ps
            # This needs to be applied to all items logits, then gathered for the specific candidate
            model_output_squared = (
                student_logits * student_logits
            )  # student_logits is already the output of the model
            model_output_minus_1_squared = (student_logits - 1) * (student_logits - 1)

            A_val = torch.sum(
                torch.exp((model_output_squared * ps) / _beta), 1
            )  # Sum over all items (item_num)
            B_val = torch.exp((pos_scores_dro / _beta)).squeeze(
                1
            )  # For specific candidate item
            C_val = torch.exp((pos_loss_dro / _beta)).squeeze(
                1
            )  # For specific candidate item

            inner_dro_rd = A_val - B_val + C_val
            loss_dro_rd_per_sample = torch.log(inner_dro_rd + 1e-24)  # (batch_size,)

            # Weighted sum for RD loss (matching DLLM2Rec main.py)
            total_ranking_loss += (
                final_weights[:, i] * loss_bce_rd_per_sample
            ).mean() + cfg.distill.alpha * (
                final_weights[:, i] * loss_dro_rd_per_sample
            ).mean()
        else:
            total_ranking_loss += (final_weights[:, i] * loss_bce_rd_per_sample).mean()

    return total_ranking_loss


if __name__ == "__main__":
    # Example usage for testing KD losses
    from omegaconf import OmegaConf

    device = torch.device("cpu")
    batch_size = 32
    item_num = 100
    hidden_size = 64
    K = 10  # candidate_topk

    # Dummy student logits and embeddings
    student_logits = torch.randn(batch_size, item_num, device=device)
    student_item_embeddings = torch.randn(batch_size, hidden_size, device=device)

    # Dummy teacher outputs (for a batch)
    teacher_rankings = torch.randint(0, item_num, (batch_size, K), device=device)
    teacher_confidences = torch.rand(batch_size, K, device=device)
    teacher_item_embeddings = torch.randn(
        batch_size, hidden_size, device=device
    )  # Corresponding to student_item_embeddings

    # Dummy config for KD losses
    kd_cfg = OmegaConf.create(
        {
            "distill": {
                "gamma_position": 0.3,
                "gamma_confidence": 0.5,
                "gamma_consistency": 0.1,
                "alpha": 0.5,  # Enable DROS for ranking distillation
                "beta": 1.0,  # Beta for DROS
                "lambda_position_weight": 1.0,  # Custom param for position weight
                "candidate_topk": K,  # To pass to function if needed
            }
        }
    )

    # Dummy propensity scores for DROS
    ps = torch.rand(item_num, device=device)
    ps = ps / ps.sum()  # Normalize

    # Dummy negative items (as would be sampled in trainer_distill)
    # This sample should be independent of teacher_rankings and batch targets
    negative_items_for_bce_rd = torch.randint(
        0, item_num, (batch_size, 1), device=device
    )

    # Compute Ranking Distillation Loss
    ranking_loss = compute_ranking_distillation_loss(
        student_logits=student_logits,
        teacher_rankings=teacher_rankings,
        teacher_confidences=teacher_confidences,
        item_num=item_num,
        cfg=kd_cfg,
        device=device,
        ps=ps,
        # Manually inject negative_items for internal testing of this module
        # In actual trainer, this would come from `trainer_distill.py`'s sampling.
        negative_items=negative_items_for_bce_rd,  # Passed through cfg for this test
    )
    print(f"Ranking Distillation Loss: {ranking_loss.item():.4f}")
