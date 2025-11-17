import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from collections import Counter
import numpy as np

class RankingDistillationLoss(nn.Module):
    """
    ランキング蒸留損失を計算します。
    教師モデルのランキングスコアと生徒モデルのランキングスコアの差を最小化します。
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor): 生徒モデルの出力ロジット (batch_size, num_items)。
            teacher_logits (torch.Tensor): 教師モデルの出力ロジット (batch_size, num_items)。

        Returns:
            torch.Tensor: ランキング蒸留損失。
        """
        # ソフトターゲットの計算
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KLダイバージェンス損失
        loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2) # 温度スケーリングの補正

        return loss

class EmbeddingDistillationLoss(nn.Module):
    """
    埋め込み蒸留損失を計算します。
    教師モデルの埋め込みと生徒モデルの埋め込みの間の距離を最小化します。
    """
    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "cosine":
            self.loss_fn = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"Unsupported embedding distillation loss type: {loss_type}")
        self.loss_type = loss_type

    def forward(self, student_embeddings: torch.Tensor, teacher_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_embeddings (torch.Tensor): 生徒モデルの埋め込み (batch_size, embedding_dim)。
            teacher_embeddings (torch.Tensor): 教師モデルの埋め込み (batch_size, embedding_dim)。

        Returns:
            torch.Tensor: 埋め込み蒸留損失。
        """
        if self.loss_type == "cosine":
            # CosineEmbeddingLossはターゲットラベルを必要とする (1: 類似, -1: 非類似)
            # ここでは類似度を最大化したいので、ターゲットを1とする
            target = torch.ones(student_embeddings.size(0), device=student_embeddings.device)
            loss = self.loss_fn(student_embeddings, teacher_embeddings, target)
        else: # mse
            loss = self.loss_fn(student_embeddings, teacher_embeddings)
        return loss

class PropensityScoreCalculator:
    """
    アイテムの出現頻度に基づいて傾向スコア (propensity score) を計算します。
    """
    def __init__(self, item_num: int, train_next_items: List[int], power: float = 0.05):
        self.item_num = item_num
        self.power = power
        self.ps = self._calculate_ps(train_next_items)

    def _calculate_ps(self, train_next_items: List[int]) -> torch.Tensor:
        freq = Counter(train_next_items)
        pop = [freq[i] if i in freq else 0 for i in range(self.item_num)]
        pop = np.array(pop)
        ps = pop + 1
        ps = ps / np.sum(ps)
        ps = np.power(ps, self.power)
        return torch.tensor(ps, dtype=torch.float)

    def get_ps(self) -> torch.Tensor:
        return self.ps

class DROLoss(nn.Module):
    """
    DLLM2RecのDRO (Distributionally Robust Optimization) 損失を計算します。
    """
    def __init__(self, ps: torch.Tensor, beta: float = 1.0):
        super().__init__()
        self.ps = ps
        self.beta = beta

    def forward(self, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            model_output (torch.Tensor): 生徒モデルの出力ロジット (batch_size, num_items)。
            target (torch.Tensor): 正解アイテムのID (batch_size, 1)。

        Returns:
            torch.Tensor: DRO損失。
        """
        # psをmodel_outputと同じデバイスに移動
        ps_on_device = self.ps.to(model_output.device)

        # pos_scores_dro = torch.gather(torch.mul(model_output * model_output, ps), 1, target)
        # DLLM2Recのコードではmodel_output * model_outputとなっているが、これは誤りである可能性が高い。
        # 通常のBCEWithLogitsLossの文脈では、model_outputはロジットであり、
        # 損失計算にはsigmoid(model_output)が使われるか、直接ロジットが使われる。
        # ここでは、参照実装のコードをそのまま再現する。
        
        # model_outputはロジットなので、二乗する意味は不明だが、参照実装に従う
        # ただし、psはnum_itemsの次元を持つため、model_outputの各アイテムに対応させる
        weighted_model_output_sq = torch.mul(model_output * model_output, ps_on_device[1:])
        
        # expの引数をクランプしてinfを防ぐ
        clamped_weighted_model_output_sq_div_beta = torch.clamp(weighted_model_output_sq / self.beta, max=80.0)
        exp_term_sum = torch.sum(torch.exp(clamped_weighted_model_output_sq_div_beta), 1) # (batch_size,)

        pos_scores_dro = torch.gather(weighted_model_output_sq, 1, target)
        pos_scores_dro = torch.squeeze(pos_scores_dro) # (batch_size,)
        clamped_pos_scores_dro_div_beta = torch.clamp(pos_scores_dro / self.beta, max=80.0)

        weighted_model_output_minus_1_sq = torch.mul((model_output - 1) * (model_output - 1), ps_on_device[1:])
        pos_loss_dro = torch.gather(weighted_model_output_minus_1_sq, 1, target)
        pos_loss_dro = torch.squeeze(pos_loss_dro) # (batch_size,)
        clamped_pos_loss_dro_div_beta = torch.clamp(pos_loss_dro / self.beta, max=80.0)
        
        inner_dro = (exp_term_sum
                     - torch.exp(clamped_pos_scores_dro_div_beta)
                     + torch.exp(clamped_pos_loss_dro_div_beta))
        
        # Ensure inner_dro is positive before taking log to prevent NaN
        loss_dro = torch.log(torch.clamp(inner_dro, min=1e-6) + 1e-24) # (batch_size,)
        
        return torch.mean(loss_dro)

class WeightedBCELoss(nn.Module):
    """
    重み付きBCE損失を計算します。
    DLLM2Recのランキング蒸留におけるDRO損失を内部で計算し、BCE損失と結合します。
    """
    def __init__(self, alpha: float = 0.0, ps: Optional[torch.Tensor] = None, beta: float = 1.0):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.ps = ps
        self.beta = beta
        if self.alpha > 0 and self.ps is None:
            raise ValueError("Propensity scores (ps) must be provided if alpha > 0 for DROLoss.")

    def forward(self, student_logits: torch.Tensor, teacher_candidates: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_logits (torch.Tensor): 生徒モデルの出力ロジット (batch_size, num_items)。
            teacher_candidates (torch.Tensor): 教師モデルからの候補アイテム (batch_size, num_candidates)。
            weights (torch.Tensor): 各候補アイテムに対する重み (batch_size, num_candidates)。

        Returns:
            torch.Tensor: 重み付きBCE損失とDRO損失の結合。
        """
        total_ranking_loss = 0
        num_candidates = teacher_candidates.size(1)
        batch_size = student_logits.size(0)

        # negtive item sampling (DLLM2Rec main.pyから移植)
        # ここでは、ネガティブサンプリングはバッチ全体で一度行われることを想定
        # ただし、DLLM2Recのmain.pyでは、各候補アイテムに対してネガティブサンプリングが行われているように見える
        # ここでは、簡略化のため、各候補アイテムに対してネガティブサンプリングは行わず、
        # BCE損失の計算にはポジティブスコアのみを使用する。
        # もしネガティブサンプリングが必要な場合は、trainer_distill.py側で生成し、ここに渡す必要がある。
        # 参照実装のloss_bce_rdはpos_labelsとneg_labelsを使っているが、neg_scoresはどこから来るのか不明。
        # ここでは、pos_labelsとpos_scoresのみでBCEを計算する。
        # 参照実装のloss_bce_rd: -(pos_labels*torch.log(torch.sigmoid(pos_scores)) + (1-neg_labels)*torch.log(torch.sigmoid(1-neg_scores)))
        # これはBCEWithLogitsLoss(reduction='none')と等価ではない。
        # BCEWithLogitsLossは内部でsigmoidを適用し、log_sigmoid(x) - 2*x*target + x^2 のような計算をする。
        # 参照実装のloss_bce_rdは、ポジティブサンプルとネガティブサンプルを明示的に分けて計算している。
        # ここでは、ポジティブサンプルに対するBCE損失のみを計算する。

        for i in range(num_candidates):
            target = teacher_candidates[:, i:i+1] # (batch_size, 1)
            pos_scores = torch.gather(student_logits, 1, target) # (batch_size, 1)
            pos_labels = torch.ones_like(pos_scores) # (batch_size, 1)
            
            # BCE損失
            loss_bce_rd = self.bce_loss(pos_scores, pos_labels) # (batch_size, 1)

            current_ranking_loss = (weights[:, i:i+1] * loss_bce_rd).mean()

            # DRO損失
            if self.alpha > 0:
                ps_on_device = self.ps.to(student_logits.device)
                weighted_model_output_sq = torch.mul(student_logits * student_logits, ps_on_device[1:])
                
                # expの引数をクランプしてinfを防ぐ
                clamped_weighted_model_output_sq_div_beta = torch.clamp(weighted_model_output_sq / self.beta, max=80.0)
                exp_term_sum = torch.sum(torch.exp(clamped_weighted_model_output_sq_div_beta), 1) # (batch_size,)

                pos_scores_dro = torch.gather(weighted_model_output_sq, 1, target)
                pos_scores_dro = torch.squeeze(pos_scores_dro) # (batch_size,)
                clamped_pos_scores_dro_div_beta = torch.clamp(pos_scores_dro / self.beta, max=80.0)

                weighted_model_output_minus_1_sq = torch.mul((student_logits - 1) * (student_logits - 1), ps_on_device[1:])
                pos_loss_dro = torch.gather(weighted_model_output_minus_1_sq, 1, target)
                pos_loss_dro = torch.squeeze(pos_loss_dro) # (batch_size,)
                clamped_pos_loss_dro_div_beta = torch.clamp(pos_loss_dro / self.beta, max=80.0)
                
                inner_dro = (exp_term_sum
                             - torch.exp(clamped_pos_scores_dro_div_beta)
                             + torch.exp(clamped_pos_loss_dro_div_beta))
                
                # Ensure inner_dro is positive before taking log to prevent NaN
                loss_dro_rd = torch.log(torch.clamp(inner_dro, min=1e-6) + 1e-24) # (batch_size,)
                
                # DRO損失も重み付けして加算
                current_ranking_loss += self.alpha * (weights[:, i:i+1].squeeze(1) * loss_dro_rd).mean()
            
            total_ranking_loss += current_ranking_loss
            
        return total_ranking_loss
if __name__ == "__main__":
    # テスト用のダミーデータ
    batch_size = 4
    num_items = 100
    embedding_dim = 64
    item_num_total = 100 # PropensityScoreCalculator用
    num_candidates = 5

    student_logits = torch.randn(batch_size, num_items, requires_grad=True)
    teacher_logits = torch.randn(batch_size, num_items)
    target_items = torch.randint(0, num_items, (batch_size, 1))
    teacher_candidates = torch.randint(0, num_items, (batch_size, num_candidates))
    weights = torch.rand(batch_size, num_candidates)
    weights = weights / weights.sum(dim=1, keepdim=True) # 正規化

    student_embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    teacher_embeddings = torch.randn(batch_size, embedding_dim)

    # ランキング蒸留損失のテスト
    ranking_loss_fn = RankingDistillationLoss(temperature=2.0)
    ranking_loss = ranking_loss_fn(student_logits, teacher_logits)
    print(f"Ranking Distillation Loss: {ranking_loss.item()}")
    student_logits.grad = None # 勾配をリセット
    ranking_loss.backward() # 勾配計算の確認
    print(f"Student logits grad (ranking): {student_logits.grad.norm().item()}")

    # 埋め込み蒸留損失 (MSE) のテスト
    embedding_mse_loss_fn = EmbeddingDistillationLoss(loss_type="mse")
    embedding_mse_loss = embedding_mse_loss_fn(student_embeddings, teacher_embeddings)
    print(f"Embedding Distillation Loss (MSE): {embedding_mse_loss.item()}")
    student_embeddings.grad = None # 勾配をリセット
    embedding_mse_loss.backward()
    print(f"Student embeddings grad (MSE): {student_embeddings.grad.norm().item()}")

    # 埋め込み蒸留損失 (Cosine) のテスト
    embedding_cosine_loss_fn = EmbeddingDistillationLoss(loss_type="cosine")
    embedding_cosine_loss = embedding_cosine_loss_fn(student_embeddings, teacher_embeddings)
    print(f"Embedding Distillation Loss (Cosine): {embedding_cosine_loss.item()}")
    student_embeddings.grad = None # 勾配をリセット
    embedding_cosine_loss.backward()
    print(f"Student embeddings grad (Cosine): {student_embeddings.grad.norm().item()}")

    # PropensityScoreCalculatorとDROLossのテスト
    # ダミーのtrain_next_itemsを生成
    dummy_train_next_items = torch.randint(0, item_num_total, (1000,)).tolist()
    ps_calculator = PropensityScoreCalculator(item_num=item_num_total, train_next_items=dummy_train_next_items)
    ps_scores = ps_calculator.get_ps()
    print(f"Propensity Scores shape: {ps_scores.shape}")
    print(f"Propensity Scores sum: {ps_scores.sum()}")

    dro_loss_fn = DROLoss(ps=ps_scores, beta=1.0)
    dro_loss = dro_loss_fn(student_logits, target_items)
    print(f"DRO Loss: {dro_loss.item()}")
    student_logits.grad = None # 勾配をリセット
    dro_loss.backward()
    print(f"Student logits grad (DRO): {student_logits.grad.norm().item()}")

    print("\nKnowledge Distillation Losses test passed!")

    print("\nWeightedBCELoss with DRO test:")
    # alpha=0 (no DRO)
    weighted_bce_loss_no_dro_fn = WeightedBCELoss(alpha=0.0)
    loss_no_dro = weighted_bce_loss_no_dro_fn(student_logits, teacher_candidates, weights)
    print(f"WeightedBCELoss (no DRO): {loss_no_dro.item()}")
    student_logits.grad = None
    loss_no_dro.backward()
    print(f"Student logits grad (WeightedBCELoss no DRO): {student_logits.grad.norm().item()}")

    # alpha > 0 (with DRO)
    weighted_bce_loss_with_dro_fn = WeightedBCELoss(alpha=0.5, ps=ps_scores, beta=1.0)
    loss_with_dro = weighted_bce_loss_with_dro_fn(student_logits, teacher_candidates, weights)
    print(f"WeightedBCELoss (with DRO): {loss_with_dro.item()}")
    student_logits.grad = None
    loss_with_dro.backward()
    print(f"Student logits grad (WeightedBCELoss with DRO): {student_logits.grad.norm().item()}")
