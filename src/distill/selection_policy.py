import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class SelectionPolicy(ABC):
    """
    蒸留にどのサンプルを使用するかを決定するポリシーの抽象基底クラス。
    """
    @abstractmethod
    def select(self, 
               student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               ground_truth: torch.Tensor) -> torch.Tensor:
        """
        与えられた教師モデルと生徒モデルの出力、および正解データに基づいて、
        蒸留に使用するサンプルを選択するためのブールマスクを返します。

        Args:
            student_logits (torch.Tensor): 生徒モデルの出力ロジット。
            teacher_logits (torch.Tensor): 教師モデルの出力ロジット。
            ground_truth (torch.Tensor): 正解アイテムのインデックス。

        Returns:
            torch.Tensor: 蒸留に使用するサンプルを示すブールマスク (batch_size,)
        """
        pass

class AllSamplesPolicy(SelectionPolicy):
    """
    すべてのサンプルを蒸留に使用するポリシー。
    """
    def select(self, 
               student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               ground_truth: torch.Tensor) -> torch.Tensor:
        """
        すべてのサンプルを選択するためのブールマスクを返します。
        """
        return torch.ones(student_logits.shape[0], dtype=torch.bool, device=student_logits.device)

class KLDivergencePolicy(SelectionPolicy):
    """
    教師モデルと生徒モデルの出力間のKLダイバージェンスが閾値を超えるサンプルのみを
    蒸留に使用するポリシー。
    """
    def __init__(self, kl_threshold: float = 0.1):
        if not (kl_threshold >= 0.0):
            raise ValueError("kl_threshold must be non-negative.")
        self.kl_threshold = kl_threshold

    def select(self, 
               student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               ground_truth: torch.Tensor) -> torch.Tensor:
        """
        教師モデルと生徒モデルの出力間のKLダイバージェンスを計算し、
        それがkl_thresholdを超えるサンプルを選択するためのブールマスクを返します。
        """
        # 生徒モデルのロジットにlog_softmaxを適用
        log_probs_student = F.log_softmax(student_logits, dim=-1)
        # 教師モデルのロジットにlog_softmaxを適用（ターゲットも対数確率分布）
        log_probs_teacher = F.log_softmax(teacher_logits, dim=-1)

        # KLダイバージェンスを計算 (reduction='none'で各サンプルのKLを計算)
        # log_target=Trueなので、ターゲットも対数確率であるべき
        kl_div_per_sample = F.kl_div(log_probs_student, log_probs_teacher, reduction='none', log_target=True).sum(dim=-1) # (batch_size,)

        # KLダイバージェンスが閾値を超えるサンプルを選択
        return kl_div_per_sample >= self.kl_threshold

class GroundTruthErrorPolicy(SelectionPolicy):
    """
    生徒モデルが正解アイテムを正しく予測できないサンプルを選択するポリシー。
    具体的には、正解アイテムに対する生徒モデルのロジットが閾値を下回る場合に選択します。
    """
    def __init__(self, logit_threshold: float = -5.0):
        self.logit_threshold = logit_threshold

    def select(self,
               student_logits: torch.Tensor,
               teacher_logits: torch.Tensor,
               ground_truth: torch.Tensor) -> torch.Tensor:
        """
        生徒モデルの正解アイテムに対するロジットがlogit_thresholdを下回るサンプルを選択します。
        """
        # ground_truthは(batch_size,)のインデックス
        # student_logitsは(batch_size, num_items)
        
        # 各バッチサンプルの正解アイテムに対応するロジットを取得
        # gather(input, dim, index)を使用
        # indexは(batch_size, 1)に整形する必要がある
        ground_truth_logits = student_logits.gather(1, ground_truth.unsqueeze(1)).squeeze(1) # (batch_size,)

        # 正解アイテムのロジットが閾値を下回るサンプルを選択
        return ground_truth_logits < self.logit_threshold

if __name__ == "__main__":
    # テスト用のダミーデータ
    batch_size = 4
    num_items = 100

    dummy_student_logits = torch.randn(batch_size, num_items)
    dummy_teacher_logits = torch.randn(batch_size, num_items)
    dummy_ground_truth = torch.randint(0, num_items, (batch_size,)) # ground_truthはKLDivergencePolicyでは直接使用しないが、インターフェースに合わせる

    # AllSamplesPolicyのテスト
    print("--- AllSamplesPolicy Test ---")
    all_policy = AllSamplesPolicy()
    all_mask = all_policy.select(dummy_student_logits, dummy_teacher_logits, dummy_ground_truth)
    print(f"Distill mask (AllSamplesPolicy): {all_mask}")
    assert torch.equal(all_mask, torch.tensor([True, True, True, True]))
    print("AllSamplesPolicy test passed!")

    # KLDivergencePolicyのテスト
    print("--- KLDivergencePolicy Test ---")

    # ケース1: 閾値0.1でポリシーをインスタンス化
    kl_policy_0_1 = KLDivergencePolicy(kl_threshold=0.1)

    # ダミーロジットを調整して、一部が閾値を超えるようにする
    # サンプル0: KLダイバージェンスが高くなるように生徒と教師の分布を大きく離す
    dummy_student_logits_case1 = dummy_student_logits.clone()
    dummy_teacher_logits_case1 = dummy_teacher_logits.clone()
    dummy_student_logits_case1[0, 0] = 10.0 # 生徒はアイテム0を強く予測
    dummy_teacher_logits_case1[0, 1] = 10.0 # 教師はアイテム1を強く予測

    # サンプル1: KLダイバージェンスが低くなるように生徒と教師の分布を近づける
    dummy_student_logits_case1[1, 5] = 5.0
    dummy_teacher_logits_case1[1, 5] = 5.1

    # サンプル2: KLダイバージェンスが中程度になるように
    dummy_student_logits_case1[2, 10] = 3.0
    dummy_teacher_logits_case1[2, 11] = 3.5

    # サンプル3: KLダイバージェンスが非常に低くなるように (ほぼ同一)
    dummy_student_logits_case1[3] = torch.randn(num_items) * 0.01 # 小さなランダム値
    dummy_teacher_logits_case1[3] = dummy_student_logits_case1[3].clone() # 同一にする

    distill_mask_kl_0_1 = kl_policy_0_1.select(
        dummy_student_logits_case1, dummy_teacher_logits_case1, dummy_ground_truth
    )

    log_probs_s = F.log_softmax(dummy_student_logits_case1, dim=-1)
    probs_t = F.softmax(dummy_teacher_logits_case1, dim=-1)
    # ユーザーの指示に従い、log_target=Trueを設定します。
    kl_divs = F.kl_div(log_probs_s, probs_t, reduction='none', log_target=True).sum(dim=-1)
    print(f"Calculated KL Divergences (threshold=0.1): {kl_divs}")
    print(f"Distill mask (KLDivergencePolicy, threshold=0.1): {distill_mask_kl_0_1}")
    
    # 期待される結果はKLダイバージェンスの値に依存するため、具体的なアサートはKL値を確認してから
    # ここでは、少なくとも一部がTrue/Falseになることを確認
    assert distill_mask_kl_0_1.any()
    assert not distill_mask_kl_0_1.all()

    # ケース2: 閾値0.001でポリシーをインスタンス化 (より多くのサンプルが選択されるはず)
    kl_policy_0_001 = KLDivergencePolicy(kl_threshold=0.001)
    distill_mask_kl_0_001 = kl_policy_0_001.select(
        dummy_student_logits_case1, dummy_teacher_logits_case1, dummy_ground_truth
    )
    print(f"Distill mask (KLDivergencePolicy, threshold=0.001): {distill_mask_kl_0_001}")
    assert distill_mask_kl_0_001.sum() >= distill_mask_kl_0_1.sum() # 閾値が低いのでより多く選択されるか同数

    # ケース3: 閾値100.0でポリシーをインスタンス化 (ほとんど選択されないはず)
    kl_policy_100 = KLDivergencePolicy(kl_threshold=100.0)
    distill_mask_kl_100 = kl_policy_100.select(
        dummy_student_logits_case1, dummy_teacher_logits_case1, dummy_ground_truth
    )
    print(f"Distill mask (KLDivergencePolicy, threshold=100.0): {distill_mask_kl_100}")
    assert not distill_mask_kl_100.any() # 誰も選択されないはず

    print("\nKLDivergencePolicy test passed!")

    # --- GroundTruthErrorPolicy Test ---
    print("\n--- GroundTruthErrorPolicy Test ---")
    gt_error_policy = GroundTruthErrorPolicy(logit_threshold=-2.0)

    # ダミーロジットを調整して、一部が閾値を下回るようにする
    dummy_student_logits_gt = torch.randn(batch_size, num_items)
    dummy_ground_truth_gt = torch.randint(0, num_items, (batch_size,))

    # サンプル0: 正解アイテムのロジットが閾値より高い (選択されない)
    dummy_student_logits_gt[0, dummy_ground_truth_gt[0]] = 0.5 
    # サンプル1: 正解アイテムのロジットが閾値より低い (選択される)
    dummy_student_logits_gt[1, dummy_ground_truth_gt[1]] = -3.0
    # サンプル2: 正解アイテムのロジットが閾値より高い (選択されない)
    dummy_student_logits_gt[2, dummy_ground_truth_gt[2]] = 1.0
    # サンプル3: 正解アイテムのロジットが閾値より低い (選択される)
    dummy_student_logits_gt[3, dummy_ground_truth_gt[3]] = -5.0

    distill_mask_gt_error = gt_error_policy.select(
        dummy_student_logits_gt, dummy_teacher_logits, dummy_ground_truth_gt
    )
    print(f"Ground Truths: {dummy_ground_truth_gt}")
    print(f"Student Logits for GT: {dummy_student_logits_gt.gather(1, dummy_ground_truth_gt.unsqueeze(1)).squeeze(1)}")
    print(f"Distill mask (GroundTruthErrorPolicy, threshold=-2.0): {distill_mask_gt_error}")

    # 期待される結果: サンプル1と3がTrue
    expected_mask = torch.tensor([False, True, False, True])
    assert torch.equal(distill_mask_gt_error, expected_mask)

    print("\nGroundTruthErrorPolicy test passed!")