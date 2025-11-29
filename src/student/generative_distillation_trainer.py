import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from typing import Dict, Any, List
import pytorch_lightning as pl
import random

from src.student.models import SASRec
from src.teacher.generative_ranker import GenerativeRanker

logger = logging.getLogger(__name__)

class GenerativeDistillationTrainer(pl.LightningModule):
    def __init__(
        self,
        student_model: SASRec,
        teacher_model: GenerativeRanker,
        learning_rate: float = 0.001,
        lambda_emb: float = 1.0,
        lambda_rank: float = 1.0,
        num_candidates: int = 20,
        item_id_to_name: Dict[int, str] = None,
    ):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.learning_rate = learning_rate
        self.lambda_emb = lambda_emb
        self.lambda_rank = lambda_rank
        self.num_candidates = num_candidates
        self.item_id_to_name = item_id_to_name or {}
        
        # Ensure teacher is in eval
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Loss functions
        self.mse_loss = nn.MSELoss()

    def forward(self, item_seq, item_seq_len):
        return self.student_model(item_seq, item_seq_len)

    def training_step(self, batch, batch_idx):
        # 1. Prepare Data
        item_seq = batch["item_seq"] # (B, SeqLen)
        item_seq_len = batch["item_seq_len"] # (B,)
        next_item = batch["next_item"].squeeze(-1) # (B,) - 1-based ID
        
        batch_size = item_seq.size(0)
        
        # 2. Candidate Selection for Teacher Reranking
        # For simplicity, we use: Ground Truth + Random Negatives
        # In production, we might use Student's top-K predictions.
        
        candidates_ids = [] # List of tensors (B, NumCand)
        candidates_texts = [] # List of List[str]
        history_texts = [] # List[str]
        
        for b in range(batch_size):
            # History Text
            # Convert sequence IDs to names
            # item_seq[b] contains 0 for padding.
            seq = item_seq[b][:item_seq_len[b]]
            hist_names = [self.item_id_to_name.get(idx.item(), f"Item_{idx.item()}") for idx in seq if idx.item() > 0]
            history_texts.append(" | ".join(hist_names[-10:])) # Use last 10 items as context
            
            # Candidates
            # 1 Pos + (N-1) Negs
            pos_item = next_item[b].item()
            neg_items = []
            while len(neg_items) < self.num_candidates - 1:
                neg = random.randint(1, self.student_model.num_items)
                if neg != pos_item and neg not in seq:
                    neg_items.append(neg)
            
            cand_list = [pos_item] + neg_items
            random.shuffle(cand_list) # Shuffle so pos is not always first
            
            candidates_ids.append(torch.tensor(cand_list, device=self.device))
            cand_names = [self.item_id_to_name.get(idx, f"Item_{idx}") for idx in cand_list]
            candidates_texts.append(cand_names)
            
        candidates_ids = torch.stack(candidates_ids) # (B, NumCand)
        
        # 3. Teacher Inference (Deep Reasoning)
        prompts = [self.teacher_model.create_prompt(h, c) for h, c in zip(history_texts, candidates_texts)]
        
        with torch.no_grad():
            # generated_texts: List[str] e.g. "[2] > [1] ..."
            # decision_states: (B, TeacherHidden)
            generated_texts, decision_states = self.teacher_model.generate_and_extract_state(prompts)
            
        # 4. Student Inference
        # User Embedding (Projected to Teacher Dim)
        u_sas = self.student_model(item_seq, item_seq_len) # (B, TeacherHidden)
        
        # 5. Loss Calculation
        total_loss = 0.0
        
        # 5.1 Embedding Loss (Distill Decision State)
        loss_emb = self.mse_loss(u_sas, decision_states)
        total_loss += self.lambda_emb * loss_emb
        self.log("train_loss_emb", loss_emb, prog_bar=True)
        
        # 5.2 Ranking Loss (Distill Ranking)
        # Parse Teacher Ranking
        rankings = [self.teacher_model.parse_ranking(t, self.num_candidates) for t in generated_texts]
        teacher_scores = self.rank_to_score(rankings, self.num_candidates, self.device) # (B, NumCand)
        
        # Student Scores for Candidates
        # student_model.predict returns scores for ALL items (B, TotalItems+1)
        all_student_scores = self.student_model.predict(item_seq, item_seq_len)
        
        # Gather scores for the specific candidates
        # candidates_ids: (B, NumCand)
        # predict returns scores for items 1..N mapped to 0..N-1
        student_cand_scores = torch.gather(all_student_scores, 1, candidates_ids - 1) # (B, NumCand)
        
        loss_rank = self.listnet_loss(student_cand_scores, teacher_scores)
        total_loss += self.lambda_rank * loss_rank
        self.log("train_loss_rank", loss_rank, prog_bar=True)
        
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.student_model.parameters(), lr=self.learning_rate)

    def listnet_loss(self, y_pred, y_true):
        """
        ListNet Loss: KL Divergence between Softmax distributions.
        y_pred: (B, N) Student scores
        y_true: (B, N) Teacher scores (derived from rank)
        """
        P_y_true = F.softmax(y_true, dim=1)
        P_y_pred = F.softmax(y_pred, dim=1)
        return -(P_y_true * torch.log(P_y_pred + 1e-10)).sum(dim=1).mean()

    def rank_to_score(self, rankings: List[List[int]], num_candidates: int, device: str) -> torch.Tensor:
        """
        Converts rank indices to reciprocal rank scores.
        rankings: List of [item_idx_1st, item_idx_2nd, ...]
        Returns: (B, NumCand) scores
        """
        scores = torch.zeros((len(rankings), num_candidates), device=device)
        for b, rank_list in enumerate(rankings):
            for rank, item_idx in enumerate(rank_list):
                if item_idx < num_candidates:
                    # Score = 1 / log2(rank + 2)
                    # rank is 0-based index in the ranking list
                    scores[b, item_idx] = 1.0 / torch.log2(torch.tensor(rank + 2.0))
        return scores
