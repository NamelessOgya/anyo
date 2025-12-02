import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class AlphaNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class EnsembleBigRecSASRec(pl.LightningModule):
    def __init__(self, sasrec_model, alpha_net, item_embeddings, popularity_scores=None, popularity_lambda=0.0, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['sasrec_model', 'alpha_net', 'item_embeddings', 'popularity_scores'])
        
        self.sasrec = sasrec_model
        # Freeze SASRec
        self.sasrec.eval()
        for p in self.sasrec.parameters():
            p.requires_grad = False
            
        self.alpha_net = alpha_net
        
        # Register buffers for embeddings/scores so they are moved to device
        self.register_buffer('item_embeddings', item_embeddings)
        if popularity_scores is not None:
            self.register_buffer('popularity_scores', popularity_scores)
        else:
            self.popularity_scores = None
            
        self.popularity_lambda = popularity_lambda
        self.lr = lr
        
    def forward(self, seq, seq_len, bigrec_emb):
        # 1. SASRec Forward
        # sasrec_emb: (B, Hidden) - Representation of the last item in sequence
        sasrec_emb = self.sasrec(seq, seq_len)
        
        # sasrec_logits: (B, NumItems+1) - Scores for all items (including padding at 0)
        sasrec_logits = self.sasrec.predict(seq, seq_len)
        
        # 2. BigRec Scores
        # bigrec_emb: (B, EmbDim)
        # Compute distances
        # item_embeddings: (NumItems+1, EmbDim)
        dists = torch.cdist(bigrec_emb.float(), self.item_embeddings.float(), p=2) # (B, NumItems+1)
        
        # Normalize Distances (Min-Max per batch)
        # Note: BigRecModel normalizes by dividing by max_dist per sample
        max_dist = dists.max(dim=1, keepdim=True)[0]
        dists = dists / (max_dist + 1e-8)
        
        # Apply Popularity Adjustment
        if self.popularity_scores is not None and self.popularity_lambda > 0:
            pop_factor = (self.popularity_scores + 1.0) ** self.popularity_lambda
            dists = dists / pop_factor.unsqueeze(0)
            
        # Convert Distances to Probabilities (Softmax)
        # We negate distances because smaller is better.
        # We might need a temperature scaling for BigRec to match SASRec's sharpness.
        # For now, let's use a learnable temperature or fixed 1.0.
        # Or just use raw negated distances.
        bigrec_logits = -dists * 10.0 # Scale up a bit?
        
        # 3. Alpha
        alpha = self.alpha_net(sasrec_emb) # (B, 1)
        
        # 4. Combine
        # We can combine logits or probabilities.
        # Mixing probabilities is safer for ensemble.
        
        sasrec_probs = F.softmax(sasrec_logits, dim=-1)
        bigrec_probs = F.softmax(bigrec_logits, dim=-1)
        
        combined_probs = alpha * bigrec_probs + (1 - alpha) * sasrec_probs
        
        return combined_probs, alpha

    def training_step(self, batch, batch_idx):
        seq = batch['seq']
        seq_len = batch['len_seq']
        target = batch['next_item']
        bigrec_emb = batch['bigrec_emb']
        
        combined_probs, alpha = self(seq, seq_len, bigrec_emb)
        
        # Loss: NLL
        # combined_probs: (B, NumItems+1)
        # target: (B,)
        
        # Add epsilon to avoid log(0)
        loss = F.nll_loss(torch.log(combined_probs + 1e-8), target)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_alpha_mean", alpha.mean(), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq = batch['seq']
        seq_len = batch['len_seq']
        target = batch['next_item']
        bigrec_emb = batch['bigrec_emb']
        
        combined_probs, alpha = self(seq, seq_len, bigrec_emb)
        
        # Metrics: HR@10, NDCG@10
        # Get top-k
        k = 10
        _, topk_indices = torch.topk(combined_probs, k=k, dim=1)
        
        hits = 0
        ndcg = 0
        batch_size = len(target)
        
        for i in range(batch_size):
            t = target[i].item()
            preds = topk_indices[i].tolist()
            
            if t in preds:
                hits += 1
                rank = preds.index(t)
                ndcg += 1.0 / torch.log2(torch.tensor(rank + 2.0))
                
        self.log("val_hr@10", hits / batch_size, prog_bar=True)
        self.log("val_ndcg@10", ndcg / batch_size, prog_bar=True)
        self.log("val_alpha_mean", alpha.mean(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        seq = batch['seq']
        seq_len = batch['len_seq']
        target = batch['next_item']
        bigrec_emb = batch['bigrec_emb']
        
        combined_probs, alpha = self(seq, seq_len, bigrec_emb)
        
        # Metrics: HR@10, NDCG@10
        k = 10
        _, topk_indices = torch.topk(combined_probs, k=k, dim=1)
        
        hits = 0
        ndcg = 0
        batch_size = len(target)
        
        for i in range(batch_size):
            t = target[i].item()
            preds = topk_indices[i].tolist()
            
            if t in preds:
                hits += 1
                rank = preds.index(t)
                ndcg += 1.0 / torch.log2(torch.tensor(rank + 2.0))
                
        self.log("test_hr@10", hits / batch_size, prog_bar=True)
        self.log("test_ndcg@10", ndcg / batch_size, prog_bar=True)
        self.log("test_alpha_mean", alpha.mean(), prog_bar=True)

    def predict_alpha(self, seq, seq_len):
        """
        Predict alpha value for a given sequence using SASRec and AlphaNetwork.
        Does not require BigRec embeddings.
        """
        # 1. SASRec Forward
        sasrec_emb = self.sasrec(seq, seq_len)
        
        # 2. Alpha
        alpha = self.alpha_net(sasrec_emb) # (B, 1)
        
        return alpha

    def configure_optimizers(self):
        return torch.optim.Adam(self.alpha_net.parameters(), lr=self.lr)
