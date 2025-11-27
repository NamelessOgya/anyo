import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import logging

logger = logging.getLogger(__name__)

class SamplingStrategy(ABC):
    """
    Base class for Active Learning sampling strategies.
    All strategies use the Student Model to select samples for the Teacher.
    """
    def __init__(self, model: torch.nn.Module, device: torch.device, mining_ratio: float, **kwargs):
        self.model = model
        self.device = device
        self.mining_ratio = mining_ratio
        self.kwargs = kwargs

    def select_indices(self, dataloader: DataLoader) -> List[int]:
        """
        Selects indices of samples to keep.
        
        Args:
            dataloader: DataLoader containing the entire training set (shuffle=False).
            
        Returns:
            List of selected indices.
        """
        scores = self.calculate_scores(dataloader)
        num_samples = len(scores)
        num_keep = int(num_samples * self.mining_ratio)
        
        # Default behavior: Select top-k samples with highest scores
        # Strategies should return scores where higher is "more informative/harder"
        sorted_indices = np.argsort(scores)[::-1]
        selected_indices = sorted_indices[:num_keep].tolist()
        
        logger.info(f"Selected {len(selected_indices)}/{num_samples} samples using {self.__class__.__name__}")
        return selected_indices

    @abstractmethod
    def calculate_scores(self, dataloader: DataLoader) -> np.ndarray:
        """
        Calculates a score for each sample in the dataloader.
        Higher score means the sample is more likely to be selected.
        """
        pass

class RandomStrategy(SamplingStrategy):
    def calculate_scores(self, dataloader: DataLoader) -> np.ndarray:
        # Random scores
        num_samples = len(dataloader.dataset)
        return np.random.rand(num_samples)

class EntropyStrategy(SamplingStrategy):
    def calculate_scores(self, dataloader: DataLoader) -> np.ndarray:
        scores = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Entropy"):
                seq = batch["seq"].to(self.device)
                len_seq = batch["len_seq"].to(self.device)
                logits = self.model.predict(seq, len_seq) # (B, num_items)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-10)
                entropy = -torch.sum(probs * log_probs, dim=-1)
                scores.extend(entropy.cpu().numpy().tolist())
        return np.array(scores)

class LeastConfidenceStrategy(SamplingStrategy):
    def calculate_scores(self, dataloader: DataLoader) -> np.ndarray:
        scores = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Least Confidence"):
                seq = batch["seq"].to(self.device)
                len_seq = batch["len_seq"].to(self.device)
                logits = self.model.predict(seq, len_seq)
                probs = torch.softmax(logits, dim=-1)
                max_probs, _ = torch.max(probs, dim=-1)
                # Least confidence: 1 - max_prob
                # Higher is more uncertain
                uncertainty = 1.0 - max_probs
                scores.extend(uncertainty.cpu().numpy().tolist())
        return np.array(scores)

class MarginStrategy(SamplingStrategy):
    def calculate_scores(self, dataloader: DataLoader) -> np.ndarray:
        scores = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Margin"):
                seq = batch["seq"].to(self.device)
                len_seq = batch["len_seq"].to(self.device)
                logits = self.model.predict(seq, len_seq)
                probs = torch.softmax(logits, dim=-1)
                # Get top 2 probabilities
                top2_probs, _ = torch.topk(probs, k=2, dim=-1)
                # Margin = p1 - p2
                # We want smallest margin -> Higher score = -Margin or 1 - Margin
                margin = top2_probs[:, 0] - top2_probs[:, 1]
                score = 1.0 - margin
                scores.extend(score.cpu().numpy().tolist())
        return np.array(scores)

class LossStrategy(SamplingStrategy):
    def calculate_scores(self, dataloader: DataLoader) -> np.ndarray:
        scores = []
        self.model.eval()
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating Loss"):
                seq = batch["seq"].to(self.device)
                len_seq = batch["len_seq"].to(self.device)
                target = (batch["next_item"] - 1).to(self.device)
                logits = self.model.predict(seq, len_seq)
                loss = loss_fn(logits, target)
                scores.extend(loss.cpu().numpy().tolist())
        return np.array(scores)

class GradientNormStrategy(SamplingStrategy):
    def calculate_scores(self, dataloader: DataLoader) -> np.ndarray:
        scores = []
        self.model.eval() # Keep eval for dropout etc, but we need grads
        # Note: Calculating gradients for all samples is expensive.
        # We will use the gradient of the loss w.r.t the last layer weights or embeddings.
        # For efficiency, we often approximate this.
        # Here, we'll calculate the gradient norm of the last layer (item_embedding used as output weights).
        
        # Important: We need to enable grad for this strategy
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for batch in tqdm(dataloader, desc="Calculating Grad Norm"):
            seq = batch["seq"].to(self.device)
            len_seq = batch["len_seq"].to(self.device)
            target = (batch["next_item"] - 1).to(self.device)
            
            # We need to clear grads
            self.model.zero_grad()
            
            # Forward
            logits = self.model.predict(seq, len_seq)
            loss = loss_fn(logits, target)
            
            # Backward
            loss.backward()
            
            # Calculate norm of gradients of the last layer (item_embedding.weight)
            # SASRec uses item_embedding for both input and output (shared)
            # We assume model.item_embedding.weight.grad is populated
            grad_norm = self.model.item_embedding.weight.grad.norm().item()
            
            # Since we process in batches, this grad_norm is for the BATCH.
            # Ideally, we need per-sample gradient norm.
            # Per-sample gradient is very expensive without Functorch/Vmap.
            # Fallback/Approximation: Use the norm of the embedding of the target item * error signal?
            # Or: Just use LossStrategy as a proxy if per-sample grad is too slow.
            
            # Alternative: Gradient of the loss w.r.t the *last hidden state* (user embedding).
            # This is (B, H). Norm is (B,). This is efficient.
            
            # Let's re-do forward to get hidden state and compute grad w.r.t it.
            # We need to hook or modify model to get hidden state.
            # SASRec.predict calls log2feats then linear.
            pass 
        
        # Re-implementation using hidden state gradient
        scores = []
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        
        for batch in tqdm(dataloader, desc="Calculating Grad Norm (Hidden)"):
            seq = batch["seq"].to(self.device)
            len_seq = batch["len_seq"].to(self.device)
            target = (batch["next_item"] - 1).to(self.device)
            
            # Get hidden states
            # We need to access the internal method of SASRec
            # Assuming model.log2feats(seq, len_seq) returns (B, max_len, H)
            # We want the last step: (B, H)
            
            # Temporarily set requires_grad to True for input/hidden if needed, 
            # but usually we just need to run forward and backward.
            
            # To get per-sample gradients efficiently:
            # Gradient of Loss L_i w.r.t Logits z_i is p_i - y_i.
            # Gradient w.r.t Hidden h_i is (p_i - y_i) @ W.
            # Norm is ||(p_i - y_i) @ W||.
            
            with torch.no_grad():
                # 1. Get Hidden States
                log_feats = self.model.log2feats(seq) # (B, L, H)
                # Select last step
                final_feats = []
                for i, l in enumerate(len_seq):
                    final_feats.append(log_feats[i, l-1, :])
                final_feats = torch.stack(final_feats) # (B, H)
                
                # 2. Get Logits
                # logits = final_feats @ item_embedding.weight.T + bias
                item_embs = self.model.item_embedding.weight # (I, H)
                logits = final_feats @ item_embs.t() # (B, I)
                
                # 3. Calculate Probabilities
                probs = torch.softmax(logits, dim=-1) # (B, I)
                
                # 4. Calculate Error Signal (p - y)
                # We only need the rows corresponding to our batch.
                # Construct one-hot targets
                one_hot_targets = torch.zeros_like(probs)
                one_hot_targets.scatter_(1, target.unsqueeze(1), 1)
                
                error = probs - one_hot_targets # (B, I)
                
                # 5. Gradient w.r.t Hidden State h: error @ W
                # (B, I) @ (I, H) -> (B, H)
                grad_h = error @ item_embs
                
                # 6. Norm
                grad_norms = torch.norm(grad_h, dim=-1) # (B,)
                scores.extend(grad_norms.cpu().numpy().tolist())
                
        return np.array(scores)

class CoresetStrategy(SamplingStrategy):
    def select_indices(self, dataloader: DataLoader) -> List[int]:
        # Override select_indices because this is not score-based sorting
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting Embeddings for Coreset"):
                seq = batch["seq"].to(self.device)
                len_seq = batch["len_seq"].to(self.device)
                
                log_feats = self.model.log2feats(seq)
                final_feats = []
                for i, l in enumerate(len_seq):
                    final_feats.append(log_feats[i, l-1, :])
                final_feats = torch.stack(final_feats)
                embeddings.append(final_feats.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0) # (N, H)
        num_samples = embeddings.shape[0]
        num_keep = int(num_samples * self.mining_ratio)
        
        # k-Center Greedy
        # Initialize with one random point
        selected_indices = [np.random.randint(num_samples)]
        
        # Maintain distances to nearest selected point
        # dists[i] = min_{j in selected} ||x_i - x_j||
        dists = pairwise_distances(embeddings, embeddings[selected_indices])
        min_dists = dists.min(axis=1)
        
        for _ in tqdm(range(num_keep - 1), desc="Coreset Selection"):
            # Select point with maximum min_dist
            if np.max(min_dists) == 0:
                # All remaining points are covered (or identical to selected)
                # Pick a random unselected point
                all_indices = set(range(num_samples))
                remaining = list(all_indices - set(selected_indices))
                if not remaining:
                    break
                new_idx = np.random.choice(remaining)
            else:
                new_idx = np.argmax(min_dists)
            
            selected_indices.append(new_idx)
            
            # Update distances
            new_dists = pairwise_distances(embeddings, embeddings[new_idx:new_idx+1])
            min_dists = np.minimum(min_dists, new_dists.flatten())
            
        return selected_indices

    def calculate_scores(self, dataloader: DataLoader) -> np.ndarray:
        raise NotImplementedError("Coreset uses custom selection logic.")

class KMeansStrategy(SamplingStrategy):
    def select_indices(self, dataloader: DataLoader) -> List[int]:
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting Embeddings for KMeans"):
                seq = batch["seq"].to(self.device)
                len_seq = batch["len_seq"].to(self.device)
                
                log_feats = self.model.log2feats(seq)
                final_feats = []
                for i, l in enumerate(len_seq):
                    final_feats.append(log_feats[i, l-1, :])
                final_feats = torch.stack(final_feats)
                embeddings.append(final_feats.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        num_samples = embeddings.shape[0]
        num_keep = int(num_samples * self.mining_ratio)
        
        # Perform KMeans
        # Note: If num_keep is large (e.g. 50% of data), KMeans with k=num_keep is slow.
        # Standard KMeans sampling usually means: Cluster into K groups, select samples nearest to centroids.
        # Here we set n_clusters = num_keep.
        # Optimization: If num_keep is very large, maybe we should just use Random or Coreset.
        # Or: Cluster into smaller K, and select N/K samples per cluster.
        # Let's assume user wants representative samples.
        # If mining_ratio is high (e.g. 0.5), KMeans is probably not the best "Selection" method directly.
        # But per request, we implement it. We will use MiniBatchKMeans for speed if available.
        
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=num_keep, batch_size=256, random_state=42, n_init='auto')
        kmeans.fit(embeddings)
        
        # Find nearest samples to centroids
        # transform returns distance to each center
        dists = kmeans.transform(embeddings) # (N, K)
        # For each cluster (column), find min index
        # This might select same sample for multiple clusters if clusters are close?
        # Usually we find the sample closest to each centroid.
        selected_indices = np.argmin(dists, axis=0).tolist()
        
        # Ensure unique
        selected_indices = list(set(selected_indices))
        
        # If we lost samples due to duplicates, fill with random remaining samples
        if len(selected_indices) < num_keep:
            num_needed = num_keep - len(selected_indices)
            all_indices = set(range(num_samples))
            remaining_indices = list(all_indices - set(selected_indices))
            if len(remaining_indices) >= num_needed:
                fill_indices = np.random.choice(remaining_indices, num_needed, replace=False).tolist()
                selected_indices.extend(fill_indices)
            else:
                selected_indices.extend(remaining_indices)
        
        return selected_indices

    def calculate_scores(self, dataloader: DataLoader) -> np.ndarray:
        raise NotImplementedError("KMeans uses custom selection logic.")

class BADGEStrategy(SamplingStrategy):
    def select_indices(self, dataloader: DataLoader) -> List[int]:
        # BADGE: Cluster Gradient Embeddings
        # Gradient Embedding g_x = \grad_\theta L(x, \hat{y})
        # For last layer W (H x C), grad is h_x^T (p - \hat{y})?
        # Actually BADGE usually uses the gradient w.r.t the last layer parameters.
        # Dimension is H * C. This is huge.
        # BADGE approximation: Use (p - \hat{y}) \otimes h_x?
        # Or just cluster h_x scaled by uncertainty?
        
        # Standard BADGE implementation:
        # 1. Compute hypothetical labels \hat{y} = argmax p(y|x)
        # 2. Compute gradients assuming \hat{y} is the true label.
        #    Grad_W = h_x^T (p - 1_{\hat{y}})
        #    This is sparse-ish? No, p is dense.
        #    But usually we only care about the magnitude or direction.
        
        # Simplified BADGE for RecSys (High dimensional output C):
        # The gradient embedding is too large (H * num_items).
        # We will use a simplified version: Cluster [h_x * Uncertainty] or similar.
        # Or: Cluster h_x weighted by gradient norm.
        
        # Let's stick to the spirit: Diversity in the space of "errors".
        # We'll compute the "Gradient w.r.t Hidden State" (as in GradientNorm) -> (B, H).
        # This represents "in which direction should the user embedding move to fix the error".
        # Clustering this (B, H) space gives diversity in error directions.
        
        self.model.eval()
        grad_embeddings = []
        
        for batch in tqdm(dataloader, desc="Extracting Gradient Embeddings (BADGE)"):
            seq = batch["seq"].to(self.device)
            len_seq = batch["len_seq"].to(self.device)
            
            # Predict
            with torch.no_grad():
                log_feats = self.model.log2feats(seq)
                final_feats = []
                for i, l in enumerate(len_seq):
                    final_feats.append(log_feats[i, l-1, :])
                final_feats = torch.stack(final_feats) # (B, H)
                
                item_embs = self.model.item_embedding.weight
                logits = final_feats @ item_embs.t()
                probs = torch.softmax(logits, dim=-1)
                
                # Hypothetical label (Top-1) - Standard BADGE uses predicted label
                preds = torch.argmax(probs, dim=-1)
                
                # One-hot predicted
                one_hot_preds = torch.zeros_like(probs)
                one_hot_preds.scatter_(1, preds.unsqueeze(1), 1)
                
                # Error signal: p - 1_{\hat{y}}
                # Note: In standard BADGE, they use the gradient of the loss at the *predicted* class.
                # Here we use the gradient vector w.r.t hidden state.
                error = probs - one_hot_preds
                
                # Grad w.r.t Hidden: error @ W
                grad_h = error @ item_embs # (B, H)
                
                grad_embeddings.append(grad_h.cpu().numpy())

        grad_embeddings = np.concatenate(grad_embeddings, axis=0) # (N, H)
        num_samples = grad_embeddings.shape[0]
        num_keep = int(num_samples * self.mining_ratio)
        
        # k-Means++ seeding (BADGE uses k-Means++ seeding to select points)
        # We can use sklearn's kmeans_plusplus
        from sklearn.cluster import kmeans_plusplus
        
        centers, indices = kmeans_plusplus(grad_embeddings, n_clusters=num_keep, random_state=42)
        selected_indices = indices.tolist()
        
        # Ensure unique
        selected_indices = list(set(selected_indices))
        
        # Fill if needed
        if len(selected_indices) < num_keep:
            num_needed = num_keep - len(selected_indices)
            all_indices = set(range(num_samples))
            remaining_indices = list(all_indices - set(selected_indices))
            if len(remaining_indices) >= num_needed:
                fill_indices = np.random.choice(remaining_indices, num_needed, replace=False).tolist()
                selected_indices.extend(fill_indices)
            else:
                selected_indices.extend(remaining_indices)
                
        return selected_indices

    def calculate_scores(self, dataloader: DataLoader) -> np.ndarray:
        raise NotImplementedError("BADGE uses custom selection logic.")

class EntropyWeightedDiversityStrategy(SamplingStrategy):
    def select_indices(self, dataloader: DataLoader) -> List[int]:
        # 1. Calculate Entropy
        entropy_strategy = EntropyStrategy(self.model, self.device, self.mining_ratio)
        entropies = entropy_strategy.calculate_scores(dataloader)
        
        num_samples = len(entropies)
        num_keep = int(num_samples * self.mining_ratio)
        
        # 2. Filter top M% (e.g. 2 * num_keep) high entropy samples
        # To ensure we have enough candidates for diversity
        pool_size = min(num_samples, num_keep * 2)
        top_entropy_indices = np.argsort(entropies)[::-1][:pool_size]
        
        # 3. Extract Embeddings for these samples
        # This requires re-running forward pass or caching.
        # For simplicity, we re-run but only for selected indices?
        # Actually we need to run for all to map indices correctly, or use a Subset dataloader.
        
        # Create a subset dataloader
        subset = torch.utils.data.Subset(dataloader.dataset, top_entropy_indices)
        subset_loader = DataLoader(
            subset, 
            batch_size=dataloader.batch_size, 
            shuffle=False, 
            collate_fn=dataloader.collate_fn,
            num_workers=dataloader.num_workers
        )
        
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in tqdm(subset_loader, desc="Extracting Embeddings for Diversity"):
                seq = batch["seq"].to(self.device)
                len_seq = batch["len_seq"].to(self.device)
                
                log_feats = self.model.log2feats(seq)
                final_feats = []
                for i, l in enumerate(len_seq):
                    final_feats.append(log_feats[i, l-1, :])
                final_feats = torch.stack(final_feats)
                embeddings.append(final_feats.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0) # (Pool, H)
        
        # 4. Cluster (KMeans) to select num_keep samples from the pool
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=num_keep, batch_size=256, random_state=42, n_init='auto')
        kmeans.fit(embeddings)
        dists = kmeans.transform(embeddings)
        local_selected_indices = np.argmin(dists, axis=0)
        
        # Map back to original indices
        global_selected_indices = [top_entropy_indices[i] for i in local_selected_indices]
        
        # Ensure unique
        global_selected_indices = list(set(global_selected_indices))
        
        # Fill if needed
        if len(global_selected_indices) < num_keep:
            num_needed = num_keep - len(global_selected_indices)
            # Candidates are from the top_entropy_indices pool
            pool_indices_set = set(top_entropy_indices)
            current_indices_set = set(global_selected_indices)
            remaining_pool = list(pool_indices_set - current_indices_set)
            
            if len(remaining_pool) >= num_needed:
                fill_indices = np.random.choice(remaining_pool, num_needed, replace=False).tolist()
                global_selected_indices.extend(fill_indices)
            else:
                # If pool is exhausted, take everything from pool
                global_selected_indices.extend(remaining_pool)
                # If still need more, take from outside pool (low entropy)
                num_still_needed = num_keep - len(global_selected_indices)
                if num_still_needed > 0:
                    all_indices = set(range(num_samples))
                    # Exclude everything we have
                    remaining_all = list(all_indices - set(global_selected_indices))
                    if len(remaining_all) >= num_still_needed:
                        fill_indices_2 = np.random.choice(remaining_all, num_still_needed, replace=False).tolist()
                        global_selected_indices.extend(fill_indices_2)
                    else:
                         global_selected_indices.extend(remaining_all)

        return global_selected_indices

    def calculate_scores(self, dataloader: DataLoader) -> np.ndarray:
        raise NotImplementedError("Hybrid strategy.")

STRATEGY_MAP = {
    "random": RandomStrategy,
    "entropy": EntropyStrategy,
    "least_confidence": LeastConfidenceStrategy,
    "margin": MarginStrategy,
    "loss": LossStrategy,
    "gradient_norm": GradientNormStrategy,
    "coreset": CoresetStrategy,
    "kmeans": KMeansStrategy,
    "badge": BADGEStrategy,
    "entropy_diversity": EntropyWeightedDiversityStrategy,
}

def get_strategy(name: str, model: torch.nn.Module, device: torch.device, mining_ratio: float, **kwargs) -> SamplingStrategy:
    if name not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_MAP.keys())}")
    return STRATEGY_MAP[name](model, device, mining_ratio, **kwargs)
