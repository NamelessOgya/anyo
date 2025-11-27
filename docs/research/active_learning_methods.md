# Active Learning Methods for Knowledge Distillation

Based on the user's request for "Effectiveness" and "Diversity", and the context of Knowledge Distillation (Student struggling), we have selected the following 10 sampling strategies.

## 1. Random Sampling (`random`)
- **Type**: Baseline
- **Description**: Randomly selects samples from the pool.
- **Purpose**: Serves as a baseline to measure the effectiveness of other strategies.

## 2. Entropy Sampling (`entropy`)
- **Type**: Uncertainty (Effectiveness)
- **Description**: Selects samples where the Student's prediction probability distribution has the highest entropy.
- **Formula**: $H(x) = - \sum p(y|x) \log p(y|x)$
- **Rationale**: High entropy indicates the model is uncertain about the class distribution.

## 3. Least Confidence (`least_confidence`)
- **Type**: Uncertainty (Effectiveness)
- **Description**: Selects samples where the probability of the most likely class is lowest.
- **Formula**: $LC(x) = 1 - \max_y p(y|x)$
- **Rationale**: Captures the "hardest" single decision.

## 4. Margin Sampling (`margin`)
- **Type**: Uncertainty (Effectiveness)
- **Description**: Selects samples with the smallest difference between the top-1 and top-2 class probabilities.
- **Formula**: $M(x) = p(y_1|x) - p(y_2|x)$ (Select smallest $M$)
- **Rationale**: Focuses on samples near the decision boundary.

## 5. Loss-based Sampling (`loss`)
- **Type**: Effectiveness (Hard Example Mining)
- **Description**: Selects samples with the highest Cross-Entropy Loss against the Ground Truth.
- **Formula**: $L(x, y) = - \log p(y_{true}|x)$
- **Rationale**: Directly targets samples the model currently misclassifies or predicts with low confidence.

## 6. Gradient Norm (`gradient_norm`)
- **Type**: Effectiveness
- **Description**: Selects samples that generate the largest gradient norms in the model's last layer.
- **Rationale**: Large gradients imply the sample would induce a large update to the model parameters, suggesting it is informative.

## 7. Coreset Selection (`coreset`)
- **Type**: Diversity
- **Description**: Uses k-Center Greedy algorithm on the sample embeddings.
- **Rationale**: Selects a subset of samples such that the maximum distance from any non-selected sample to its nearest selected sample is minimized. Ensures coverage of the embedding space.

## 8. K-Means Clustering (`kmeans`)
- **Type**: Diversity (Representative)
- **Description**: Performs K-Means clustering on embeddings and selects the samples nearest to the centroids.
- **Rationale**: Selects representative samples from the underlying data distribution.

## 9. BADGE (`badge`)
- **Type**: Hybrid (Uncertainty + Diversity)
- **Description**: Batch Active learning by Diverse Gradient Embeddings. Clusters the "gradient embeddings" (gradient of the last layer) using k-Means++.
- **Rationale**: Captures both uncertainty (gradient magnitude) and diversity (direction in parameter space).

## 10. Entropy-weighted Diversity (`entropy_diversity`)
- **Type**: Hybrid
- **Description**: A heuristic combination. First, filters the top $M\%$ high-entropy samples, then performs clustering (e.g., K-Means) on this subset to select $N$ samples.
- **Rationale**: Ensures that the selected uncertain samples are also diverse, avoiding selecting many similar "hard" examples (e.g., outliers).

---

## Implementation Strategy
- **Interface**: `SamplingStrategy` abstract base class.
- **Input**: Student Model, DataLoader (unshuffled).
- **Output**: List of indices.
- **Configuration**: Hydra config to switch strategies and parameters (e.g., `n_clusters`).
