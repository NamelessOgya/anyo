import torch
import torch.nn as nn
import math

class SASRec(nn.Module):
    """
    SASRec (Self-Attentive Sequential Recommendation) モデルの実装。
    """
    def __init__(self, num_items: int, hidden_size: int, num_heads: int, num_layers: int, dropout_rate: float, max_seq_len: int, teacher_embedding_dim: int = None, ed_weight: float = 0.0, padding_item_id: int = 0):
        super(SASRec, self).__init__()

        self.num_items = num_items
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.teacher_embedding_dim = teacher_embedding_dim
        self.ed_weight = ed_weight

        # ユーザーとアイテムの埋め込み層
        # 0番目のアイテムIDはパディング用として予約
        self.item_embeddings = nn.Embedding(num_items + 2, hidden_size, padding_idx=padding_item_id)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

        # Transformerエンコーダ
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])

        self.layernorm = nn.LayerNorm(hidden_size)

        # 蒸留のために教師モデルの埋め込み次元に合わせるプロジェクション層
        self.embedding_projection = None
        if teacher_embedding_dim is not None and hidden_size != teacher_embedding_dim:
            self.embedding_projection = nn.Linear(hidden_size, teacher_embedding_dim)

        # For embedding distillation
        self.teacher_embedding_projection = None
        if teacher_embedding_dim is not None and hidden_size != teacher_embedding_dim:
            self.teacher_embedding_projection = nn.Linear(teacher_embedding_dim, hidden_size)

    def _get_last_item_representation(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor, teacher_embeddings: torch.Tensor = None):
        """
        シーケンスの最後のアイテムの表現を計算します。
        """
        # アイテム埋め込み
        item_embeddings = self.item_embeddings(item_seq)


        # Add teacher embeddings for distillation
        if teacher_embeddings is not None and self.ed_weight > 0:
            if self.teacher_embedding_projection:
                teacher_embeddings = self.teacher_embedding_projection(teacher_embeddings)
            # Ensure teacher_embeddings are detached before adding to avoid breaking student's grad flow
            # Even if already no_grad, explicit detach can sometimes help autograd
            teacher_embeddings_detached = teacher_embeddings.detach()
            item_embeddings = item_embeddings + self.ed_weight * teacher_embeddings_detached.unsqueeze(1)
        # 位置埋め込み
        positions = torch.arange(self.max_seq_len, device=item_seq.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(positions)

        # 埋め込みの合計
        input_embeddings = item_embeddings + position_embeddings
        input_embeddings = self.dropout(input_embeddings)

        # アテンションマスクの作成
        # パディング部分 (ID=0) はアテンションしない
        attention_mask = (item_seq != 0).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, max_seq_len)
        # 自己回帰的なアテンションを適用 (未来のアイテムを見ない)
        # 下三角行列を作成
        subsequent_mask = torch.triu(
            torch.ones((self.max_seq_len, self.max_seq_len), device=item_seq.device), diagonal=1
        ).bool()
        attention_mask = attention_mask & ~subsequent_mask # (batch_size, 1, max_seq_len, max_seq_len)

        # Transformerブロックを通過
        for transformer in self.transformer_blocks:
            input_embeddings = transformer(input_embeddings, attention_mask)

        # LayerNorm
        output = self.layernorm(input_embeddings)

        # 各シーケンスの最後のアイテムの表現を取得
        # item_seq_lenは実際の長さなので、-1して0-indexedにする
        last_item_indices = item_seq_len - 1
        # gatherを使って、各バッチの最後のアイテムの埋め込みを取得
        # output: (batch_size, max_seq_len, hidden_size)
        # last_item_indices: (batch_size)
        # 期待される結果: (batch_size, hidden_size)
        # gatherはdimとindexの形状を合わせる必要がある
        last_item_representation = torch.gather(
            output,
            1,
            last_item_indices.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
        ).squeeze(1)
        return last_item_representation

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor, teacher_embeddings: torch.Tensor = None):
        """
        Args:
            item_seq (torch.Tensor): ユーザーのアイテムシーケンス (batch_size, max_seq_len)。
                                     パディングは0。
            item_seq_len (torch.Tensor): 各シーケンスの実際の長さ (batch_size)。
            teacher_embeddings (torch.Tensor): 教師モデルの埋め込み (batch_size, teacher_embedding_dim)。
        Returns:
            torch.Tensor: 各シーケンスの最後のアイテムの表現 (batch_size, hidden_size)。
                          teacher_embedding_dimが指定されている場合はその次元にプロジェクションされる。
        """
        last_item_representation = self._get_last_item_representation(item_seq, item_seq_len, teacher_embeddings)
        if self.embedding_projection:
            return self.embedding_projection(last_item_representation)
        return last_item_representation

    def predict(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor):
        """
        推薦スコアを計算します。
        Args:
            item_seq (torch.Tensor): ユーザーのアイテムシーケンス (batch_size, max_seq_len)。
            item_seq_len (torch.Tensor): 各シーケンスの実際の長さ (batch_size)。
        Returns:
            torch.Tensor: 各アイテムに対する推薦スコア (batch_size, num_items + 1)。
        """
        # predictではプロジェクション前の表現を使用
        last_item_representation_for_prediction = self._get_last_item_representation(item_seq, item_seq_len)
        # 全アイテム埋め込みとの内積を計算
        # (batch_size, hidden_size) @ (hidden_size, num_items + 1) -> (batch_size, num_items + 1)
        scores = torch.matmul(last_item_representation_for_prediction, self.item_embeddings.weight[:-1].transpose(0, 1))
        return scores


class TransformerBlock(nn.Module):
    """
    SASRecで使用されるTransformerブロック。
    Multi-head Self-Attention と Point-wise Feed-forward Network から構成されます。
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout_rate)
        self.feed_forward = PointWiseFeedForward(hidden_size, dropout_rate)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        # Self-Attention
        # 残差接続とLayerNorm
        x = self.layernorm1(x + self.dropout1(self.attention(x, attention_mask)))
        # Point-wise Feed-forward
        # 残差接続とLayerNorm
        x = self.layernorm2(x + self.dropout2(self.feed_forward(x)))
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head Self-Attentionの実装。
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.scaling = self.head_size ** -0.5

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len, _ = x.size()

        # Q, K, Vを線形変換し、ヘッドに分割
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        # スコアの計算 (Q @ K^T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # マスクの適用
        # attention_mask: (batch_size, 1, max_seq_len, max_seq_len)
        # scores: (batch_size, num_heads, max_seq_len, max_seq_len)
        scores.masked_fill_(attention_mask == 0, -1e9) # マスクされた部分を非常に小さな値に設定

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Valueとの積算 (Attention @ V)
        output = torch.matmul(attention_weights, v)

        # ヘッドを結合し、最終的な線形変換
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out(output)
        return output


class PointWiseFeedForward(nn.Module):
    """
    Point-wise Feed-forward Networkの実装。
    """
    def __init__(self, hidden_size: int, dropout_rate: float):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4) # 論文では通常4倍
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU() # BERTなどではGELUが使われることが多い

    def forward(self, x: torch.Tensor):
        return self.dropout(self.linear2(self.activation(self.linear1(x))))


if __name__ == "__main__":
    # テスト用のパラメータ
    num_users = 100
    num_items = 5000
    hidden_size = 64
    num_heads = 2
    num_layers = 2
    dropout_rate = 0.1
    max_seq_len = 50
    batch_size = 4
    teacher_embedding_dim = 768 # 教師モデルの埋め込み次元

    # ダミーデータ
    # item_seq: (batch_size, max_seq_len)
    # 0はパディング、1からnum_itemsまでのIDを使用
    item_seq = torch.randint(1, num_items, (batch_size, max_seq_len))
    # シーケンスの長さをランダムに設定 (1からmax_seq_lenまで)
    item_seq_len = torch.randint(1, max_seq_len + 1, (batch_size,))

    # パディングを適用
    for i in range(batch_size):
        if item_seq_len[i] < max_seq_len:
            item_seq[i, item_seq_len[i]:] = 0 # 実際の長さ以降を0でパディング

    print(f"Item Sequence Shape: {item_seq.shape}")
    print(f"Item Sequence Lengths: {item_seq_len}")

    # モデルのインスタンス化 (プロジェクションあり)
    model_with_projection = SASRec(num_items, hidden_size, num_heads, num_layers, dropout_rate, max_seq_len, teacher_embedding_dim=teacher_embedding_dim)
    print(f"Model with projection: {model_with_projection}")

    # forwardパスのテスト (プロジェクションあり)
    output_representation_proj = model_with_projection(item_seq, item_seq_len)
    print(f"Output Representation (with projection) Shape: {output_representation_proj.shape}") # 期待: (batch_size, teacher_embedding_dim)
    assert output_representation_proj.shape == (batch_size, teacher_embedding_dim)

    # predictパスのテスト (プロジェクションあり)
    prediction_scores_proj = model_with_projection.predict(item_seq, item_seq_len)
    print(f"Prediction Scores (with projection) Shape: {prediction_scores_proj.shape}") # 期待: (batch_size, num_items + 1)
    assert prediction_scores_proj.shape == (batch_size, num_items + 1)

    # モデルのインスタンス化 (プロジェクションなし)
    model_no_projection = SASRec(num_items, hidden_size, num_heads, num_layers, dropout_rate, max_seq_len)
    print(f"Model without projection: {model_no_projection}")

    # forwardパスのテスト (プロジェクションなし)
    output_representation_no_proj = model_no_projection(item_seq, item_seq_len)
    print(f"Output Representation (no projection) Shape: {output_representation_no_proj.shape}") # 期待: (batch_size, hidden_size)
    assert output_representation_no_proj.shape == (batch_size, hidden_size)

    print("\nSASRec model test passed!")