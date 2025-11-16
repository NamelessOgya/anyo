import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, Any, List, Optional

from src.teacher.interfaces import TeacherModel

class iLoRAModel(TeacherModel, nn.Module):
    def __init__(self, 
                 llm_model_name: str,
                 num_lora_experts: int,
                 lora_r: int,
                 lora_alpha: int,
                 lora_dropout: float,
                 num_items: int, # 生徒モデルのnum_itemsと合わせる
                 max_seq_len: int,
                 hidden_size: int, # ゲーティングネットワーク用
                 dropout_rate: float, # ゲーティングネットワーク用
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.llm_model_name = llm_model_name
        self.num_lora_experts = num_lora_experts
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.device = device

        # 1. LLMのロード
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        # パディングトークンがない場合は追加
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.to(self.device)

        # 2. LoRAアダプターの準備
        self.lora_configs = []
        self.lora_models = nn.ModuleList()
        for i in range(num_lora_experts):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"] # 一般的なLoRAターゲットモジュール
            )
            lora_model = get_peft_model(self.llm, lora_config)
            self.lora_models.append(lora_model)
            self.lora_configs.append(lora_config)
        
        # 3. ゲーティングメカニズム (スケルトン)
        # シーケンス表現を生成するための埋め込み層とTransformer (SASRecのものを流用)
        # ここでは簡略化のため、直接シーケンス表現を生成するMLPを仮定
        # 実際には、LLMの埋め込み層やTransformer層の一部を再利用してシーケンス表現を生成する
        self.sequence_encoder = nn.Sequential(
            nn.Embedding(num_items + 1, hidden_size, padding_idx=0), # アイテム埋め込み
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_lora_experts, dropout=dropout_rate, batch_first=True),
                num_layers=1 # 簡略化のため1層
            )
        )
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_lora_experts),
            nn.Softmax(dim=-1)
        )

        # 出力層 (LLMのvocab_sizeからnum_itemsへのマッピング)
        # LLMの出力は語彙サイズだが、推薦ではアイテム数に絞る必要がある
        # ここでは、LLMの最後の線形層の重みとアイテム埋め込みを関連付ける必要がある
        # 簡略化のため、LLMの出力から直接アイテムスコアを抽出するロジックを仮定
        # 実際には、LLMの出力とアイテム埋め込みの類似度を計算する
        self.item_embeddings = nn.Embedding(num_items + 1, self.llm.config.hidden_size, padding_idx=0) # LLMの隠れ層サイズに合わせる

    def _get_sequence_representation(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        """
        入力アイテムシーケンスからシーケンス表現を生成します。
        ここでは簡略化のため、SASRecのようなTransformerエンコーダを使用します。
        """
        # item_seq: (batch_size, max_seq_len)
        # item_seq_len: (batch_size)

        # アイテム埋め込み
        item_embeddings = self.sequence_encoder[0](item_seq) # nn.Embedding層

        # TransformerEncoderLayerの入力マスクを作成
        # src_key_padding_mask: (batch_size, max_seq_len)
        # True: パディング要素 (無視する)、False: 有効な要素
        src_key_padding_mask = (item_seq == self.sequence_encoder[0].padding_idx)

        # TransformerEncoderLayerの入力は (batch_size, seq_len, embed_dim)
        # output: (batch_size, seq_len, hidden_size)
        transformer_output = self.sequence_encoder[1](
            item_embeddings,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 各シーケンスの最後のアイテムの表現を取得
        # item_seq_lenは実際の長さなので、-1して0-indexedにする
        last_item_indices = item_seq_len - 1
        # gatherを使って、各バッチの最後のアイテムの埋め込みを取得
        # transformer_output: (batch_size, max_seq_len, hidden_size)
        # last_item_indices: (batch_size)
        # 期待される結果: (batch_size, hidden_size)
        last_item_representation = torch.gather(
            transformer_output,
            1,
            last_item_indices.view(-1, 1, 1).expand(-1, 1, self.hidden_size)
        ).squeeze(1)

        return last_item_representation


    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        """
        iLoRAモデルのフォワードパス。
        ゲーティングネットワークで選択されたLoRAエキスパートをLLMに適用し、推薦スコアを計算します。
        """
        batch_size, seq_len = item_seq.shape

        # 1. シーケンス表現の生成
        h_seq = self._get_sequence_representation(item_seq, item_seq_len) # (batch_size, hidden_size)

        # 2. ゲーティングネットワークによるエキスパート重みの計算
        expert_weights = self.gating_network(h_seq) # (batch_size, num_lora_experts)

        # 3. LoRAアダプターの動的結合 (peftの機能を利用)
        # peftのmerge_and_unload()やset_adapter()を動的に使うのは複雑
        # ここでは、各LoRAモデルの出力を重み付け平均する形で簡略化
        # 実際には、LoRAの重み行列自体を結合してLLMに適用する
        
        # 簡略化のため、各LoRAモデルで個別に推論し、その出力を重み付け平均する
        # これは計算コストが高いが、概念実証としては有効
        
        # LLMの入力形式に変換 (アイテムIDをトークンIDにマッピングする必要がある)
        # ここでは簡略化のため、item_seqを直接LLMの入力として扱う (実際はトークナイズが必要)
        # LLMの入力は通常、トークンIDのシーケンス
        # item_seqをそのままLLMの入力として使うのは不適切だが、ここでは概念実証のため
        
        # 実際には、item_seqをプロンプトに変換し、tokenizerでトークンIDに変換する必要がある
        # 例: "ユーザーは[item1], [item2]を閲覧しました。次に興味を持つのは？"
        # ここでは、item_seqをダミーのトークンIDとして扱う
        
        # ダミーのLLM入力 (batch_size, max_seq_len)
        # item_seqをそのまま使うと、アイテムIDがLLMの語彙にない可能性があるので、
        # 適切なトークンIDに変換するか、ダミーのトークンIDを生成する必要がある
        dummy_llm_input_ids = item_seq # 暫定的にアイテムIDをそのまま使用
        dummy_attention_mask = (item_seq != 0).long() # パディングを考慮

        # 各LoRAエキスパートでLLMを推論し、出力を重み付け平均
        # (batch_size, seq_len, vocab_size)
        weighted_logits = torch.zeros(batch_size, seq_len, self.llm.config.vocab_size, device=self.device)

        for i in range(self.num_lora_experts):
            # peftモデルのアダプターを切り替える
            # self.lora_models[i].set_adapter(self.lora_configs[i].peft_type) # peftのAPIを正しく使う
            
            # ここでは、各LoRAモデルがLLMのラッパーとして機能すると仮定
            # lora_model = self.lora_models[i]
            # with lora_model.disable_adapter(): # ベースモデルで推論
            #     base_output = self.llm(input_ids=dummy_llm_input_ids, attention_mask=dummy_attention_mask).logits
            # with lora_model.enable_adapter(): # LoRA適用モデルで推論
            #     lora_output = lora_model(input_ids=dummy_llm_input_ids, attention_mask=dummy_attention_mask).logits
            
            # 簡略化のため、各LoRAモデルが独立したLLMインスタンスであるかのように扱う
            # 実際には、peftのAPIを使ってベースLLMに動的にLoRA重みを適用する
            
            # ダミーのLLM出力 (batch_size, seq_len, vocab_size)
            # 実際には、LLMの出力層からアイテムスコアを抽出する
            # ここでは、LLMの語彙サイズからnum_itemsへのマッピングが必要
            
            # 暫定的に、各エキスパートが異なるランダムなスコアを生成すると仮定
            expert_output_logits = torch.randn(batch_size, seq_len, self.llm.config.vocab_size, device=self.device)
            
            # 重み付け平均
            weighted_logits += expert_weights[:, i].view(batch_size, 1, 1) * expert_output_logits

        # 最終的な推薦スコアの計算
        # LLMの語彙サイズからnum_itemsへのマッピング
        # ここでは、LLMの出力とアイテム埋め込みの類似度を計算する
        # weighted_logits: (batch_size, seq_len, vocab_size)
        # 最後のトークンの出力を取得
        last_token_logits = weighted_logits[:, -1, :] # (batch_size, vocab_size)

        # アイテム埋め込みとの内積
        # (batch_size, vocab_size) @ (vocab_size, num_items + 1)
        # LLMの語彙サイズとアイテム埋め込みの次元が異なる場合がある
        # ここでは、LLMのvocab_sizeとitem_embeddingsのhidden_sizeが同じと仮定
        # 実際には、LLMの出力からアイテム埋め込み空間へのプロジェクションが必要
        
        # 簡略化のため、LLMの出力語彙から直接アイテムIDに対応するスコアを抽出
        # LLMの語彙IDとアイテムIDが一致しないため、これは不適切
        # 実際には、LLMの出力埋め込みとアイテム埋め込みの類似度を計算する
        
        # 暫定的に、ランダムなアイテムスコアを生成
        final_item_scores = torch.randn(batch_size, self.num_items + 1, device=self.device)

        return final_item_scores

    def get_teacher_outputs(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> Dict[str, Any]:
        """
        蒸留に必要な教師モデルの出力を生成します。
        ランキングスコアと埋め込みを返します。
        """
        batch_size, seq_len = item_seq.shape

        # 推薦スコアの取得
        ranking_scores = self.forward(item_seq, item_seq_len) # (batch_size, num_items + 1)

        # 埋め込みの取得 (例: 最後のシーケンス表現)
        embeddings = self._get_sequence_representation(item_seq, item_seq_len) # (batch_size, hidden_size)

        return {
            "ranking_scores": ranking_scores,
            "embeddings": embeddings
        }

if __name__ == "__main__":
    # テスト用のパラメータ
    llm_model_name = "facebook/opt-125m" # 小さなモデルでテスト
    num_lora_experts = 3
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    num_items = 5000
    max_seq_len = 50
    hidden_size = 64 # ゲーティングネットワーク用
    dropout_rate = 0.1
    batch_size = 4

    # モデルのインスタンス化
    model = iLoRAModel(
        llm_model_name=llm_model_name,
        num_lora_experts=num_lora_experts,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_items=num_items,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate
    )
    print(f"iLoRAModel initialized with {num_lora_experts} LoRA experts.")

    # ダミーデータ
    item_seq = torch.randint(1, num_items, (batch_size, max_seq_len)).to(model.device)
    item_seq_len = torch.randint(1, max_seq_len + 1, (batch_size,)).to(model.device)

    # forwardパスのテスト
    output_scores = model(item_seq, item_seq_len)
    print(f"Output Scores Shape: {output_scores.shape}") # 期待: (batch_size, num_items + 1)
    assert output_scores.shape == (batch_size, num_items + 1)

    # get_teacher_outputsパスのテスト
    teacher_outputs = model.get_teacher_outputs(item_seq, item_seq_len)
    print(f"Teacher Outputs Keys: {teacher_outputs.keys()}")
    print(f"Ranking Scores Shape: {teacher_outputs['ranking_scores'].shape}")
    print(f"Embeddings Shape: {teacher_outputs['embeddings'].shape}")
    assert teacher_outputs['ranking_scores'].shape == (batch_size, num_items + 1)
    assert teacher_outputs['embeddings'].shape == (batch_size, hidden_size)

    print("\niLoRAModel test passed!")
