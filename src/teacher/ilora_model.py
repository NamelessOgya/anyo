from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F # 追加
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import random

from src.teacher.mlp_projector import MLPProjector # 修正されたインポートパス

class iLoRAModel(nn.Module):
    def __init__(self, 
                 llm: AutoModelForCausalLM, 
                 tokenizer: AutoTokenizer,
                 num_items: int,
                 max_seq_len: int,
                 num_lora_experts: int,
                 lora_r: int,
                 lora_alpha: int,
                 lora_dropout: float,
                 hidden_size: int, # ゲーティングネットワークの隠れ層サイズ
                 dropout_rate: float,
                 item_id_to_name: Dict[int, str],
                 padding_item_id: int):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.num_lora_experts = num_lora_experts # 追加
        self.padding_item_id = padding_item_id
        self.item_id_to_name = item_id_to_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LLMのLoRAアダプターを準備
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"] # OPTモデルのAttention層のクエリとバリューに適用
        )
        self.peft_models = nn.ModuleList()
        for _ in range(num_lora_experts):
            peft_model = get_peft_model(self.llm, self.peft_config)
            self.peft_models.append(peft_model)
        
        # アイテム埋め込み層
        self.item_embeddings = nn.Embedding(num_items + 1, hidden_size, padding_idx=0).to(self.device)
        # アイテム埋め込みをLLMの入力埋め込み次元に合わせるためのプロジェクション層
        self.item_embedding_projection = nn.Linear(hidden_size, self.llm.config.hidden_size).to(self.device)

        # ゲーティングネットワーク
        self.gating_network = MLPProjector( # 直接インスタンス化
            input_dim=self.llm.config.hidden_size, # LLMの隠れ状態の次元
            output_dim=num_lora_experts,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate
        ).to(self.device)

        # 出力層 (LLMの最終隠れ状態からアイテムロジットを予測)
        self.output_layer = nn.Linear(self.llm.config.hidden_size, num_items + 1).to(self.device)

    def _get_item_names(self, item_ids: torch.Tensor) -> List[str]:
        """アイテムIDのテンソルをアイテム名のリストに変換する"""
        return [self.item_id_to_name.get(item_id.item(), "[UNK]") for item_id in item_ids]

    def _generate_prompt(self, item_seq_names: List[str]) -> str:
        """アイテム名のシーケンスからプロンプトを生成する"""
        # 例: "User has watched: MovieA, MovieB, MovieC. Predict the next movie:"
        return f"User has interacted with: {', '.join(item_seq_names)}. Predict the next item:"

    def forward(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        # item_seq: (batch_size, max_seq_len)
        # item_seq_len: (batch_size,)

        batch_size = item_seq.shape[0]

        # 各シーケンスの最後のアイテムの埋め込みを取得 (ゲーティングネットワークの入力)
        # item_seq_lenは実際のシーケンス長なので、それを使って最後のアイテムを特定
        last_item_ids = item_seq[torch.arange(batch_size), item_seq_len - 1]
        last_item_embeds = self.item_embedding_projection(self.item_embeddings(last_item_ids)) # (batch_size, llm_hidden_size)

        # ゲーティングネットワークで各LoRAエキスパートの重みを予測
        gate_weights = F.softmax(self.gating_network(last_item_embeds), dim=-1) # (batch_size, num_lora_experts)

        # プロンプトの準備
        input_texts = []
        for i in range(batch_size):
            current_item_ids = item_seq[i, :item_seq_len[i]]
            item_seq_names = self._get_item_names(current_item_ids)
            input_texts.append(self._generate_prompt(item_seq_names))

        # トークナイズ
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len).to(self.device)

        # 各LoRAエキスパートで推論を実行し、重み付け和を取る
        combined_logits = torch.zeros(batch_size, self.num_items + 1, device=self.device) # 最終的なアイテムロジット

        for i in range(self.num_lora_experts):
            peft_model = self.peft_models[i]
            
            # LLMのフォワードパス
            outputs = peft_model(**inputs, output_hidden_states=True)
            
            # 最終トークンの隠れ状態を取得
            last_hidden_state = outputs.hidden_states[-1][:, -1, :] # (batch_size, llm_hidden_size)

            # 出力層でアイテムロジットを予測
            expert_logits = self.output_layer(last_hidden_state) # (batch_size, num_items + 1)

            # ゲーティングネットワークの重みを適用
            # gate_weightsは(batch_size, num_lora_experts)なので、expert_logitsにブロードキャストして適用
            combined_logits += expert_logits * gate_weights[:, i].unsqueeze(1)

        return combined_logits

    def get_teacher_outputs(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor) -> Dict[str, Any]:
        """
        教師モデルの出力（ランキングスコアと埋め込み）を返す。
        蒸留プロセスで使用される。
        """
        batch_size = item_seq.shape[0]

        # 各シーケンスの最後のアイテムの埋め込みを取得 (ゲーティングネットワークの入力)
        last_item_ids = item_seq[torch.arange(batch_size), item_seq_len - 1]
        last_item_embeds = self.item_embedding_projection(self.item_embeddings(last_item_ids)) # (batch_size, llm_hidden_size)

        # ゲーティングネットワークで各LoRAエキスパートの重みを予測
        gate_weights = F.softmax(self.gating_network(last_item_embeds), dim=-1) # (batch_size, num_lora_experts)

        # プロンプトの準備
        input_texts = []
        for i in range(batch_size):
            current_item_ids = item_seq[i, :item_seq_len[i]]
            item_seq_names = self._get_item_names(current_item_ids)
            input_texts.append(self._generate_prompt(item_seq_names))

        # トークナイズ
        inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len).to(self.device)

        combined_logits = torch.zeros(batch_size, self.num_items + 1, device=self.device)
        combined_hidden_states = torch.zeros(batch_size, self.llm.config.hidden_size, device=self.device)

        for i in range(self.num_lora_experts):
            peft_model = self.peft_models[i]
            
            # LLMのフォワードパス
            outputs = peft_model(**inputs, output_hidden_states=True)
            
            # 最終トークンの隠れ状態を取得
            last_hidden_state = outputs.hidden_states[-1][:, -1, :] # (batch_size, llm_hidden_size)

            # 出力層でアイテムロジットを予測
            expert_logits = self.output_layer(last_hidden_state) # (batch_size, num_items + 1)

            # ゲーティングネットワークの重みを適用
            combined_logits += expert_logits * gate_weights[:, i].unsqueeze(1)
            combined_hidden_states += last_hidden_state * gate_weights[:, i].unsqueeze(1)

        return {
            "ranking_scores": combined_logits,
            "embeddings": combined_hidden_states
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

    # ダミーのitem_id_to_nameを作成
    dummy_item_id_to_name = {i: f"Item {i}" for i in range(num_items + 1)}

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
        dropout_rate=dropout_rate,
        item_id_to_name=dummy_item_id_to_name # 追加
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
