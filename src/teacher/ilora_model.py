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
                 item_id_to_name: Dict[int, str], # アイテムIDから名前へのマッピング
                 padding_item_id: int = 0, # パディング用のアイテムID
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
        self.item_id_to_name = item_id_to_name
        self.padding_item_id = padding_item_id # 追加
        self.device = device

        # 1. LLMのロード
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        # パディングトークンがない場合は追加
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        base_llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        base_llm.resize_token_embeddings(len(self.tokenizer))
        base_llm.to(self.device)

        # 2. LoRAアダプターの準備
        # まず、ベースLLMをPeftModelでラップし、すべてのアダプターをそこに追加する
        lora_config_template = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"] # 一般的なLoRAターゲットモジュール
        )
        # ベースLLMをPeftModelでラップ (最初はアダプターなし)
        self.llm = get_peft_model(base_llm, lora_config_template, adapter_name="dummy_adapter")
        self.llm.delete_adapter("dummy_adapter") # ダミーアダプターは削除

        for i in range(num_lora_experts):
            adapter_name = f"expert_{i}"
            self.llm.add_adapter(adapter_name, lora_config_template)
        
        # 3. ゲーティングメカニズム (スケルトン)
        # シーケンス表現を生成するための埋め込み層とTransformer (SASRecのものを流用)
        # ここでは簡略化のため、直接シーケンス表現を生成するMLPを仮定
        # 実際には、LLMの埋め込み層やTransformer層の一部を再利用してシーケンス表現を生成する
        self.sequence_encoder = nn.Sequential(
            nn.Embedding(num_items + 1, hidden_size, padding_idx=0), # アイテム埋め込み
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dropout=dropout_rate, batch_first=True), # nheadをhidden_sizeの約数に固定
                num_layers=1 # 簡略化のため1層
            )
        ).to(self.device) # デバイスに移動
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_lora_experts),
            nn.Softmax(dim=-1)
        ).to(self.device) # デバイスに移動

        # 出力層 (LLMのvocab_sizeからnum_itemsへのマッピング)
        # LLMの出力は語彙サイズだが、推薦ではアイテム数に絞る必要がある
        # ここでは、LLMの最後の線形層の重みとアイテム埋め込みを関連付ける必要がある
        # 簡略化のため、LLMの出力から直接アイテムスコアを抽出するロジックを仮定
        # 実際には、LLMの出力とアイテム埋め込みの類似度を計算する
        self.item_embeddings = nn.Embedding(num_items + 1, self.llm.config.vocab_size, padding_idx=0).to(self.device) # LLMの語彙サイズに合わせる

    def _generate_prompt(self, item_seq: torch.Tensor) -> List[str]:
        """
        アイテムシーケンスからLLMへのプロンプトを生成します。
        例: "User has viewed [item_name_1], [item_name_2]. What would they be interested in next?"
        """
        prompts = []
        for seq in item_seq:
            # パディングを除外し、実際のアイテムIDのみを取得
            actual_items = [self.item_id_to_name.get(item_id.item(), f"Unknown Item {item_id.item()}") 
                            for item_id in seq if item_id.item() != self.padding_item_id]
            
            if not actual_items:
                prompts.append("The user has not viewed any items. What would they be interested in?")
            else:
                # ユーザーが閲覧したアイテムのリスト
                viewed_items_str = ", ".join(actual_items)
                prompt = f"The user has viewed the following items: {viewed_items_str}. Based on this history, what would be the most relevant item for them next? Please list only the item name."
                prompts.append(prompt)
        return prompts

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
        
        # プロンプトの生成
        prompts = self._generate_prompt(item_seq)
        
        # プロンプトのトークナイズ
        tokenized_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_seq_len, # LLMの入力長を制限
        ).to(self.device)
        
        llm_input_ids = tokenized_inputs["input_ids"]
        llm_attention_mask = tokenized_inputs["attention_mask"]

        # 3. LoRAアダプターの動的結合とLLM推論
        # 各バッチのexpert_weightsに基づいて、動的にLoRAアダプターを結合し、LLMを推論する
        
        # weighted_logitsを格納するテンソルを初期化
        weighted_logits = torch.zeros(batch_size, llm_input_ids.shape[1], self.llm.config.vocab_size, device=self.device)

        # アダプター名をリスト化
        adapter_names = [f"expert_{i}" for i in range(self.num_lora_experts)]

        for b in range(batch_size):
            # 現在のバッチサンプルのエキスパート重みを取得
            current_expert_weights = expert_weights[b] # (num_lora_experts,)

            # 重み付けされたアダプターを一時的に追加
            # add_weighted_adapterは新しいアダプターを作成する
            # combination_type="linear"で重み付け線形結合
            self.llm.add_weighted_adapter(
                adapters=adapter_names,
                weights=current_expert_weights.tolist(), # weightsはリストで渡す
                adapter_name="dynamic_adapter",
                combination_type="linear"
            )
            
            # 一時的に作成したアダプターをアクティブに設定
            self.llm.set_adapter("dynamic_adapter")

            # LLMのフォワードパスを実行
            with torch.no_grad(): # 推論時は勾配計算不要
                output = self.llm(
                    input_ids=llm_input_ids[b].unsqueeze(0), # バッチサイズ1で入力
                    attention_mask=llm_attention_mask[b].unsqueeze(0)
                ).logits
            
            # 出力をweighted_logitsに格納
            weighted_logits[b] = output.squeeze(0)

            # 一時アダプターを削除して次のバッチサンプルに備える
            self.llm.delete_adapter("dynamic_adapter")

        # LLMの出力は通常、最後のトークンに対するロジットが重要
        # weighted_logits: (batch_size, seq_len, vocab_size)
        # 最後のトークンの出力を取得
        last_token_logits = weighted_logits[:, -1, :] # (batch_size, vocab_size)

        # アイテム埋め込みとの内積
        # last_token_logits: (batch_size, vocab_size)
        # self.item_embeddings.weight: (num_items + 1, vocab_size)
        # final_item_scores: (batch_size, num_items + 1)
        final_item_scores = torch.matmul(last_token_logits, self.item_embeddings.weight.t())

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
