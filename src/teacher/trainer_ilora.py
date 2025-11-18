import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from typing import Dict, Any
import torch.nn.functional as F # Import F

from src.teacher.ilora_model import iLoRAModel
from src.core.metrics import calculate_metrics

class iLoRATrainer(pl.LightningModule):
    def __init__(self, 
                 ilora_model: iLoRAModel,
                 num_items: int,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 metrics_k: int = 10,
                 item_id_to_name: Dict[int, str] = None): # item_id_to_nameを追加
        super().__init__()
        # ilora_modelは複雑なオブジェクトなのでignoreする
        # item_id_to_nameも直接保存せず、ilora_model経由でアクセスする
        self.save_hyperparameters(ignore=['ilora_model', 'item_id_to_name']) 

        self.model = ilora_model
        self.metrics_k = metrics_k
        self.loss_fn = F.cross_entropy # Use F.cross_entropy directly
        # self.item_id_to_name = item_id_to_name # ilora_modelが持つので不要

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        # iLoRAModel.forward returns outputs object from LLM
        return self.model(batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch)
        
        # Extract last hidden state from LLM outputs
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        
        # Project last hidden state to num_items logits using item_prediction_head
        logits = self.model.item_prediction_head(last_hidden_state)
        
        # Get next_item from batch for loss calculation
        next_item = batch["next_item"].squeeze(-1) # Squeeze to (batch_size,)
        
        # Calculate loss using the projected logits and next_item
        loss = self.loss_fn(logits, next_item)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        logits = self.model.item_prediction_head(last_hidden_state)
        next_item = batch["next_item"].squeeze(-1) # Squeeze to (batch_size,)
        loss = self.loss_fn(logits, next_item)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Metric calculation
        _, predicted_indices = torch.topk(logits, k=self.metrics_k, dim=1)
        predictions = predicted_indices.tolist()
        
        ground_truths_list = []
        for item_id_tensor in next_item:
            item_id = item_id_tensor.item()
            # Assuming padding_item_id is not present here or handled by the datamodule
            ground_truths_list.append([item_id])
        
        metrics = calculate_metrics(predictions, ground_truths_list, k=self.metrics_k)
        for metric_name, metric_value in metrics.items():
            self.log(f"val_{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"val_loss": loss}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :]
        logits = self.model.item_prediction_head(last_hidden_state)
        next_item = batch["next_item"].squeeze(-1) # Squeeze to (batch_size,)
        loss = self.loss_fn(logits, next_item)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Metric calculation
        _, predicted_indices = torch.topk(logits, k=self.metrics_k, dim=1)
        predictions = predicted_indices.tolist()
        
        ground_truths_list = []
        for item_id_tensor in next_item:
            item_id = item_id_tensor.item()
            # Assuming padding_item_id is not present here or handled by the datamodule
            ground_truths_list.append([item_id])

        metrics = calculate_metrics(predictions, ground_truths_list, k=self.metrics_k)
        for metric_name, metric_value in metrics.items():
            self.log(f"test_{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {"test_loss": loss}

    def configure_optimizers(self) -> Any:
        # iLoRAでは、LoRAアダプターとゲーティングネットワークのパラメータのみを学習対象とする
        # LLMのベースモデルのパラメータはフリーズされている
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

if __name__ == "__main__":
    # テスト用のダミーデータとデータモジュール
    from src.student.datamodule import SASRecDataModule # データモジュールは共通
    from omegaconf import OmegaConf
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor
    from transformers import AutoModelForCausalLM, AutoTokenizer # LLMとTokenizerをロードするために追加

    # iLoRAModelのダミー設定
    ilora_cfg = OmegaConf.create({
        "llm_model_name": "facebook/opt-125m",
        "num_lora_experts": 3,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "hidden_size": 64,
        "dropout_rate": 0.1
    })

    # LLMとTokenizerをロード
    llm = AutoModelForCausalLM.from_pretrained(ilora_cfg.llm_model_name)
    tokenizer = AutoTokenizer.from_pretrained(ilora_cfg.llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[PH]','[HistoryEmb]','[CansEmb]','[ItemEmb]']})
    llm.resize_token_embeddings(len(tokenizer))

    # ダミーデータモジュール
    dm = SASRecDataModule(
        batch_size=4, 
        max_seq_len=50, 
        num_workers=0,
        llm_model_name=ilora_cfg.llm_model_name,
        tokenizer=tokenizer,
        max_gen_length=64
    )
    dm.prepare_data()
    dm.setup()

    # ダミーのrec_modelとprojectorを作成
    class DummyRecModel(nn.Module):
        def __init__(self, hidden_size_rec, num_items_rec):
            super().__init__()
            self.item_embeddings = nn.Embedding(num_items_rec + 1, hidden_size_rec)
            self.cacu_x = lambda x: self.item_embeddings(x)
            self.cacul_h = lambda x, y: torch.randn(x.shape[0], hidden_size_rec)
    
    dummy_rec_model = DummyRecModel(ilora_cfg.hidden_size, dm.num_items).to(llm.device)
    dummy_projector = MLPProjector(
        input_dim=ilora_cfg.hidden_size,
        output_dim=llm.config.hidden_size,
        hidden_size=ilora_cfg.hidden_size,
        dropout_rate=ilora_cfg.dropout_rate
    ).to(llm.device)

    # iLoRAModelのインスタンス化
    ilora_model_instance = iLoRAModel(
        llm=llm,
        tokenizer=tokenizer,
        num_lora_experts=ilora_cfg.num_lora_experts,
        lora_r=ilora_cfg.lora_r,
        lora_alpha=ilora_cfg.lora_alpha,
        lora_dropout=ilora_cfg.lora_dropout,
        num_items=dm.num_items,
        max_seq_len=dm.max_seq_len,
        hidden_size=ilora_cfg.hidden_size,
        dropout_rate=ilora_cfg.dropout_rate,
        item_id_to_name=dm.item_id_to_name,
        padding_item_id=tokenizer.pad_token_id,
        rec_model=dummy_rec_model,
        projector=dummy_projector,
        candidate_topk=10
    )

    # iLoRATrainerのインスタンス化
    ilora_trainer = iLoRATrainer(
        ilora_model=ilora_model_instance,
        num_items=dm.num_items,
        learning_rate=1e-4,
        weight_decay=0.01,
        metrics_k=10
    )

    # PyTorch Lightning Trainer
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu", # テスト用にCPUを使用
        devices=1,
        logger=False, # テスト時はロガーを無効化
        callbacks=[lr_monitor],
        enable_checkpointing=False,
    )

    print("\n--- Starting dummy iLoRA training ---")
    trainer.fit(ilora_trainer, dm.train_dataloader(), dm.val_dataloader())
    print("Dummy iLoRA training finished.")

    print("\n--- Starting dummy iLoRA testing ---")
    trainer.test(ilora_trainer, dm.test_dataloader())
    print("Dummy iLoRA testing finished.")
