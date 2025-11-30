import torch
from src.student.models import SASRec
from src.student.datamodule import SASRecDataModule
from src.student.evaluator import SASRecEvaluator
import pytorch_lightning as pl
import os

def verify_sasrec():
    print("Verifying SASRec Performance...")
    
    # 1. Setup DataModule
    data_dir = "data/ml-100k"
    if not os.path.exists(data_dir):
        print(f"Data dir {data_dir} not found!")
        return

    # Use a small batch size and limit rows for quick check
    dm = SASRecDataModule(
        dataset_name="ml-100k",
        data_dir=data_dir,
        batch_size=32,
        max_seq_len=50,
        num_workers=0,
        limit_data_rows=None # Use full data to get realistic item count
    )
    dm.prepare_data()
    dm.setup()
    
    print(f"Num Items: {dm.num_items}")
    print(f"Train Size: {len(dm.train_dataset)}")
    print(f"Val Size: {len(dm.val_dataset)}")
    
    # 2. Setup Model
    # Use standard params
    model = SASRec(
        num_items=dm.num_items,
        hidden_size=64,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        max_seq_len=50,
        padding_item_id=dm.padding_item_id
    )
    
    # 3. Quick Train Loop (1 epoch on small subset)
    # We can't easily run full training here, but we can check if it learns AT ALL.
    # Or we can just check the EVALUATOR logic.
    
    # Let's check Evaluator Logic first.
    # If we feed random scores, what is the metric?
    # Expected: ~10/1682 = 0.006
    
    evaluator = SASRecEvaluator(model, dm, metrics_k=10)
    
    print("\n--- Evaluating Initial Model (Random Weights) ---")
    # Use a subset of val_loader for speed
    val_loader = dm.val_dataloader()
    # Take first 10 batches
    # We can't easily slice DataLoader, so we iterate
    
    # But wait, the user says 0.09 after 20 epochs.
    # If random is 0.006, then 0.09 is learning SOMETHING.
    # But standard SASRec on ML-100k should be much higher.
    
    # Check if we are evaluating on ALL items or sampled negatives?
    # The predict() method calculates scores for ALL items:
    # scores = torch.matmul(last_item_representation_for_prediction, valid_item_embeds.transpose(0, 1))
    # This is Full Ranking.
    
    # ML-100k HR@10 for Full Ranking is typically ~0.2 - 0.3?
    # iLoRA paper reports HR@10 ~ 0.28 for SASRec on ML-1M.
    # ML-100k is smaller, maybe easier? Or harder due to sparsity?
    # Usually ML-100k HR@10 is around 0.6-0.7 (if leave-one-out).
    
    # Let's check if the DataModule split is correct.
    # SASRecDataModule uses:
    # train_df = pd.read_csv(..., nrows=...)
    # val_df = pd.read_csv(..., nrows=...)
    # test_df = pd.read_csv(..., nrows=...)
    
    # And:
    # self.train_df = self.train_df[self.train_df['seq'].apply(len) >= 3]
    
    # And:
    # combined_df = pd.concat(...)
    # self.item_id_map = ...
    
    # If the user says 0.09, maybe they are using HR@1?
    # metrics_k is set to 10 in config.
    
    # Let's run a quick evaluation on random weights to confirm baseline.
    metrics = evaluator.evaluate(val_loader)
    print(f"Random Weights Metrics: {metrics}")
    
    # Check if we can overfit a single batch
    print("\n--- Overfitting Single Batch ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    seq = batch["seq"]
    len_seq = batch["len_seq"]
    next_item = batch["next_item"] # Target IDs
    
    model.train()
    for i in range(50):
        optimizer.zero_grad()
        # Forward
        # SASRec forward returns last_item_representation
        # But we need logits for CrossEntropy
        # predict() returns logits
        logits = model.predict(seq, len_seq)
        
        # CrossEntropyLoss expects (Batch, NumClasses) and (Batch)
        # logits: (Batch, NumItems+1)
        # next_item: (Batch)
        
        loss = criterion(logits, next_item)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss.item()}")
            
    # Check metrics on this batch
    model.eval()
    logits = model.predict(seq, len_seq)
    _, predicted_indices = torch.topk(logits, 10, dim=-1)
    
    hits = 0
    for j in range(len(next_item)):
        if next_item[j] in predicted_indices[j]:
            hits += 1
            
    hr = hits / len(next_item)
    print(f"Overfit Batch HR@10: {hr}")
    
    if hr < 0.8:
        print("WARNING: Failed to overfit single batch! Model or Data issue.")
    else:
        print("Success: Model can overfit single batch.")

if __name__ == "__main__":
    verify_sasrec()
