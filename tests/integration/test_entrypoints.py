import pytest
from unittest.mock import MagicMock, patch
from omegaconf import DictConfig

# Import the run_experiment functions
# Note: We need to make sure imports in the scripts don't trigger side effects (like hydra.main execution)
# Since we refactored main to call run_experiment, importing the module is safe if main is guarded by if __name__ == "__main__".

from src.exp.run_teacher import run_experiment as run_teacher_exp
from src.exp.run_bigrec import run_experiment as run_bigrec_exp
from src.exp.run_distill import run_experiment as run_distill_exp

@pytest.fixture
def mock_cfg():
    cfg = MagicMock()
    # Common config structure
    cfg.run.dir = "tmp_test_dir"
    cfg.seed = 42
    cfg.dataset.name = "movielens"
    cfg.dataset.data_dir = "dummy_dir"
    cfg.dataset.limit_data_rows = 10
    cfg.train.batch_size = 2
    cfg.train.num_workers = 0
    cfg.train.max_epochs = 1
    cfg.train.accelerator = "cpu"
    cfg.train.devices = 1
    cfg.train.val_check_interval = 1.0
    cfg.train.log_every_n_steps = 1
    cfg.train.precision = "32"
    cfg.train.accumulate_grad_batches = 1
    cfg.train.learning_rate = 1e-3
    cfg.train.weight_decay = 0.0
    
    cfg.student.max_seq_len = 10
    cfg.student.hidden_size = 16
    cfg.student.num_heads = 2
    cfg.student.num_layers = 2
    cfg.student.dropout_rate = 0.1
    
    cfg.teacher.llm_model_name = "dummy-llm"
    cfg.teacher.lora_r = 4
    cfg.teacher.lora_alpha = 16
    cfg.teacher.lora_dropout = 0.1
    cfg.teacher.learning_rate = 1e-4
    cfg.teacher.max_source_length = 20
    cfg.teacher.max_target_length = 10
    cfg.teacher.batch_size = 2
    
    cfg.distill.candidate_topk = 5
    cfg.distill.teacher_checkpoint_path = None
    cfg.distill.teacher_outputs_batches_dir = "dummy_out"
    cfg.distill.ranking_loss_weight = 1.0
    cfg.distill.embedding_loss_weight = 1.0
    cfg.distill.ce_loss_weight = 1.0
    cfg.distill.ranking_temperature = 1.0
    cfg.distill.embedding_loss_type = "mse"
    cfg.distill.gamma_position = 1.0
    cfg.distill.gamma_confidence = 1.0
    cfg.distill.ed_weight = 1.0
    cfg.distill.alpha = 0.5
    cfg.distill.beta = 0.5
    cfg.distill.lam = 0.5
    cfg.distill.num_neg_samples = 1
    cfg.distill.ps_power = 1.0

    cfg.eval.metrics_k = [10]
    
    # Mock get method for DictConfig
    def get_side_effect(key, default=None):
        if key == "model_type": return "bigrec" # Default for test
        if key == "use_cot": return False
        if key == "distill_lambda": return 0.0
        if key == "distill_loss_type": return "mse"
        if key == "distill_decay_type": return "none"
        if key == "distill_min_lambda": return 0.0
        if key == "distill_decay_steps": return None
        if key == "subset_indices_path": return None
        if key == "type": return "dllm2rec"
        return default
    
    cfg.teacher.get.side_effect = get_side_effect
    cfg.distill.get.side_effect = get_side_effect
    
    return cfg

@patch("src.exp.run_teacher.SASRecDataModule")
@patch("src.teacher.bigrec_model.BigRecModel")
@patch("src.data.collators.BigRecCollator")
@patch("src.exp.run_teacher.pl.Trainer")
@patch("src.exp.run_teacher.AutoTokenizer")
@patch("src.exp.run_teacher.setup_logging")
@patch("src.exp.run_teacher.set_seed")
@patch("src.exp.run_teacher.OmegaConf.to_yaml", return_value="dummy")
@patch("src.exp.run_teacher.subprocess.run")
@patch("src.exp.run_teacher.DataLoader")
@patch("src.exp.run_teacher.get_git_info")
def test_run_teacher_bigrec(mock_git, mock_dl, mock_subprocess, mock_omegaconf, mock_seed, mock_log, mock_tok, mock_trainer, mock_collator, mock_model, mock_dm, mock_cfg):
    # Setup mocks
    dm_instance = MagicMock()
    dm_instance.mapped_id_to_title = {1: "Item1"}
    # We don't delete item_id_to_name because MagicMock creates it on access.
    # Instead, we verify that the value passed to collator is indeed mapped_id_to_title.
    
    mock_dm.return_value = dm_instance
    
    mock_cfg.teacher.get.side_effect = lambda k, d=None: "bigrec" if k == "model_type" else d
    
    # Run
    run_teacher_exp(mock_cfg)
    
    # Verify
    mock_collator.assert_called()
    call_args = mock_collator.call_args
    assert call_args is not None
    # Check if item_id_to_name arg in BigRecCollator init was passed the mapped_id_to_title object
    assert call_args.kwargs['item_id_to_name'] == dm_instance.mapped_id_to_title

@patch("src.exp.run_bigrec.SASRecDataModule")
@patch("src.exp.run_bigrec.BigRecModel")
@patch("src.exp.run_bigrec.BigRecCollator")
@patch("src.exp.run_bigrec.DataLoader")
@patch("src.exp.run_bigrec.pl.Trainer")
def test_run_bigrec(mock_trainer, mock_dl, mock_collator, mock_model, mock_dm, mock_cfg):
    dm_instance = MagicMock()
    dm_instance.mapped_id_to_title = {1: "Item1"}
    mock_dm.return_value = dm_instance
    
    run_bigrec_exp(mock_cfg)
    
    call_args = mock_collator.call_args
    assert call_args.kwargs['item_id_to_name'] == dm_instance.mapped_id_to_title

@patch("src.exp.run_distill.SASRecDataModule")
@patch("src.exp.run_distill.create_teacher_model")
@patch("src.exp.run_distill.iLoRATrainer")
@patch("src.exp.run_distill.DistillationTrainer")
@patch("src.exp.run_distill.pl.Trainer")
@patch("src.exp.run_distill.PropensityScoreCalculator")
@patch("src.exp.run_distill.TeacherOutputDataset")
@patch("src.exp.run_distill.DataLoader")
@patch("src.exp.run_distill.SASRec")
@patch("src.exp.run_distill.setup_logging")
@patch("src.exp.run_distill.set_seed")
@patch("src.exp.run_distill.OmegaConf.to_yaml", return_value="dummy")
@patch("src.exp.run_distill.get_git_info")
def test_run_distill_dllm2rec(mock_git, mock_omegaconf, mock_seed, mock_log, mock_sasrec, mock_dl, mock_tod, mock_psc, mock_trainer, mock_dt, mock_ilora, mock_create_teacher, mock_dm, mock_cfg):
    dm_instance = MagicMock()
    dm_instance.mapped_id_to_title = {1: "Item1"}
    dm_instance.num_items = 10
    dm_instance.padding_item_id = 0
    mock_dm.return_value = dm_instance
    
    # Mock config for dllm2rec
    mock_cfg.distill.get.side_effect = lambda k, d=None: "dllm2rec" if k == "type" else d
    mock_cfg.distill.teacher_checkpoint_path = "runs/exp/checkpoints/dummy.ckpt"
    
    with patch("pathlib.Path.exists", return_value=True):
        # Mock loading teacher config
        with patch("src.exp.run_distill.OmegaConf.load") as mock_load:
            mock_load.return_value = mock_cfg # Reuse mock cfg as teacher cfg
            
            run_distill_exp(mock_cfg)
    
    # Verify DistillationTrainer init
    call_args = mock_dt.call_args
    assert call_args.kwargs['item_id_to_name'] == dm_instance.mapped_id_to_title
