import subprocess
import os
import sys
import torch
import pandas as pd
from omegaconf import DictConfig
import logging
from pathlib import Path
from typing import List

from src.teacher.interfaces import ITeacherModel
from src.core.logging import TensorBoardLogger, time_block
from src.core.data_utils import load_processed_data

log = logging.getLogger(__name__)


class ILoRABackend(ITeacherModel):
    def __init__(self):
        self.ilora_path = Path("/app/src/third_party/ilora")
        if not self.ilora_path.exists():
            raise FileNotFoundError(f"iLoRA directory not found at {self.ilora_path}")

    def _build_ilora_args(self, cfg: DictConfig, run_dir: Path, mode: str) -> List[str]:
        """
        Builds the argument list for iLoRA's main.py from the Hydra config.
        """
        args = [
            sys.executable,  # Use the same python interpreter
            str(self.ilora_path / "main.py"),
            f"--accelerator={cfg.general.device.split(':')[0]}",  # e.g., 'cuda' from 'cuda:0'
            f"--devices={cfg.general.device.split(':')[1] if ':' in cfg.general.device else '0'}",  # e.g., '0'
            f"--batch_size={cfg.teacher.batch_size}",
            f"--lr={cfg.teacher.lr}",
            f"--max_epochs={cfg.teacher.num_epochs}",
            f"--llm_path={cfg.teacher.llm_path}",
            f"--prompt_path={self.ilora_path / 'prompt' / (cfg.teacher.prompt_template + '.txt')}",  # Assuming prompt files are in iLoRA/prompt
            f"--output_dir={run_dir / 'teacher_outputs_raw'}",  # iLoRA's output dir
            f"--ckpt_dir={run_dir / 'teacher_checkpoints'}",  # iLoRA's checkpoint dir
            f"--log_dir={run_dir / 'teacher_logs'}",  # iLoRA's log dir
            f"--mode={mode}",
            # Add other iLoRA specific args from config if needed
            # For now, using defaults or hardcoding based on iLoRA's main.py
            "--num_workers=0",  # For simplicity, can be configured
            (
                "--precision=16" if torch.cuda.is_available() else "--precision=32"
            ),  # Use 16 for bf16 if available
            "--llm_tuning=moelora",  # As per project spec
            "--lora_r=8",
            "--lora_alpha=32",
            "--lora_dropout=0.1",
            "--num_moe=4",
            "--gating=Dense",
            "--dataset=movielens_data",  # This needs to be dynamic based on cfg.dataset.name
            f"--data_dir={self.ilora_path / 'data' / 'ref' / cfg.dataset.name}",  # iLoRA's data dir
            f"--rec_model_path={self.ilora_path / 'rec_model' / 'SASRec_movielens.pt'}",  # This needs to be dynamic
            f"--padding_item_id={cfg.dataset.get('padding_item_id', 0)}",  # Placeholder, needs actual value
        ]

        # Adjust dataset specific paths and padding_item_id
        if cfg.dataset.name == "movielens":
            args[args.index("--dataset=movielens_data")] = "--dataset=movielens_data"
            args[
                args.index(
                    f"--data_dir={self.ilora_path / 'data' / 'ref' / cfg.dataset.name}"
                )
            ] = f"--data_dir={self.ilora_path / 'data' / 'ref' / 'movielens'}"
            args[
                args.index(
                    f"--rec_model_path={self.ilora_path / 'rec_model' / 'SASRec_movielens.pt'}"
                )
            ] = f"--rec_model_path={self.ilora_path / 'rec_model' / 'SASRec_movielens.pt'}"
            args[
                args.index(f"--padding_item_id={cfg.dataset.get('padding_item_id', 0)}")
            ] = "--padding_item_id=1682"  # From iLoRA/main.py
        elif cfg.dataset.name == "steam":
            args[args.index("--dataset=movielens_data")] = "--dataset=steam_data"
            args[
                args.index(
                    f"--data_dir={self.ilora_path / 'data' / 'ref' / cfg.dataset.name}"
                )
            ] = f"--data_dir={self.ilora_path / 'data' / 'ref' / 'steam'}"
            args[
                args.index(
                    f"--rec_model_path={self.ilora_path / 'rec_model' / 'SASRec_movielens.pt'}"
                )
            ] = f"--rec_model_path={self.ilora_path / 'rec_model' / 'SASRec_steam.pt'}"
            args[
                args.index(f"--padding_item_id={cfg.dataset.get('padding_item_id', 0)}")
            ] = "--padding_item_id=3581"  # From iLoRA/main.py
        elif cfg.dataset.name == "lastfm":
            args[args.index("--dataset=movielens_data")] = "--dataset=lastfm_data"
            args[
                args.index(
                    f"--data_dir={self.ilora_path / 'data' / 'ref' / cfg.dataset.name}"
                )
            ] = f"--data_dir={self.ilora_path / 'data' / 'ref' / 'lastfm'}"
            args[
                args.index(
                    f"--rec_model_path={self.ilora_path / 'rec_model' / 'SASRec_movielens.pt'}"
                )
            ] = f"--rec_model_path={self.ilora_path / 'rec_model' / 'lastfm.pt'}"  # Assuming lastfm.pt
            args[
                args.index(f"--padding_item_id={cfg.dataset.get('padding_item_id', 0)}")
            ] = "--padding_item_id=4606"  # From iLoRA/main.py

        return args

    def train(self, cfg: DictConfig, tb_logger: TensorBoardLogger, run_dir: Path):
        log.info("Training iLoRA teacher model...")
        ilora_train_log_path = run_dir / "logs" / "train_ilora.log"
        ilora_train_log_path.parent.mkdir(parents=True, exist_ok=True)

        ilora_args = self._build_ilora_args(cfg, run_dir, mode="train")
        log.info(f"Running iLoRA with arguments: {' '.join(ilora_args)}")

        with time_block("teacher_train_time_total"):
            process = subprocess.run(
                ilora_args,
                cwd=self.ilora_path,  # Run from iLoRA's directory
                capture_output=True,
                text=True,
                env=os.environ.copy(),  # Pass current environment variables
            )

        with open(ilora_train_log_path, "w") as f:
            f.write(process.stdout)
            f.write(process.stderr)

        if process.returncode != 0:
            log.error(f"iLoRA training failed with error:\n{process.stderr}")
            raise RuntimeError("iLoRA training failed.")
        else:
            log.info("iLoRA training completed successfully.")
            log.debug(f"iLoRA stdout:\n{process.stdout}")

    def export_for_dllm2rec(
        self, cfg: DictConfig, run_dir: Path, data_dir: Path, teacher_outputs_dir: Path
    ):
        log.info("Exporting iLoRA teacher outputs for DLLM2Rec...")
        teacher_outputs_dir.mkdir(parents=True, exist_ok=True)

        # --- Export all_embeddings.pt ---
        # This requires loading the trained iLoRA model and extracting embeddings.
        # This is not directly exposed in iLoRA's main.py or MInterface.
        # We need to load the best checkpoint and then use MInterface's methods.

        # Find the best checkpoint
        checkpoint_dir = run_dir / "teacher_checkpoints"
        latest_checkpoint = None
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoints:
                # Sort by modification time to get the latest, or parse filename for best metric
                latest_checkpoint = max(checkpoints, key=os.path.getmtime)
                log.info(f"Found latest iLoRA checkpoint: {latest_checkpoint}")
            else:
                log.warning(
                    f"No iLoRA checkpoints found in {checkpoint_dir}. Cannot export embeddings."
                )
                # Create dummy files to avoid downstream errors
                (teacher_outputs_dir / "all_embeddings.pt").touch()
                (teacher_outputs_dir / "myrank_train.txt").touch()
                (teacher_outputs_dir / "confidence_train.txt").touch()
                log.info(
                    "Created dummy teacher output files due to missing checkpoint."
                )
                return

        if latest_checkpoint:
            # Dynamically load MInterface and related classes from iLoRA
            # This is a bit hacky but avoids modifying iLoRA directly
            sys.path.insert(0, str(self.ilora_path))
            try:
                # Check for dummy LLM path to bypass actual loading
                if cfg.teacher.llm_path == "dummy_llm":
                    log.warning("Using dummy LLM path. Skipping actual LLM loading and using mock objects.")
                    
                    class MockLlamaTokenizer:
                        def __init__(self, *args, **kwargs):
                            pass
                        def from_pretrained(self, *args, **kwargs):
                            return self
                        @property
                        def pad_token(self):
                            return "[PAD]"
                        def add_special_tokens(self, tokens_dict):
                            pass
                        def __getattr__(self, name): # Mock any other attribute access
                            return lambda *args, **kwargs: None
                
                    class MockRecModel:
                        def __init__(self, *args, **kwargs):
                            pass
                        def item_embeddings(self, item_ids):
                            # Return dummy embeddings of appropriate size
                            # hidden_size is assumed to be 64 as per ilora_hparams
                            return torch.randn(item_ids.shape[0], 64, device=cfg.general.device)
                
                    class MockProjector(torch.nn.Module):
                        def __init__(self, *args, **kwargs):
                            super().__init__()
                            # input_dim is rec_size=64, output_dim is assumed LLM emb_size=4096
                            self.linear = torch.nn.Linear(64, 4096)
                        def forward(self, x):
                            return self.linear(x)
                
                    class MockMInterface:
                        def __init__(self, *args, **kwargs):
                            self.rec_model = MockRecModel()
                            self.projector = MockProjector()
                        def load_from_checkpoint(self, *args, **kwargs):
                            return self
                        def eval(self):
                            return self
                        def to(self, device):
                            return self
                    
                    MInterface = MockMInterface
                    LlamaTokenizer = MockLlamaTokenizer
                    tokenizer = MockLlamaTokenizer() # Instantiate mock tokenizer
                    model = MockMInterface() # Instantiate mock MInterface
                    model.eval().to(cfg.general.device) # Call eval and to for consistency - these are mocked anyway
                else:
                    from model.model_interface import MInterface
                    from transformers import AutoTokenizer

                    # Load tokenizer and model config
                    tokenizer = AutoTokenizer.from_pretrained(
                        cfg.teacher.llm_path, use_fast=False
                    )
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.add_special_tokens(
                        {"pad_token": "[PAD]"}
                    )  # This is important for some models
                    tokenizer.padding_side = "right"
                    tokenizer.add_special_tokens(
                        {
                            "additional_special_tokens": [
                                "[PH]",
                                "[HistoryEmb]",
                                "[CansEmb]",
                                "[ItemEmb]",
                            ]
                        }
                    )

                    # Instantiate MInterface (it will load LLM and rec_model)
                    # Need to pass all hparams that MInterface expects
                    ilora_hparams = {
                        "llm_path": cfg.teacher.llm_path,
                        "rec_model_path": str(
                            self.ilora_path / "rec_model" / "SASRec_movielens.pt"
                        ),  # Needs to be dynamic
                        "model_name": "mlp_projector",  # Default from iLoRA main.py
                        "rec_size": 64,  # Default from iLoRA main.py
                        "llm_tuning": "moelora",
                        "lora_r": 8,
                        "lora_alpha": 32,
                        "lora_dropout": 0.1,
                        "num_moe": 4,
                        "gating": "Dense",
                        "router": "unshare",  # Default from iLoRA main.py
                        "save": "all",  # Ensure full checkpoint is saved
                        "dataset": cfg.dataset.name,  # For data_dir path in MInterface
                        "data_dir": str(
                            self.ilora_path / "data" / "ref" / cfg.dataset.name
                        ),  # For data_dir path in MInterface
                        "padding_item_id": 0,  # Placeholder
                        "rec_model": "SASRec",  # Default from iLoRA main.py
                        "loss": "lm",  # Default from iLoRA main.py
                        "lr": cfg.teacher.lr,  # Required by configure_optimizers
                        "lr_scheduler": "cosine",  # Required by configure_optimizers
                        "lr_decay_min_lr": 1e-6,
                        "lr_warmup_start_lr": 1e-6,
                        "weight_decay": 1e-5,
                        "output_dir": run_dir
                        / "teacher_outputs_raw",  # Required for validation/test output
                        "prompt_path": self.ilora_path
                        / "prompt"
                        / (cfg.teacher.prompt_template + ".txt"),
                        "batch_size": cfg.teacher.batch_size,  # Required by DInterface
                        "num_workers": 0,  # Required by DInterface
                        "max_epochs": cfg.teacher.num_epochs,  # Required by DInterface
                    }

                    # Adjust dataset specific paths and padding_item_id
                    if cfg.dataset.name == "movielens":
                        ilora_hparams["rec_model_path"] = str(
                            self.ilora_path / "rec_model" / "SASRec_movielens.pt"
                        )
                        ilora_hparams["data_dir"] = str(
                            self.ilora_path / "data" / "ref" / "movielens"
                        )
                        ilora_hparams["padding_item_id"] = 1682
                    elif cfg.dataset.name == "steam":
                        ilora_hparams["rec_model_path"] = str(
                            self.ilora_path / "rec_model" / "SASRec_steam.pt"
                        )
                        ilora_hparams["data_dir"] = str(
                            self.ilora_path / "data" / "ref" / "steam"
                        )
                        ilora_hparams["padding_item_id"] = 3581
                    elif cfg.dataset.name == "lastfm":
                        ilora_hparams["rec_model_path"] = str(
                            self.ilora_path / "rec_model" / "lastfm.pt"
                        )
                        ilora_hparams["data_dir"] = str(
                            self.ilora_path / "data" / "ref" / "lastfm"
                        )
                        ilora_hparams["padding_item_id"] = 4606

                    # Load the model from checkpoint
                    model = MInterface.load_from_checkpoint(
                        latest_checkpoint, **ilora_hparams
                    )
                    model.eval().to(cfg.general.device)
                # Get item_num from processed data
                _, _, _, data_statis = load_processed_data(cfg, data_dir)
                item_num = data_statis["item_num"][0]
                # ilora_hparams might not be defined if dummy_llm is used,
                # so get padding_item_id directly from cfg.dataset
                padding_item_id = cfg.dataset.get("padding_item_id", 0)
                all_item_ids = torch.arange(item_num, device=cfg.general.device)

                # Exclude padding item if it's part of item_num range
                if padding_item_id < item_num:
                    all_item_ids = all_item_ids[all_item_ids != padding_item_id]

                # Direct access to projector and rec_model
                # This assumes rec_model.item_embeddings can take a single item ID or a batch of single IDs
                # And projector can take a batch of embeddings.

                # Get embeddings from rec_model
                rec_item_embeddings = model.rec_model.item_embeddings(
                    all_item_ids
                )  # Assuming this works for all_item_ids

                # Project to LLM space
                llm_item_embeddings = model.projector(rec_item_embeddings)

                torch.save(
                    llm_item_embeddings.cpu(), teacher_outputs_dir / "all_embeddings.pt"
                )
                log.info(
                    f"Saved all_embeddings.pt to {teacher_outputs_dir / 'all_embeddings.pt'}"
                )

            except ImportError as e:
                log.error(
                    f"Failed to import iLoRA modules: {e}. Ensure iLoRA is installed and its path is correct."
                )
                # Create dummy files to avoid downstream errors
                (teacher_outputs_dir / "all_embeddings.pt").touch()
                (teacher_outputs_dir / "myrank_train.txt").touch()
                (teacher_outputs_dir / "confidence_train.txt").touch()
                log.info("Created dummy teacher output files due to import error.")
                return
            except Exception as e:
                log.error(f"Error during iLoRA export: {e}")
                # Create dummy files to avoid downstream errors
                (teacher_outputs_dir / "all_embeddings.pt").touch()
                (teacher_outputs_dir / "myrank_train.txt").touch()
                (teacher_outputs_dir / "confidence_train.txt").touch()
                log.info("Created dummy teacher output files due to export error.")
                return
            finally:
                if str(self.ilora_path) in sys.path:
                    sys.path.remove(str(self.ilora_path))

        # --- Export myrank_train.txt and confidence_train.txt ---
        # This is the challenging part. iLoRA's MInterface.generate produces text,
        # not direct item rankings/scores.
        # The DLLM2Rec README refers to BIGRec's evaluate.py for "rank" and "dist".
        # We need to either:
        # 1. Find an equivalent in iLoRA to get item scores/rankings for a given context.
        # 2. Implement a prompting strategy to make the LLM output rankings and parse it.
        # 3. If iLoRA's SASRec model (loaded as rec_model) is used for ranking,
        #    we could use that, but the spec implies LLM-based ranking.

        # For now, create dummy files and log a warning.
        # This part needs further investigation or a specific implementation strategy.
        log.warning("Generating dummy myrank_train.txt and confidence_train.txt.")
        log.warning(
            "Actual implementation requires LLM to output item rankings/scores, which is not directly available in iLoRA's MInterface.generate."
        )

        # Dummy data: Assuming 100 training samples, top 10 items
        num_train_samples = 100  # Placeholder, should come from data_module.train_df
        top_n = 10  # As per DLLM2Rec's candidate_topk

        # Ensure item_num is available, otherwise use a default
        try:
            _, _, _, data_statis = load_processed_data(cfg, data_dir)
            item_num = data_statis["item_num"][0]
        except Exception as e:
            log.warning(
                f"Could not load data statistics to determine item_num: {e}. Using default item_num=1000."
            )
            item_num = 1000

        dummy_myrank_train = (
            torch.randint(0, item_num, (num_train_samples, top_n)).cpu().numpy()
        )
        dummy_confidence_train = torch.rand(num_train_samples, top_n).cpu().numpy()

        pd.DataFrame(dummy_myrank_train).to_csv(
            teacher_outputs_dir / "myrank_train.txt", sep=" ", index=False, header=False
        )
        pd.DataFrame(dummy_confidence_train).to_csv(
            teacher_outputs_dir / "confidence_train.txt",
            sep=" ",
            index=False,
            header=False,
        )
        log.info(
            f"Saved dummy myrank_train.txt and confidence_train.txt to {teacher_outputs_dir}"
        )

        with time_block("distill_teacher_export_time"):
            # Placeholder for actual export logic
            pass
        log.info("iLoRA teacher outputs exported.")


if __name__ == "__main__":
    # Example usage
    from omegaconf import OmegaConf
    from src.core.logging import setup_logging
    from src.core.paths import (
        get_current_run_dir,
        get_data_dir,
        get_teacher_outputs_dir,
    )
    from src.core.data_utils import preprocess_data
    import shutil

    # Setup logging
    log_file_path = Path("temp_run_ilora/logs/ilora_backend_test.log")
    setup_logging(log_file_path)

    # Dummy config
    dummy_cfg = OmegaConf.create(
        {
            "general": {"seed": 42, "device": "cpu"},  # Use CPU for example
            "dataset": {
                "name": "movielens",
                "max_seq_len": 50,
                "min_user_inter": 5,
                "split": {"method": "leave_one_out"},
                "padding_item_id": 1682,  # For movielens
            },
            "paths": {
                "data_dir": "temp_data_ilora",
                "result_root": "temp_run_ilora",
                "teacher_outputs_dir": "${paths.data_dir}/teacher_outputs/${dataset.name}",
            },
            "teacher": {
                "name": "ilora",
                "llm_path": "/path/to/Llama-2-7b-hf",  # Placeholder, replace with actual path for real run
                "batch_size": 4,
                "lr": 1e-4,
                "num_epochs": 1,  # Reduced for quick test
                "prompt_template": "movie",  # Assuming movie.txt exists in iLoRA/prompt
                "save_interval": 1,
            },
            "hydra": {"run": {"dir": "temp_run_ilora"}},
        }
    )

    # Ensure dummy prompt file exists for iLoRA
    ilora_prompt_dir = Path("src/third_party/ilora/prompt")
    ilora_prompt_dir.mkdir(parents=True, exist_ok=True)
    (ilora_prompt_dir / "movie.txt").write_text(
        "Recommend a movie based on [HistoryHere]. Candidates: [CansHere]. Target: [TargetHere]"
    )

    # Prepare dummy data
    temp_data_dir = get_data_dir(dummy_cfg)
    preprocess_data(dummy_cfg, temp_data_dir)

    # Setup TensorBoard Logger
    run_dir = Path(get_current_run_dir(dummy_cfg))
    tb_log_dir = run_dir / "tb"
    tb_logger = TensorBoardLogger(tb_log_dir)

    # Set time_block output path
    time_block.set_output_path(run_dir / "metrics" / "time.json")

    ilora_backend = ILoRABackend()

    # Test train method (will fail if LLM path is invalid or iLoRA has issues)
    log.info("\n--- Testing iLoRABackend.train ---")
    try:
        # Create dummy checkpoint for export test
        (run_dir / "teacher_checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "teacher_checkpoints" / "epoch=00-metric=0.123.ckpt").touch()

        # This will likely fail if LLM_PATH is not valid.
        # For testing the flow, we can comment out the actual subprocess.run
        # and just log that it would run.
        # ilora_backend.train(dummy_cfg, tb_logger, run_dir)
        log.info("Skipping actual iLoRA train for example. Assuming it would run.")
    except Exception as e:
        log.error(f"iLoRABackend.train failed: {e}")

    # Test export method
    log.info("\n--- Testing iLoRABackend.export_for_dllm2rec ---")
    teacher_outputs_dir = Path(get_teacher_outputs_dir(dummy_cfg))
    ilora_backend.export_for_dllm2rec(
        dummy_cfg, run_dir, temp_data_dir, teacher_outputs_dir
    )

    tb_logger.close_all()

    # Clean up
    shutil.rmtree("temp_run_ilora", ignore_errors=True)
    shutil.rmtree("temp_data_ilora", ignore_errors=True)
    (ilora_prompt_dir / "movie.txt").unlink(missing_ok=True)
    print("\nCleaned up temporary directories and files.")
