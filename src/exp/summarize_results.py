import argparse
import pandas as pd
from pathlib import Path
import yaml
import re
import json

def parse_metrics_file(file_path: Path) -> dict:
    """
    Parses test_metrics.txt or all_evaluation_results.json to extract metrics.
    """
    metrics = {}
    if file_path.name == "test_metrics.txt":
        with open(file_path, 'r') as f:
            content = f.read()
            # Example: {'recall@10': 0.08994708994708994, 'ndcg@10': 0.04089547865674424, 'hit_ratio@10': 0.08994708994708994}
            # Use regex to find all float values associated with keys
            matches = re.findall(r"(\w+@\d+): ([\d.]+)", content)
            for key, value in matches:
                metrics[key] = float(value)
    elif file_path.name == "all_evaluation_results.json":
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Flatten the nested dictionary
            for model_name, model_metrics in data.items():
                for metric_name, value in model_metrics.items():
                    metrics[f"{model_name}_{metric_name}"] = value
    return metrics

def get_experiment_type(config: dict) -> str:
    """
    Determines the experiment type from the config.
    """
    # This is a heuristic based on the hydra config structure
    if "teacher" in config and "distill" not in config:
        return "teacher"
    elif "student" in config and "distill" not in config:
        return "student_baseline"
    elif "distill" in config:
        return "distill"
    elif "eval" in config:
        return "eval_all"
    return "unknown"


def main(results_dir: str, output_csv: str):
    root_path = Path(results_dir)
    if not root_path.exists():
        print(f"Error: Directory not found at {root_path}")
        return

    all_results = []

    for exp_dir in root_path.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("result_"):
            continue

        config_path = exp_dir / "config.yaml"
        metrics_path_txt = exp_dir / "test_metrics.txt"
        metrics_path_json = exp_dir / "all_evaluation_results.json"

        # if not config_path.exists():
        #     continue

        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            print(f"Warning: config.yaml not found for {exp_dir.name}. Skipping config parsing for this experiment.")

        exp_type = get_experiment_type(config)
        
        # Basic info from config
        result_info = {
            "experiment_dir": exp_dir.name,
            "experiment_type": exp_type,
            "seed": config.get("seed"),
            "dataset": config.get("dataset", {}).get("name"),
            "timestamp": exp_dir.name.replace("result_", "")
        }
        
        # Add model-specific params
        if exp_type == "student_baseline":
            student_cfg = config.get("student", {})
            result_info["student_model_type"] = student_cfg.get("model_type")
            result_info["student_hidden_size"] = student_cfg.get("hidden_size")
            result_info["student_num_heads"] = student_cfg.get("num_heads")
            result_info["student_num_layers"] = student_cfg.get("num_layers")
        elif exp_type == "teacher":
            teacher_cfg = config.get("teacher", {})
            result_info["teacher_model_type"] = teacher_cfg.get("model_type")
            result_info["teacher_llm"] = teacher_cfg.get("llm_model_name")
        elif exp_type == "distill":
            distill_cfg = config.get("distill", {})
            result_info["distill_ranking_loss_weight"] = distill_cfg.get("ranking_loss_weight")
            result_info["distill_embedding_loss_weight"] = distill_cfg.get("embedding_loss_weight")
            result_info["distill_lam"] = distill_cfg.get("lam")


        # Parse metrics
        metrics = {}
        if metrics_path_txt.exists():
            metrics.update(parse_metrics_file(metrics_path_txt))
        elif metrics_path_json.exists():
            metrics.update(parse_metrics_file(metrics_path_json))
        
        if not metrics:
            # If no metrics file, maybe it's a teacher run that only logs loss
            log_path = exp_dir / "tensorboard_logs" / "default" / "version_0" / "metrics.csv"
            if log_path.exists():
                try:
                    df_metrics = pd.read_csv(log_path)
                    if "val_loss" in df_metrics.columns:
                        best_val_loss = df_metrics["val_loss"].dropna().min()
                        metrics["best_val_loss"] = best_val_loss
                except Exception as e:
                    print(f"Could not parse metrics.csv for {exp_dir.name}: {e}")


        result_info.update(metrics)
        all_results.append(result_info)

    if not all_results:
        print("No results found.")
        return

    df = pd.DataFrame(all_results)
    # Sort by timestamp
    df = df.sort_values("timestamp", ascending=False)
    
    # Reorder columns for better readability
    core_cols = ["timestamp", "experiment_type", "dataset", "seed"]
    metric_cols = sorted([col for col in df.columns if 'recall' in col or 'ndcg' in col or 'hit' in col or 'loss' in col])
    param_cols = sorted([col for col in df.columns if col not in core_cols and col not in metric_cols and col != "experiment_dir"])
    
    ordered_cols = core_cols + metric_cols + param_cols + ["experiment_dir"]
    df = df[[col for col in ordered_cols if col in df.columns]]

    df.to_csv(output_csv, index=False)
    print(f"Successfully summarized {len(df)} experiments into {output_csv}")
    print("\n--- Summary ---")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize experiment results from Hydra output directories.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="result",
        help="The root directory containing the result folders."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="result/summary.csv",
        help="Path to save the summary CSV file."
    )
    args = parser.parse_args()
    main(args.results_dir, args.output_csv)
