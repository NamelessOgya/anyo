import os
import sys
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import argparse

def analyze_tb_logs(result_dir):
    # Find tfevents file
    tb_logs_dir = os.path.join(result_dir, "tb_logs")
    event_files = glob.glob(os.path.join(tb_logs_dir, "events.out.tfevents.*"))
    
    if not event_files:
        abs_path = os.path.abspath(tb_logs_dir)
        print(f"No event files found in {tb_logs_dir} (Absolute: {abs_path})")
        if os.path.exists(result_dir):
            print(f"Contents of {result_dir}: {os.listdir(result_dir)}")
            if os.path.exists(tb_logs_dir):
                print(f"Contents of {tb_logs_dir}: {os.listdir(tb_logs_dir)}")
            else:
                print(f"{tb_logs_dir} does not exist.")
        else:
            print(f"{result_dir} does not exist.")
        return

    # Use the latest file if multiple
    event_file = sorted(event_files)[-1]
    print(f"Analyzing event file: {event_file}")

    ea = EventAccumulator(event_file)
    ea.Reload()

    # Get available tags
    tags = ea.Tags()['scalars']
    print(f"Available tags: {tags}")

    # Define metrics of interest
    metrics_of_interest = ["epoch", "val_hr@10", "val_ndcg@10", "train_alpha", "train_loss"]
    
    data = {}
    
    # Extract data for each metric
    for tag in metrics_of_interest:
        # Find exact match or close match
        found_tag = None
        if tag in tags:
            found_tag = tag
        else:
            # Try to find partial match (e.g. "train_alpha_epoch")
            matches = [t for t in tags if tag in t]
            if matches:
                # Prefer exact match if possible, otherwise shortest match (likely the base name)
                found_tag = sorted(matches, key=len)[0]
        
        if found_tag:
            events = ea.Scalars(found_tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            # Use step as index to align different metrics
            data[tag] = pd.Series(values, index=steps)
        else:
            print(f"Warning: Metric '{tag}' not found in logs.")

    if not data:
        print("No relevant metrics found.")
        return

    df = pd.DataFrame(data)
    df.index.name = "Step"
    
    # If 'epoch' exists, use it to sort or display. 
    # Note: 'epoch' might be logged at different steps than validation metrics.
    # We'll forward fill epoch to align with validation steps if needed, 
    # but usually PL logs them together or close enough.
    # Let's just display the DataFrame sorted by Step.
    
    # Drop rows where all columns are NaN (unlikely)
    df = df.dropna(how='all')
    
    # Sort by Step
    df = df.sort_index()
    
    print("\nMetrics Summary (Aligned by Step):")
    print(df)
    
    # Filter to show only rows where val_hr@10 is present (Validation steps)
    if "val_hr@10" in df.columns:
        print("\nValidation Epochs Summary:")
        val_df = df[df["val_hr@10"].notna()]
        print(val_df)
    
    # Save to CSV
    output_csv = os.path.join(result_dir, "metrics_summary.csv")
    df.to_csv(output_csv)
    print(f"\nFull metrics saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze TensorBoard logs and extract validation metrics.")
    parser.add_argument("result_dir", type=str, help="Path to the result directory containing tb_logs")
    args = parser.parse_args()
    
    analyze_tb_logs(args.result_dir)
