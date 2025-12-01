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
        print(f"No event files found in {tb_logs_dir}")
        return

    # Use the latest file if multiple
    event_file = sorted(event_files)[-1]
    print(f"Analyzing event file: {event_file}")

    ea = EventAccumulator(event_file)
    ea.Reload()

    # Get available tags
    tags = ea.Tags()['scalars']
    print(f"Available tags: {tags}")

    # Filter for validation metrics
    val_tags = [t for t in tags if "val" in t or "hr" in t or "ndcg" in t]
    
    data = {}
    for tag in val_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = pd.Series(values, index=steps)

    if not data:
        print("No validation metrics found.")
        return

    df = pd.DataFrame(data)
    df.index.name = "Step"
    
    print("\nValidation Metrics Summary:")
    print(df)
    
    # Save to CSV
    output_csv = os.path.join(result_dir, "validation_metrics.csv")
    df.to_csv(output_csv)
    print(f"\nMetrics saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_tb_logs.py <result_dir>")
        sys.exit(1)
        
    result_dir = sys.argv[1]
    analyze_tb_logs(result_dir)
