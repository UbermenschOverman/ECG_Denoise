import os
import csv
import glob

def merge_csvs(output_root):
    """
    Scans subdirectories in output_root.
    Merges 'summary_average.csv' and 'metrics.csv' into 'merged_metrics.csv'.
    Layout:
      [Summary CSV Content]
      
      [Metrics CSV Content]
    """
    subdirs = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
    
    for d in subdirs:
        dir_path = os.path.join(output_root, d)
        print(f"Processing {d}...")
        
        summary_path = os.path.join(dir_path, "summary_average.csv")
        metrics_path = os.path.join(dir_path, "metrics.csv")
        merged_path = os.path.join(dir_path, "merged_metrics.csv")
        
        summary_rows = []
        metrics_rows = []
        
        # Read Summary
        if os.path.exists(summary_path):
            with open(summary_path, 'r', newline='') as f:
                reader = csv.reader(f)
                summary_rows = list(reader)
        else:
            print(f"  ⚠️ Warning: {summary_path} not found.")

        # Read Metrics
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', newline='') as f:
                reader = csv.reader(f)
                metrics_rows = list(reader)
        else:
            print(f"  ⚠️ Warning: {metrics_path} not found.")
            
        if not summary_rows and not metrics_rows:
            print("  Skipping: No data found.")
            continue
            
        # Write Merged
        with open(merged_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write Summary Section
            if summary_rows:
                writer.writerow(["=== SUMMARY ==="])
                writer.writerows(summary_rows)
                writer.writerow([]) # Empty line
            
            # Write Metrics Section
            if metrics_rows:
                writer.writerow(["=== DETAIL METRICS ==="])
                writer.writerows(metrics_rows)
                
        print(f"  ✅ Created {merged_path}")

if __name__ == "__main__":
    # Determine root based on script location
    this_file = os.path.abspath(__file__)
    codev2_dir = os.path.dirname(this_file)
    ecg_root = os.path.dirname(codev2_dir)
    
    target_dir = os.path.join(ecg_root, "evaluation_outputs_BKIC_csv")
    
    if os.path.exists(target_dir):
        print(f"Merging CSVs in: {target_dir}")
        merge_csvs(target_dir)
    else:
        print(f"Error: Directory {target_dir} does not exist.")
