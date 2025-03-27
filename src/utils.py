import os
import pandas as pd
import numpy as np

def save_results(results_dict, output_file):
    """
    Save results to a CSV file.
    
    Args:
        results_dict (dict): Dictionary containing results
        output_file (str): Output CSV file path
    """
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_dict)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def process_results(accuracy_results, f1_results, output_dir="results"):
    """
    Process and save results.
    
    Args:
        accuracy_results (dict): Dictionary with accuracy results
        f1_results (dict): Dictionary with F1 score results
        output_dir (str): Output directory
    """
    # Save results to CSV
    accuracy_file = os.path.join(output_dir, "accuracy_results.csv")
    f1_file = os.path.join(output_dir, "f1_results.csv")
    
    save_results(accuracy_results, accuracy_file)
    save_results(f1_results, f1_file)
    
    # Generate summary statistics
    acc_df = pd.DataFrame(accuracy_results)
    f1_df = pd.DataFrame(f1_results)
    
    print("\nAccuracy Summary Statistics:")
    for col in acc_df.columns:
        if col != 'dataset':
            print(f"{col}: mean={acc_df[col].mean():.4f}, min={acc_df[col].min():.4f}, max={acc_df[col].max():.4f}")
    
    print("\nF1 Score Summary Statistics:")
    for col in f1_df.columns:
        if col != 'dataset':
            print(f"{col}: mean={f1_df[col].mean():.4f}, min={f1_df[col].min():.4f}, max={f1_df[col].max():.4f}")