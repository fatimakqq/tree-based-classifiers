import os
import pandas as pd
import matplotlib.pyplot as plt
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

def generate_accuracy_plot(results_df, output_file):
    """
    Generate accuracy comparison plot.
    
    Args:
        results_df (pd.DataFrame): Results DataFrame
        output_file (str): Output file path for the plot
    """
    plt.figure(figsize=(15, 8))
    
    # Group by dataset size
    data_sizes = [100, 1000, 5000]
    clause_counts = [300, 500, 1000, 1500, 1800]
    
    # Plot accuracy for each dataset size
    for i, size in enumerate(data_sizes):
        plt.subplot(1, 3, i+1)
        
        # Filter data for current size
        size_data = results_df[results_df['dataset'].str.contains(f"d{size}")]
        
        # Extract clause counts and classifier accuracies
        x = np.arange(len(clause_counts))
        width = 0.2
        
        # Plot each classifier
        plt.bar(x - 0.3, size_data['decision_tree'], width, label='Decision Tree')
        plt.bar(x - 0.1, size_data['bagging'], width, label='Bagging')
        plt.bar(x + 0.1, size_data['random_forest'], width, label='Random Forest')
        plt.bar(x + 0.3, size_data['gradient_boosting'], width, label='Gradient Boosting')
        
        plt.xlabel('Number of Clauses')
        plt.ylabel('Accuracy')
        plt.title(f'Dataset Size: {size}')
        plt.xticks(x, clause_counts)
        plt.ylim(0, 1)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

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
    
    # Generate plots
    accuracy_plot = os.path.join(output_dir, "figures", "accuracy_comparison.png")
    f1_plot = os.path.join(output_dir, "figures", "f1_comparison.png")
    
    generate_accuracy_plot(pd.DataFrame(accuracy_results), accuracy_plot)
    generate_accuracy_plot(pd.DataFrame(f1_results), f1_plot)