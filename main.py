import os
import pandas as pd
import numpy as np
from src.decision_tree import run_decision_tree_experiments
from src.bagging import run_bagging_experiments
from src.random_forest import run_random_forest_experiments
from src.gradient_boosting import run_gradient_boosting_experiments
from src.mnist import run_mnist_experiments

def save_results(results_dict, output_file):
    """
    Save results to a CSV file.
    """
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def process_results(accuracy_results, f1_results, output_dir="results"):
    """
    Process and save results.
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

def merge_results(dt_results, bag_results, rf_results, gb_results):
    """
    Merge results from different classifiers.
    """
    # Merge accuracy results
    accuracy_results = {'dataset': dt_results[0]['dataset']}
    accuracy_results['decision_tree'] = dt_results[0]['decision_tree']
    accuracy_results['bagging'] = bag_results[0]['bagging']
    accuracy_results['random_forest'] = rf_results[0]['random_forest']
    accuracy_results['gradient_boosting'] = gb_results[0]['gradient_boosting']
    
    # Merge F1 results
    f1_results = {'dataset': dt_results[1]['dataset']}
    f1_results['decision_tree'] = dt_results[1]['decision_tree']
    f1_results['bagging'] = bag_results[1]['bagging']
    f1_results['random_forest'] = rf_results[1]['random_forest']
    f1_results['gradient_boosting'] = gb_results[1]['gradient_boosting']
    
    return accuracy_results, f1_results

def print_summary(accuracy_results, f1_results):
    """
    Print a summary of results.
    """
    # Convert to DataFrame for easier analysis
    acc_df = pd.DataFrame(accuracy_results)
    f1_df = pd.DataFrame(f1_results)
    
    # Calculate mean performance for each classifier
    print("\nMean Accuracy:")
    for col in acc_df.columns:
        if col != 'dataset':
            mean_acc = acc_df[col].mean()
            print(f"{col}: {mean_acc:.4f}")
    
    print("\nMean F1 Score:")
    for col in f1_df.columns:
        if col != 'dataset':
            mean_f1 = f1_df[col].mean()
            print(f"{col}: {mean_f1:.4f}")
    
    # Find best classifier for each dataset
    print("\nBest classifier per dataset (by accuracy):")
    for i, dataset in enumerate(acc_df['dataset']):
        row = acc_df.iloc[i]
        classifiers = [col for col in row.index if col != 'dataset']
        accuracies = [row[col] for col in classifiers]
        best_idx = accuracies.index(max(accuracies))
        best_clf = classifiers[best_idx]
        print(f"{dataset}: {best_clf} ({accuracies[best_idx]:.4f})")

def main():
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Part 1-4: Run experiments on Boolean formula datasets
    print("="*50)
    print("Running experiments on Boolean formula datasets")
    print("="*50)
    
    # Run Decision Tree experiments
    print("\nRunning Decision Tree experiments...")
    dt_results = run_decision_tree_experiments()
    
    # Run Bagging experiments
    print("\nRunning Bagging experiments...")
    bag_results = run_bagging_experiments()
    
    # Run Random Forest experiments
    print("\nRunning Random Forest experiments...")
    rf_results = run_random_forest_experiments()
    
    # Run Gradient Boosting experiments
    print("\nRunning Gradient Boosting experiments...")
    gb_results = run_gradient_boosting_experiments()
    
    # Merge and process results
    print("\nProcessing Boolean formula dataset results...")
    accuracy_results, f1_results = merge_results(dt_results, bag_results, rf_results, gb_results)
    process_results(accuracy_results, f1_results)
    
    # Print summary
    print("\nBoolean Formula Dataset Results Summary:")
    print_summary(accuracy_results, f1_results)
    
    # Part 6: Run experiments on MNIST dataset
    print("\n" + "="*50)
    print("Running experiments on MNIST dataset")
    print("="*50)
    
    mnist_results = run_mnist_experiments()
    
    # Save MNIST results
    mnist_df = pd.DataFrame(mnist_results)
    mnist_df.to_csv('results/mnist_results.csv', index=False)
    
    # Print MNIST summary
    print("\nMNIST Results Summary:")
    for i, clf in enumerate(mnist_results['classifier']):
        print(f"{clf}: {mnist_results['accuracy'][i]:.4f}")
    
    # Identify best classifier
    best_idx = mnist_results['accuracy'].index(max(mnist_results['accuracy']))
    best_clf = mnist_results['classifier'][best_idx]
    print(f"\nBest classifier on MNIST: {best_clf} with accuracy {mnist_results['accuracy'][best_idx]:.4f}")

if __name__ == "__main__":
    main()