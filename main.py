import os
import argparse
import pandas as pd
from src.decision_tree import run_decision_tree_experiments
from src.bagging import run_bagging_experiments
from src.random_forest import run_random_forest_experiments
from src.gradient_boosting import run_gradient_boosting_experiments
from src.mnist import run_mnist_experiments
from src.utils import process_results

def merge_results(dt_results, bag_results, rf_results, gb_results):
    """
    Merge results from different classifiers.
    
    Returns:
        tuple: (accuracy_results, f1_results)
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

def test_setup():
    """Run a minimal test to verify the setup is working."""
    print("Testing project setup...")
    
    # Test data loading
    from src.data_loader import load_dataset
    try:
        # Try loading a single dataset
        clause_count = 300
        data_size = 100
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(clause_count, data_size)
        print(f"Successfully loaded dataset c{clause_count}_d{data_size}")
        print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}, Test shape: {X_test.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Test a quick classifier run (with minimal parameters)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    
    try:
        # Train a simple model
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Quick test - Decision Tree accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"Error running test classifier: {e}")
        return False
    
    print("Setup test completed successfully!")
    return True

def quick_run():
    """Run a quick test with one dataset and one classifier."""
    print("Running quick test with one dataset...")
    
    # Load one small dataset
    from src.data_loader import load_dataset
    try:
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(300, 100)
        
        # Run a simple Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, f1_score
        
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Quick test results - Decision Tree: Accuracy={accuracy:.4f}, F1={f1:.4f}")
        return True
    except Exception as e:
        print(f"Error in quick run: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run tree-based classifier experiments')
    parser.add_argument('--datasets', action='store_true', help='Run Boolean formula dataset experiments')
    parser.add_argument('--mnist', action='store_true', help='Run MNIST dataset experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--test', action='store_true', help='Run a quick test to verify setup')
    parser.add_argument('--quick', action='store_true', help='Run a quick test with one dataset')
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    
    # Run test or quick run if requested
    if args.test:
        if test_setup():
            print("Setup is working correctly!")
        else:
            print("Setup test failed. Please check the error messages.")
        return
    
    if args.quick:
        if quick_run():
            print("Quick run completed successfully!")
        else:
            print("Quick run failed. Please check the error messages.")
        return
    
    # Run Boolean formula dataset experiments
    if args.datasets or args.all:
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
    
    # Run MNIST experiments
    if args.mnist or args.all:
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