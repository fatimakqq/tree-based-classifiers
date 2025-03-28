# utilities.py
import os
import pandas as pd

# Data loading functions
def load_dataset(clause_count, data_size, data_dir="data"):
    """Load dataset for a specific configuration."""
    # Define file paths
    train_file = os.path.join(data_dir, f"train_c{clause_count}_d{data_size}.csv")
    valid_file = os.path.join(data_dir, f"valid_c{clause_count}_d{data_size}.csv")
    test_file = os.path.join(data_dir, f"test_c{clause_count}_d{data_size}.csv")
    
    # Load datasets
    train_data = pd.read_csv(train_file, header=None)
    valid_data = pd.read_csv(valid_file, header=None)
    test_data = pd.read_csv(test_file, header=None)
    
    # Split features and labels in a more explicit way
    # For training data
    num_columns = len(train_data.columns)
    feature_columns = list(range(0, num_columns - 1))  # All columns except the last one
    label_column = num_columns - 1  # The last column
    
    X_train = train_data.iloc[:, feature_columns]  # Select all features
    y_train = train_data.iloc[:, label_column]     # Select the label
    
    # For validation data
    X_valid = valid_data.iloc[:, feature_columns]  # Select all features
    y_valid = valid_data.iloc[:, label_column]     # Select the label
    
    # For test data
    X_test = test_data.iloc[:, feature_columns]    # Select all features
    y_test = test_data.iloc[:, label_column]       # Select the label
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def combine_train_valid(X_train, y_train, X_valid, y_valid):
    """Combine training and validation sets."""
    X_combined = pd.concat([X_train, X_valid])
    y_combined = pd.concat([y_train, y_valid])
    return X_combined, y_combined

def get_all_dataset_configs():
    """Return a list of all dataset configurations."""
    clause_counts = [300, 500, 1000, 1500, 1800]
    data_sizes = [100, 1000, 5000]
    
    configs = []
    for clause_count in clause_counts:
        for data_size in data_sizes:
            configs.append((clause_count, data_size))
    
    return configs

# Results handling functions
def merge_results(dt_results, bag_results, rf_results, gb_results):
    """Merge results from different classifiers."""
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

def process_results(accuracy_results, f1_results):
    """Process and print results."""
    # Convert to DataFrames for easier processing
    acc_df = pd.DataFrame(accuracy_results)
    f1_df = pd.DataFrame(f1_results)
    
    # Print accuracy table
    print("\nClassification Accuracy Table:")
    print(acc_df.to_string(index=False))
    
    # Print F1 score table
    print("\nF1 Score Table:")
    print(f1_df.to_string(index=False))
    
    # Print summary statistics
    print("\nAccuracy Summary Statistics:")
    for col in acc_df.columns:
        if col != 'dataset':
            print(f"{col}: mean={acc_df[col].mean():.4f}, min={acc_df[col].min():.4f}, max={acc_df[col].max():.4f}")
    
    print("\nF1 Score Summary Statistics:")
    for col in f1_df.columns:
        if col != 'dataset':
            print(f"{col}: mean={f1_df[col].mean():.4f}, min={f1_df[col].min():.4f}, max={f1_df[col].max():.4f}")

def print_summary(accuracy_results, f1_results):
    """Print a summary of results."""
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