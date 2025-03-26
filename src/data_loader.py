import os
import pandas as pd

def load_dataset(clause_count, data_size, data_dir="data"):
    """
    Load training, validation, and test datasets for a specific configuration.
    
    Args:
        clause_count (int): Number of clauses (300, 500, 1000, 1500, 1800)
        data_size (int): Number of examples (100, 1000, 5000)
        data_dir (str): Directory containing the datasets
        
    Returns:
        tuple: (X_train, y_train, X_valid, y_valid, X_test, y_test)
    """
    # Define file paths
    train_file = os.path.join(data_dir, f"train_c{clause_count}_d{data_size}.csv")
    valid_file = os.path.join(data_dir, f"valid_c{clause_count}_d{data_size}.csv")
    test_file = os.path.join(data_dir, f"test_c{clause_count}_d{data_size}.csv")
    
    # Load datasets
    train_data = pd.read_csv(train_file, header=None)
    valid_data = pd.read_csv(valid_file, header=None)
    test_data = pd.read_csv(test_file, header=None)
    
    # Split features and labels
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    
    X_valid = valid_data.iloc[:, :-1]
    y_valid = valid_data.iloc[:, -1]
    
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def combine_train_valid(X_train, y_train, X_valid, y_valid):
    """
    Combine training and validation sets.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_valid: Validation features
        y_valid: Validation labels
        
    Returns:
        tuple: (X_combined, y_combined)
    """
    X_combined = pd.concat([X_train, X_valid])
    y_combined = pd.concat([y_train, y_valid])
    return X_combined, y_combined

def get_all_dataset_configs():
    """
    Return a list of all dataset configurations.
    
    Returns:
        list: List of (clause_count, data_size) tuples
    """
    clause_counts = [300, 500, 1000, 1500, 1800]
    data_sizes = [100, 1000, 5000]
    
    configs = []
    for clause_count in clause_counts:
        for data_size in data_sizes:
            configs.append((clause_count, data_size))
    
    return configs

