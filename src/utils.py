import os
import pandas as pd

def load_dataset(clause_count, data_size, data_dir="data"):
    train_file = os.path.join(data_dir, f"train_c{clause_count}_d{data_size}.csv")
    train_data = pd.read_csv(train_file, header=None)
    valid_file = os.path.join(data_dir, f"valid_c{clause_count}_d{data_size}.csv")
    valid_data = pd.read_csv(valid_file, header=None)
    test_file = os.path.join(data_dir, f"test_c{clause_count}_d{data_size}.csv")
    test_data = pd.read_csv(test_file, header=None)
    
    
    
    
    
    #split data features + labels
    num_columns = len(train_data.columns)
    feature_columns = list(range(0, num_columns - 1))
    label_column = num_columns - 1
    #training
    X_train = train_data.iloc[:, feature_columns]
    y_train = train_data.iloc[:, label_column]
    
    #validation data
    X_valid = valid_data.iloc[:, feature_columns]
    y_valid = valid_data.iloc[:, label_column]
    
    #test data
    X_test = test_data.iloc[:, feature_columns]
    y_test = test_data.iloc[:, label_column]
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def combine_train_valid(X_train, y_train, X_valid, y_valid):
    X_train_valid = pd.concat([X_train, X_valid])
    y_train_valid = pd.concat([y_train, y_valid])
    return X_train_valid, y_train_valid
def get_all_dataset_configs():
    #make a organized list for parsing the dataset names later
    clauses = [300, 500, 1000, 1500, 1800]
    sizes = [100, 1000, 5000]
    configs = []
    for clause_count in clauses:
        for data_size in sizes:
            configs.append((clause_count, data_size))
    return configs
