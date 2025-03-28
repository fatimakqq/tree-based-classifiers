import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_openml

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
    
    # Split features and labels
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_valid = valid_data.iloc[:, :-1]
    y_valid = valid_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    
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

# Classifier experiment functions
def run_dt():
    """Run Decision Tree experiments on all datasets."""
    # Initialize results dictionaries
    accuracy_results = {'dataset': []}
    f1_results = {'dataset': []}
    
    # Get all dataset configs
    configs = get_all_dataset_configs()
    
    for clause_count, data_size in configs:
        dataset_name = f"c{clause_count}_d{data_size}"
        accuracy_results['dataset'].append(dataset_name)
        f1_results['dataset'].append(dataset_name)
        
        print(f"Running Decision Tree on dataset {dataset_name}")
        
        # Load data
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(clause_count, data_size)
        
        # Define parameter grid
        param_grid = {
            'criterion': ['gini'],
            'max_depth': [None, 10],
            'min_samples_split': [2]
        }
        
        # Create and tune the model
        dt_clf = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(dt_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        
        # Combine training and validation sets
        X_combined, y_combined = combine_train_valid(X_train, y_train, X_valid, y_valid)
        
        # Train with best parameters
        best_model = DecisionTreeClassifier(**best_params, random_state=42)
        best_model.fit(X_combined, y_combined)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        if 'decision_tree' not in accuracy_results:
            accuracy_results['decision_tree'] = []
            f1_results['decision_tree'] = []
            
        accuracy_results['decision_tree'].append(accuracy)
        f1_results['decision_tree'].append(f1)
        
        print(f"  Best params: {best_params}")
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return accuracy_results, f1_results

def run_bagging():
    """Run Bagging experiments on all datasets."""
    # Initialize results dictionaries
    accuracy_results = {'dataset': []}
    f1_results = {'dataset': []}
    
    # Get all dataset configs
    configs = get_all_dataset_configs()
    
    for clause_count, data_size in configs:
        dataset_name = f"c{clause_count}_d{data_size}"
        accuracy_results['dataset'].append(dataset_name)
        f1_results['dataset'].append(dataset_name)
        
        print(f"Running Bagging on dataset {dataset_name}")
        
        # Load data
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(clause_count, data_size)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [10],
            'estimator__max_depth': [None, 10]
        }
        
        # Create model
        base_estimator = DecisionTreeClassifier(random_state=42)
        bagging_clf = BaggingClassifier(estimator=base_estimator, random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(bagging_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        
        # Combine training and validation sets
        X_combined, y_combined = combine_train_valid(X_train, y_train, X_valid, y_valid)
        
        # Extract estimator params
        estimator_depth = best_params.pop('estimator__max_depth')
        
        # Retrain with best parameters on combined data
        base_estimator = DecisionTreeClassifier(max_depth=estimator_depth, random_state=42)
        best_model = BaggingClassifier(estimator=base_estimator, **best_params, random_state=42)
        best_model.fit(X_combined, y_combined)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Add estimator params back for reporting
        best_params['estimator__max_depth'] = estimator_depth
        
        # Store results
        if 'bagging' not in accuracy_results:
            accuracy_results['bagging'] = []
            f1_results['bagging'] = []
            
        accuracy_results['bagging'].append(accuracy)
        f1_results['bagging'].append(f1)
        
        print(f"  Best params: {best_params}")
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return accuracy_results, f1_results

def run_rf():
    """Run Random Forest experiments on all datasets."""
    # Initialize results dictionaries
    accuracy_results = {'dataset': []}
    f1_results = {'dataset': []}
    
    # Get all dataset configs
    configs = get_all_dataset_configs()
    
    for clause_count, data_size in configs:
        dataset_name = f"c{clause_count}_d{data_size}"
        accuracy_results['dataset'].append(dataset_name)
        f1_results['dataset'].append(dataset_name)
        
        print(f"Running Random Forest on dataset {dataset_name}")
        
        # Load data
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(clause_count, data_size)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [10],
            'max_depth': [None, 10],
            'max_features': ['sqrt']
        }
        
        # Create and tune the model
        rf_clf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        
        # Combine training and validation sets
        X_combined, y_combined = combine_train_valid(X_train, y_train, X_valid, y_valid)
        
        # Retrain with best parameters on combined data
        best_model = RandomForestClassifier(**best_params, random_state=42)
        best_model.fit(X_combined, y_combined)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        if 'random_forest' not in accuracy_results:
            accuracy_results['random_forest'] = []
            f1_results['random_forest'] = []
            
        accuracy_results['random_forest'].append(accuracy)
        f1_results['random_forest'].append(f1)
        
        print(f"  Best params: {best_params}")
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return accuracy_results, f1_results

def run_gb():
    """Run Gradient Boosting experiments on all datasets."""
    # Initialize results dictionaries
    accuracy_results = {'dataset': []}
    f1_results = {'dataset': []}
    
    # Get all dataset configs
    configs = get_all_dataset_configs()
    
    for clause_count, data_size in configs:
        dataset_name = f"c{clause_count}_d{data_size}"
        accuracy_results['dataset'].append(dataset_name)
        f1_results['dataset'].append(dataset_name)
        
        print(f"Running Gradient Boosting on dataset {dataset_name}")
        
        # Load data
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(clause_count, data_size)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50],
            'learning_rate': [0.1],
            'max_depth': [3, 5]
        }
        
        # Create and tune the model
        gb_clf = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(gb_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        
        # Combine training and validation sets
        X_combined, y_combined = combine_train_valid(X_train, y_train, X_valid, y_valid)
        
        # Retrain with best parameters on combined data
        best_model = GradientBoostingClassifier(**best_params, random_state=42)
        best_model.fit(X_combined, y_combined)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        if 'gradient_boosting' not in accuracy_results:
            accuracy_results['gradient_boosting'] = []
            f1_results['gradient_boosting'] = []
            
        accuracy_results['gradient_boosting'].append(accuracy)
        f1_results['gradient_boosting'].append(f1)
        
        print(f"  Best params: {best_params}")
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return accuracy_results, f1_results

def run_mnist():
    """Run experiments on MNIST dataset."""
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X / 255.0  # Normalize pixel values to [0,1]
    
    # Split into training and test sets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    # Use a smaller subset for faster training
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    
    results = {'classifier': [], 'accuracy': []}
    
    # Decision Tree
    print("Training Decision Tree on MNIST...")
    dt_clf = DecisionTreeClassifier(max_depth=20, random_state=42)
    dt_clf.fit(X_train, y_train)
    dt_accuracy = accuracy_score(y_test, dt_clf.predict(X_test))
    results['classifier'].append('decision_tree')
    results['accuracy'].append(dt_accuracy)
    print(f"Decision Tree accuracy: {dt_accuracy:.4f}")
    
    # Bagging
    print("Training Bagging on MNIST...")
    base_estimator = DecisionTreeClassifier(max_depth=20, random_state=42)
    bagging_clf = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=10,
        max_samples=0.5,
        random_state=42
    )
    bagging_clf.fit(X_train, y_train)
    bagging_accuracy = accuracy_score(y_test, bagging_clf.predict(X_test))
    results['classifier'].append('bagging')
    results['accuracy'].append(bagging_accuracy)
    print(f"Bagging accuracy: {bagging_accuracy:.4f}")
    
    # Random Forest
    print("Training Random Forest on MNIST...")
    rf_clf = RandomForestClassifier(
        n_estimators=10,
        max_depth=20,
        random_state=42
    )
    rf_clf.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf_clf.predict(X_test))
    results['classifier'].append('random_forest')
    results['accuracy'].append(rf_accuracy)
    print(f"Random Forest accuracy: {rf_accuracy:.4f}")
    
    # Gradient Boosting
    print("Training Gradient Boosting on MNIST...")
    gb_clf = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_clf.fit(X_train, y_train)
    gb_accuracy = accuracy_score(y_test, gb_clf.predict(X_test))
    results['classifier'].append('gradient_boosting')
    results['accuracy'].append(gb_accuracy)
    print(f"Gradient Boosting accuracy: {gb_accuracy:.4f}")
    
    return results

# Main function
def main():
    # Part 1-4: Run experiments on Boolean formula datasets
    print("="*50)
    print("Running experiments on Boolean formula datasets")
    print("="*50)
    
    # Run Decision Tree experiments
    print("\nRunning Decision Tree experiments...")
    dt_results = run_dt()
    
    # Run Bagging experiments
    print("\nRunning Bagging experiments...")
    bag_results = run_bagging()
    
    # Run Random Forest experiments
    print("\nRunning Random Forest experiments...")
    rf_results = run_rf()
    
    # Run Gradient Boosting experiments
    print("\nRunning Gradient Boosting experiments...")
    gb_results = run_gb()
    
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
    
    mnist_results = run_mnist()
    
    # Print MNIST summary
    print("\nMNIST Results Summary:")
    mnist_df = pd.DataFrame(mnist_results)
    print(mnist_df.to_string(index=False))
    
    # Identify best classifier
    best_idx = mnist_results['accuracy'].index(max(mnist_results['accuracy']))
    best_clf = mnist_results['classifier'][best_idx]
    print(f"\nBest classifier on MNIST: {best_clf} with accuracy {mnist_results['accuracy'][best_idx]:.4f}")

if __name__ == "__main__":
    main()