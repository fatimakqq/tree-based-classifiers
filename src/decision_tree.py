from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from src.data_loader import load_dataset, combine_train_valid, get_all_dataset_configs

def train_decision_tree(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    Train and evaluate a Decision Tree classifier.
    
    Args:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data
        
    Returns:
        tuple: (best_params, accuracy, f1)
    """
    # Define parameter grid
    param_grid = {
        'criterion': ['gini'],
        'max_depth': [None, 10],
        'min_samples_split': [2]
    }
    
    # Create model
    dt_clf = DecisionTreeClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(dt_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    
    # Combine training and validation sets
    X_combined, y_combined = combine_train_valid(X_train, y_train, X_valid, y_valid)
    
    # Retrain with best parameters on combined data
    best_model = DecisionTreeClassifier(**best_params, random_state=42)
    best_model.fit(X_combined, y_combined)
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return best_params, accuracy, f1

def run_dt():
    """
    Run Decision Tree experiments on all datasets.
    
    Returns:
        tuple: (accuracy_results, f1_results)
    """
    # Initialize results dictionaries
    accuracy_results = {'dataset': []}
    f1_results = {'dataset': []}
    best_params_dict = {}
    
    # Get all dataset configs
    configs = get_all_dataset_configs()
    
    for clause_count, data_size in configs:
        dataset_name = f"c{clause_count}_d{data_size}"
        accuracy_results['dataset'].append(dataset_name)
        f1_results['dataset'].append(dataset_name)
        
        print(f"Running Decision Tree on dataset {dataset_name}")
        
        # Load data
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(clause_count, data_size)
        
        # Train and evaluate
        best_params, accuracy, f1 = train_decision_tree(X_train, y_train, X_valid, y_valid, X_test, y_test)
        
        # Store results
        if 'decision_tree' not in accuracy_results:
            accuracy_results['decision_tree'] = []
            f1_results['decision_tree'] = []
            
        accuracy_results['decision_tree'].append(accuracy)
        f1_results['decision_tree'].append(f1)
        best_params_dict[dataset_name] = best_params
        
        print(f"  Best params: {best_params}")
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return accuracy_results, f1_results, best_params_dict