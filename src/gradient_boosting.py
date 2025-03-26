from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from src.data_loader import load_dataset, combine_train_valid, get_all_dataset_configs

def train_gradient_boosting(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    Train and evaluate a Gradient Boosting classifier.
    
    Args:
        X_train, y_train: Training data
        X_valid, y_valid: Validation data
        X_test, y_test: Test data
        
    Returns:
        tuple: (best_params, accuracy, f1)
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.7, 0.8, 1.0]
    }
    
    # Create model
    gb_clf = GradientBoostingClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(gb_clf, param_grid, cv=5, scoring='accuracy')
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
    
    return best_params, accuracy, f1

def run_gradient_boosting_experiments():
    """
    Run Gradient Boosting experiments on all datasets.
    
    Returns:
        tuple: (accuracy_results, f1_results, best_params_dict)
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
        
        print(f"Running Gradient Boosting on dataset {dataset_name}")
        
        # Load data
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(clause_count, data_size)
        
        # Train and evaluate
        best_params, accuracy, f1 = train_gradient_boosting(X_train, y_train, X_valid, y_valid, X_test, y_test)
        
        # Store results
        if 'gradient_boosting' not in accuracy_results:
            accuracy_results['gradient_boosting'] = []
            f1_results['gradient_boosting'] = []
            
        accuracy_results['gradient_boosting'].append(accuracy)
        f1_results['gradient_boosting'].append(f1)
        best_params_dict[dataset_name] = best_params
        
        print(f"  Best params: {best_params}")
        print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    return accuracy_results, f1_results, best_params_dict