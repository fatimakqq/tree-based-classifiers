# src/decision_tree.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from src.utils import load_dataset, combine_train_valid, get_all_dataset_configs

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