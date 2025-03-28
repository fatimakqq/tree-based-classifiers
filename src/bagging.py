from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from main import load_dataset, combine_train_valid, get_all_dataset_configs

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