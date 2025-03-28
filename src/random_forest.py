from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from src.utils import load_dataset, combine_train_valid, get_all_dataset_configs

def run_rf():
    accuracy_results = {'dataset': []}
    f1_results = {'dataset': []}
    configs = get_all_dataset_configs()
    
    for clause, size in configs:
        dataset_name = f"c{clause}_d{size}"
        accuracy_results['dataset'].append(dataset_name)
        f1_results['dataset'].append(dataset_name)
        print(f"running random forest on {dataset_name}")
        
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(clause, size)
        #hyperparams
        param_grid = {
            'n_estimators': [10],
            'max_depth': [None, 10],
            'max_features': ['sqrt']
        }
        #grid search for best params
        rf_clf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        
        X_combined, y_combined = combine_train_valid(X_train, y_train, X_valid, y_valid)
        best_model = RandomForestClassifier(**best_params, random_state=42)
        best_model.fit(X_combined, y_combined)
        
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        #store results
        if 'random_forest' not in accuracy_results:
            accuracy_results['random_forest'] = []
            f1_results['random_forest'] = []
            
        accuracy_results['random_forest'].append(accuracy)
        f1_results['random_forest'].append(f1)
        
        print(f"best params  {best_params}")
        print(f"accuracy {accuracy:.4f}, F1: {f1:.4f}")
    
    return accuracy_results, f1_results