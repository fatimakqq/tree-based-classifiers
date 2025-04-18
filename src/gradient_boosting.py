from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from src.utils import load_dataset, combine_train_valid, get_all_dataset_configs

def run_gb():
    accuracy_results = {'dataset': []}
    f1_results = {'dataset': []}
    configs = get_all_dataset_configs()
    
    for clause_count, data_size in configs:
        dataset_name = f"c{clause_count}_d{data_size}"
        accuracy_results['dataset'].append(dataset_name)
        f1_results['dataset'].append(dataset_name)
        
        print(f"gradient boosting for  {dataset_name }")
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(clause_count, data_size)
        
        #hyperparamts
        param_grid = {
            'n_estimators': [50],
            'learning_rate': [0.1],
            'max_depth': [3, 5]
        }
        #grid search for best params
        gb_clf = GradientBoostingClassifier(random_state= 42)
        grid_search = GridSearchCV(gb_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        
        X_combined, y_combined = combine_train_valid(X_train, y_train, X_valid, y_valid)
        
        #retrain
        best_model = GradientBoostingClassifier(**best_params, random_state=42)
        best_model.fit(X_combined, y_combined)
        
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 =  f1_score(y_test, y_pred)
        
        if 'gradient_boosting' not in accuracy_results:
            accuracy_results['gradient_boosting'] = []
            f1_results['gradient_boosting'] = []
            
        accuracy_results['gradient_boosting'].append(accuracy)
        f1_results['gradient_boosting'].append(f1)
        
        print(f"best params  {best_params}")
        print(f"accuracy {accuracy:.4f}, F1: {f1:.4f}")
    return accuracy_results, f1_results