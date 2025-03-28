from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score #f1


from sklearn.model_selection import GridSearchCV

from src.utils import load_dataset, combine_train_valid, get_all_dataset_configs

def run_bagging():
    accuracy_results = {'dataset': []}
    f1_results = {'dataset': []}
    configs = get_all_dataset_configs()
    
    for clause, size in configs:
        dataset_name = f"c{clause}_d{size}"
        accuracy_results['dataset'].append(dataset_name)
        f1_results['dataset'].append(dataset_name)
        
        print(f"bagging for: {dataset_name}")
        
        #load data
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(clause, size)
        
        #hyperparameters
        param_grid = {
            'n_estimators': [10],
            'estimator__max_depth':  [None, 10]
        }
        
        #create  model
        base_estimator = DecisionTreeClassifier(random_state=42)
        bagging_clf =  BaggingClassifier(estimator=base_estimator, random_state=42)
        
        #grid search for best parameters
        grid_search = GridSearchCV(bagging_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        
        X_combined, y_combined = combine_train_valid(X_train, y_train, X_valid, y_valid)
        estimator_depth = best_params.pop('estimator__max_depth')
        
        #retrain w best params 
        base_estimator = DecisionTreeClassifier(max_depth=estimator_depth, random_state=42)
        best_model = BaggingClassifier(estimator=base_estimator, **best_params, random_state=42)
        best_model.fit(X_combined, y_combined)
        
        #evaluate on test data
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        best_params['estimator__max_depth'] = estimator_depth
        
        #store results
        if 'bagging' not in accuracy_results:
            accuracy_results['bagging'] = []
            f1_results['bagging'] = []
        accuracy_results[ 'bagging'].append(accuracy)
        f1_results['bagging'].append(f1)
        
        print(f"best params  {best_params}")
        print(f"accuracy {accuracy:.4f}, F1: {f1:.4f}")
    return accuracy_results, f1_results