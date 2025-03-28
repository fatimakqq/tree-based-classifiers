from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from src.utils import load_dataset, combine_train_valid, get_all_dataset_configs

def run_dt():
    accuracy_results = {'dataset': []}
    f1_results = {'dataset': []}
    configs = get_all_dataset_configs()
    
    for clause, size in configs:
        dataset_name = f"c{clause}_d{size}"
        accuracy_results['dataset'].append(dataset_name)
        f1_results['dataset'].append(dataset_name)
        
        print(f"decision tree for  {dataset_name}")
        
        #load data
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset(clause, size)
        
        #hyperparams
        param_grid = {

            'criterion': ['gini'], #removed entropy
            'max_depth': [None,  10],
            'min_samples_split': [2]
        }
        
        #grid search for best parameters
        dt_clf = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(dt_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        
        #combine train + validsation sets
        X_combined, y_combined  = combine_train_valid(X_train, y_train, X_valid, y_valid)
        
        #train and eval on test set
        best_model = DecisionTreeClassifier(**best_params,  random_state=42)
        best_model.fit(X_combined, y_combined)
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        #store results
        if 'decision_tree' not in accuracy_results:
            accuracy_results['decision_tree'] = []
            f1_results['decision_tree'] = []
            
        accuracy_results['decision_tree'].append(accuracy)
        f1_results['decision_tree'].append(f1)
        
        print(f"best params  {best_params}")
        print(f"accuracy {accuracy:.4f}, F1: {f1:.4f}")
    
    return accuracy_results, f1_results