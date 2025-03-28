import os
import pandas as pd
from src.decision_tree import run_dt
from src.bagging import run_bagging
from src.random_forest import run_rf
from src.gradient_boosting import run_gb
from src.mnist import run_mnist


def combine_train_valid(X_train, y_train, X_valid, y_valid):
    X_train_valid = pd.concat([X_train, X_valid])
    y_train_valid = pd.concat([y_train, y_valid])
    return X_train_valid, y_train_valid


def merge_results(dt_results, bag_results, rf_results, gb_results):
    #generate  tables for project report 

    #for accuracy table
    accuracy_results = {'dataset': dt_results[0]['dataset']}
    accuracy_results['decision_tree'] = dt_results[0]['decision_tree']
    accuracy_results['bagging'] = bag_results[0]['bagging']
    accuracy_results['random_forest'] = rf_results[0]['random_forest']
    accuracy_results['gradient_boosting'] = gb_results[0]['gradient_boosting']
    
    #for f1 table
    f1_results = {'dataset': dt_results[1]['dataset']}
    f1_results['decision_tree'] = dt_results[1]['decision_tree']
    f1_results['bagging'] = bag_results[1]['bagging']
    f1_results['random_forest'] = rf_results[1]['random_forest']

    f1_results['gradient_boosting'] = gb_results[1]['gradient_boosting']
    return accuracy_results, f1_results

def process_results(accuracy_results, f1_results):
    acc_df = pd.DataFrame(accuracy_results) 
    f1_df = pd.DataFrame( f1_results)
    
    print(f"classification accuracy table: + \n{acc_df.to_string(index=False)}")
    print(f"f1 table: + \n{f1_df.to_string(index=False)}")
    
def main():
    
    print("running experiments 1-4 !!!")
    dt_results = run_dt() #decision tree
    bag_results = run_bagging() #bagging
    rf_results = run_rf()#random forest
    gb_results = run_gb() #gradient boosting
    
    print("processing results...")
    accuracy_results, f1_results = merge_results(dt_results, bag_results, rf_results, gb_results)
    process_results(accuracy_results,  f1_results)
    
    print("running MNIST experiments !!!")    
    mnist_results = run_mnist()
    mnist_df = pd.DataFrame(mnist_results)
    print(mnist_df.to_string(index=False))

if __name__ == "__main__":
    main()