from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def load_mnist_data():
    """
    Load and preprocess MNIST dataset.
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X / 255.0  # Normalize pixel values to [0,1]
    
    # Convert labels to numeric if they're not already
    if not isinstance(y.iloc[0], (int, float, np.number)):
        y = y.astype(int)
    
    # Split into training and test sets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    return X_train, y_train, X_test, y_test

def train_decision_tree_mnist(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Decision Tree on MNIST.
    
    Returns:
        float: Accuracy score
    """
    print("Training Decision Tree on MNIST...")
    clf = DecisionTreeClassifier(max_depth=20, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Decision Tree accuracy: {accuracy:.4f}")
    return accuracy

def train_bagging_mnist(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Bagging on MNIST.
    
    Returns:
        float: Accuracy score
    """
    print("Training Bagging on MNIST...")
    base_estimator = DecisionTreeClassifier(max_depth=20, random_state=42)
    clf = BaggingClassifier(
        estimator=base_estimator,  # Change from base_estimator to estimator
        n_estimators=50,
        max_samples=0.5,
        max_features=0.5,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Bagging accuracy: {accuracy:.4f}")
    return accuracy

def train_random_forest_mnist(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Random Forest on MNIST.
    
    Returns:
        float: Accuracy score
    """
    print("Training Random Forest on MNIST...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest accuracy: {accuracy:.4f}")
    return accuracy

def train_gradient_boosting_mnist(X_train, y_train, X_test, y_test):
    """
    Train and evaluate Gradient Boosting on MNIST.
    
    Returns:
        float: Accuracy score
    """
    print("Training Gradient Boosting on MNIST...")
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Gradient Boosting accuracy: {accuracy:.4f}")
    return accuracy

def run_mnist():
    """
    Run all classifier experiments on MNIST dataset.
    
    Returns:
        dict: Results dictionary
    """
    # Load MNIST data
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Initialize results dictionary
    results = {'classifier': [], 'accuracy': []}
    
    # Decision Tree
    dt_accuracy = train_decision_tree_mnist(X_train, y_train, X_test, y_test)
    results['classifier'].append('decision_tree')
    results['accuracy'].append(dt_accuracy)
    
    # Bagging
    bagging_accuracy = train_bagging_mnist(X_train, y_train, X_test, y_test)
    results['classifier'].append('bagging')
    results['accuracy'].append(bagging_accuracy)
    
    # Random Forest
    rf_accuracy = train_random_forest_mnist(X_train, y_train, X_test, y_test)
    results['classifier'].append('random_forest')
    results['accuracy'].append(rf_accuracy)
    
    # Gradient Boosting
    gb_accuracy = train_gradient_boosting_mnist(X_train, y_train, X_test, y_test)
    results['classifier'].append('gradient_boosting')
    results['accuracy'].append(gb_accuracy)
    
    return results