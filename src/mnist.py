from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def run_mnist():
    """Run experiments on MNIST dataset."""
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X / 255.0  # Normalize pixel values to [0,1]
    
    # Split into training and test sets
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    
    # Use a smaller subset for faster training
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    
    results = {'classifier': [], 'accuracy': []}
    
    # Decision Tree
    print("Training Decision Tree on MNIST...")
    dt_clf = DecisionTreeClassifier(max_depth=20, random_state=42)
    dt_clf.fit(X_train, y_train)
    dt_accuracy = accuracy_score(y_test, dt_clf.predict(X_test))
    results['classifier'].append('decision_tree')
    results['accuracy'].append(dt_accuracy)
    print(f"Decision Tree accuracy: {dt_accuracy:.4f}")
    
    # Bagging
    print("Training Bagging on MNIST...")
    base_estimator = DecisionTreeClassifier(max_depth=20, random_state=42)
    bagging_clf = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=10,
        max_samples=0.5,
        random_state=42
    )
    bagging_clf.fit(X_train, y_train)
    bagging_accuracy = accuracy_score(y_test, bagging_clf.predict(X_test))
    results['classifier'].append('bagging')
    results['accuracy'].append(bagging_accuracy)
    print(f"Bagging accuracy: {bagging_accuracy:.4f}")
    
    # Random Forest
    print("Training Random Forest on MNIST...")
    rf_clf = RandomForestClassifier(
        n_estimators=10,
        max_depth=20,
        random_state=42
    )
    rf_clf.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf_clf.predict(X_test))
    results['classifier'].append('random_forest')
    results['accuracy'].append(rf_accuracy)
    print(f"Random Forest accuracy: {rf_accuracy:.4f}")
    
    # Gradient Boosting
    print("Training Gradient Boosting on MNIST...")
    gb_clf = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_clf.fit(X_train, y_train)
    gb_accuracy = accuracy_score(y_test, gb_clf.predict(X_test))
    results['classifier'].append('gradient_boosting')
    results['accuracy'].append(gb_accuracy)
    print(f"Gradient Boosting accuracy: {gb_accuracy:.4f}")
    
    return results