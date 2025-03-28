from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier # ,GradientBoostingClassifier

from sklearn.metrics import accuracy_score

def run_mnist():
    print("loading MNIST dataset!!!")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    X = X / 255.0  #normalize pixels
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    X_train = X_train[:5000] #faster
    y_train = y_train[:5000]
    results = {'classifier': [], 'accuracy': []}
    
    #experiment 1
    print("training decision tree on MNIST")
    dt_clf = DecisionTreeClassifier(max_depth=20, random_state=42)
    dt_clf.fit(X_train, y_train)
    dt_accuracy = accuracy_score(y_test, dt_clf.predict(X_test))
    results['classifier'].append('decision_tree')
    results['accuracy'].append(dt_accuracy)
    print(f"DT accuracy {dt_accuracy:.4f}")

    #expermint 2
    print("training bagging on MNIST")
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
    print(f"Bagging accuracy{bagging_accuracy:.4f}")
    
    #experiment 3
    print("training Random Forest on MNIST ")
    rf_clf = RandomForestClassifier(
        n_estimators=10,
        max_depth=20,
        random_state=42
    )
    rf_clf.fit(X_train, y_train)
    rf_accuracy = accuracy_score(y_test, rf_clf.predict(X_test))
    results['classifier'].append('random_forest')
    results['accuracy'].append(rf_accuracy)
    print(f"RF accuracy {rf_accuracy:.4f}")
    
    #experiemnt 4
    print("training Gradient Boosting on MNIST")
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
    print(f"GB accuracy {gb_accuracy:.4f}")
    
    return results