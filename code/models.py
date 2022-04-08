from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def evaluate(y_test, y_pred):
    scores = {}
    scores["Accuracy"] = accuracy_score(y_test, y_pred)
    scores["F1 score"] = f1_score(y_test, y_pred, average='weighted')
    scores["Recall"] = recall_score(y_test, y_pred, average='weighted')
    scores["Precision"] = precision_score(y_test, y_pred, average='weighted')
    return scores

def RandomForest(X_train, y_train, X_test, y_test):
    clf= RandomForestClassifier(
        n_estimators=200, 
        criterion='gini', 
        min_samples_split=2, 
        min_samples_leaf=1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores = evaluate(y_test, y_pred)
    return scores

def SVM(X_train, y_train, X_test, y_test):
    clf = svm.LinearSVC(
        penalty='l2', 
        loss='squared_hinge',
        dual=True,
        tol=0.0001,
        C=1.0,
        multi_class='ovr',
        fit_intercept=True,
        intercept_scaling=1,
        max_iter=1000
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores = evaluate(y_test, y_pred)
    return scores

def XGBoost(X_train, y_train, X_test, y_test):
    clf = XGBClassifier(
        objective='multi:softmax',
        booster='gbtree',
        verbosity=0,
        silent=0,
        learning_rate=0.1,
        use_label_encoder=False
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores = evaluate(y_test, y_pred)
    return scores
    