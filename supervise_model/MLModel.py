import sys
sys.path.append("/home/hoangnv68/BankFraudDetection")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, \
            StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, \
                    roc_auc_score, accuracy_score, classification_report
import numpy as np
import pandas as pd
from utils import Downsample_data, Remove_ouliers, read_data, convert_data
import joblib 
import os

def train(X_train, X_test, y_train, y_test, save_path):        
    for key, classifier in classifiers.items():
        print("-"*80)
        print(f"Start training {key}:")
        classifier.fit(X_train, y_train)
        training_score = cross_val_score(classifier, X_train, y_train, cv=5)
        print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", \
                    round(training_score.mean(), 2) * 100, "% accuracy score")
        print("-"*80)
        
        name_model = key + ".joblib"
        model_path = os.path.join(save_path,name_model)
        joblib.dump(classifier, model_path)

    # Logistic Regression 
    log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
    grid_log_reg.fit(X_train, y_train)
    # We automatically get the logistic regression with the best parameters.
    log_reg = grid_log_reg.best_estimator_

    knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

    grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
    grid_knears.fit(X_train, y_train)
    # KNears best estimator
    knears_neighbors = grid_knears.best_estimator_

    # Support Vector Classifier
    svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
    grid_svc = GridSearchCV(SVC(), svc_params)
    grid_svc.fit(X_train, y_train)

    # SVC best estimator
    svc = grid_svc.best_estimator_

    # DecisionTree Classifier
    tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
                "min_samples_leaf": list(range(5,7,1))}
    grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
    grid_tree.fit(X_train, y_train)

    # tree best estimator
    tree_clf = grid_tree.best_estimator_

    # Overfitting Case

    log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
    print('Logistic Regression Cross Validation Score: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')


    knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
    print('Knears Neighbors Cross Validation Score', round(knears_score.mean() * 100, 2).astype(str) + '%')

    svc_score = cross_val_score(svc, X_train, y_train, cv=5)
    print('Support Vector Classifier Cross Validation Score', round(svc_score.mean() * 100, 2).astype(str) + '%')

    tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
    print('DecisionTree Classifier Cross Validation Score', round(tree_score.mean() * 100, 2).astype(str) + '%')



if __name__ =="__main__":
    path = "/home/hoangnv68/BankFraudDetection/creditcard.csv"
    df = read_data(path)
    sub_df = Downsample_data(df)
    sub_df = Remove_ouliers(sub_df, ["V14", "V12", "V10"], 25, 75, 1.5)

    X = sub_df.drop("Class", axis=1)
    y = sub_df["Class"]

    X_train, X_test, y_train, y_test = convert_data(X, y)
    classifiers = {
        "LogisiticRegression": LogisticRegression(max_iter=1000),
        "KNearest": KNeighborsClassifier(),
        "Support Vector Classifier": SVC(),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }

    save_path = "/home/hoangnv68/BankFraudDetection/supervise_model/pretrained_model"
    train(X_train, X_test, y_train, y_test, save_path)
