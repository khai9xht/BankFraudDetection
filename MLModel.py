from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, \
                    roc_auc_score, accuracy_score, classification_report
import numpy as np
import pandas as pd
from utils import Downsample_data, Remove_ouliers, read_data

def train(X:pd.core.frame.DataFrame, y:pd.core.frame.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    
    classifiers = {
    "LogisiticRegression": LogisticRegression(max_iter=1000),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
    }
    for key, classifier in classifiers.items():
        print("-"*80)
        print(f"Start training {key}:")
        classifier.fit(X_train, y_train)
        training_score = cross_val_score(classifier, X_train, y_train, cv=5)
        print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", \
                    round(training_score.mean(), 2) * 100, "% accuracy score")
        print("-"*80)


if __name__ =="__main__":
    path = "creditcard.csv"
    df = read_data(path)
    sub_df = Downsample_data(df)

    sub_df = Remove_ouliers(sub_df, ["V14", "V12", "V10"], 25, 75, 1.5)

    X = df.drop("Class", axis=1)
    y = df["Class"]
    train(X, y)