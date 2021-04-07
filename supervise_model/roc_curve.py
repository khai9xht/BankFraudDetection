import sys
sys.path.append("/home/hoangnv68/BankFraudDetection")

import joblib
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, \
        precision_recall_curve, recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import ShuffleSplit, learning_curve, cross_val_predict
from utils import plot_learning_curve, graph_roc_curve_multiple, \
            read_data, Downsample_data, Remove_ouliers, convert_data


def learn_curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train):
    print('-'*80)
    print("Start learning Curve ...")
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
    plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, \
                X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)
    print("Finish learning Curve !!!")
    print('-'*80)


def ROC_Curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train):
    print('-'*80)
    print("Cross validation")
    print('-'*60)
    print("Logistic Regression")
    log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,
                             method="decision_function")
    print('-'*60)
    print("K Neirest Neighbors")
    knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)
    print('-'*60)
    print("Support vector classifier")
    svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,
                                method="decision_function")
    print('-'*60)
    print("Decision Tree Classifier")
    tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)
    print('-'*60)

    print('Logistic Regression: ', roc_auc_score(y_train, log_reg_pred))
    print('KNears Neighbors: ', roc_auc_score(y_train, knears_pred))
    print('Support Vector Classifier: ', roc_auc_score(y_train, svc_pred))
    print('Decision Tree Classifier: ', roc_auc_score(y_train, tree_pred))

    print('-'*60)
    print("Calculating roc curve")
    log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
    knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
    svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
    tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)
    print('-'*60)
    print("Building graph ROC Curve")
    graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, \
        svc_fpr, svc_tpr, tree_fpr, tree_tpr, y_train, log_reg_pred, knears_pred, svc_pred, tree_pred)
    print('-'*80)


if __name__ == "__main__":
    path = "/home/hoangnv68/BankFraudDetection/creditcard.csv"
    df = read_data(path)
    sub_df = Downsample_data(df)
    sub_df = Remove_ouliers(sub_df, ["V14", "V12", "V10"], 25, 75, 1.5)

    X = sub_df.drop("Class", axis=1)
    y = sub_df["Class"]

    X_train, X_test, y_train, y_test = convert_data(X, y)

    log_reg = joblib.load("/home/hoangnv68/BankFraudDetection/supervise_model/pretrained_model/lgr_gridsearchcv.joblib")
    knears_neighbors = joblib.load("/home/hoangnv68/BankFraudDetection/supervise_model/pretrained_model/knn_gridsearchcv.joblib")
    svc = joblib.load("/home/hoangnv68/BankFraudDetection/supervise_model/pretrained_model/svc_gridsearchcv.joblib")
    tree_clf = joblib.load("/home/hoangnv68/BankFraudDetection/supervise_model/pretrained_model/dt_gridsearchcv.joblib")

    learn_curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train)
    ROC_Curve(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train)

    