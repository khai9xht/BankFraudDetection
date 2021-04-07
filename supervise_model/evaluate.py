import sys
sys.path.append("/home/hoangnv68/BankFraudDetection")

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, \
        average_precision_score, recall_score, precision_score, classification_report
from utils import read_data, Downsample_data, Remove_ouliers, convert_data
def evaluate(log_reg, knears_neighbors, svc, tree_clf, X_test, y_test):
    y_pred_log_reg = log_reg.predict(X_test)

    # Other models fitted with UnderSampling
    y_pred_knear = knears_neighbors.predict(X_test)
    y_pred_svc = svc.predict(X_test)
    y_pred_tree = tree_clf.predict(X_test)


    log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
    kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
    svc_cf = confusion_matrix(y_test, y_pred_svc)
    tree_cf = confusion_matrix(y_test, y_pred_tree)

    fig, ax = plt.subplots(2, 2,figsize=(22,12))
    sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper)
    ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
    ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, cmap=plt.cm.copper)
    ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix", fontsize=14)
    ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(svc_cf, ax=ax[1][0], annot=True, cmap=plt.cm.copper)
    ax[1][0].set_title("Suppor Vector Classifier \n Confusion Matrix", fontsize=14)
    ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)

    sns.heatmap(tree_cf, ax=ax[1][1], annot=True, cmap=plt.cm.copper)
    ax[1][1].set_title("DecisionTree Classifier \n Confusion Matrix", fontsize=14)
    ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
    ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)


    plt.savefig("/home/hoangnv68/BankFraudDetection/image_evaluation/confusion_matrix.png")
    plt.cla()

    print('Logistic Regression:')
    print(classification_report(y_test, y_pred_log_reg))

    print('KNears Neighbors:')
    print(classification_report(y_test, y_pred_knear))

    print('Support Vector Classifier:')
    print(classification_report(y_test, y_pred_svc))

    print('DecisionTree Classifier:')
    print(classification_report(y_test, y_pred_tree))

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

    evaluate(log_reg, knears_neighbors, svc, tree_clf, X_test, y_test)