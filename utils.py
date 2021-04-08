import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from imblearn.under_sampling import NearMiss
from sklearn.metrics import roc_auc_score

def read_data(path):
    df = pd.read_csv(path)
    return df

def Downsample_data(df:pd.core.frame.DataFrame):
    Fraud = df.loc[df["Class"]==1]
    NonFraud = df.loc[df["Class"]==0].sample(len(Fraud))
    # NonFraud = df.loc[df["Class"]==0][:492]
    new_df = pd.concat([Fraud, NonFraud])
    new_df = new_df.sample(frac=1, random_state=42)
    return new_df

def Remove_ouliers(df, fields, min_thresh, max_thresh, r):
    print("\nStarting remove outliers.")
    print("-"*80)
    for field in fields:
        field_fraud = df[field].loc[df["Class"]==1].values
        min_th, max_th = np.percentile(field_fraud, min_thresh), \
                                np.percentile(field_fraud, max_thresh)
        field_iqr = max_th - min_th
        field_cut_off = field_iqr * r
        field_lower, field_upper = min_th - field_cut_off, max_th + field_cut_off
        df = df.drop(df[(df[field]>field_upper) | (df[field]<field_lower)].index)

        print(f"field: {field}")
        print(f"Quartile {min_thresh}: {min_th}\t|\tQuartile {max_thresh}: {max_th}")
        print(f"iqr: {field_iqr}\t\t|\tCut Off: {field_cut_off}")
        print(f"{field} Lower: {field_lower}\t|\t{field} Upper: {field_upper}")
        outliers = [x for x in field_fraud if x<field_lower or x>field_upper]
        print(f"Feature {field} Outliers for Fraud Cases: {len(outliers)}")
        print(f"{field} Outliers:\n{outliers}")
        print(f"Number of Instances after outliers removal: {len(df)}")
        print("-"*80)

    return df

def convert_data(X, y, ratio=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)    
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    return X_train,  X_test, y_train, y_test

def plot_learning_curve_per_ax(estimator, ax, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(\
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print("train_sizes: ", train_sizes)
    print("train_scores_mean: ", train_scores_mean)
    print("train_scores_std: ", train_scores_std)
    print("test_scores_mean: ", test_scores_mean)
    print("test_scores_std: ", test_scores_std)

    lower_train = train_scores_mean - train_scores_std
    lower_test = test_scores_mean - test_scores_std
    upper_train = train_scores_mean + train_scores_std
    upper_test = test_scores_mean + test_scores_std
    ax.fill_between(train_sizes, lower_train, upper_train, alpha=0.1, color="r")
    ax.fill_between(train_sizes, lower_test, upper_test, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.set_xlabel('Training size (m)')
    ax.set_ylabel('Score')
    ax.grid(True)
    ax.legend(loc="best")
    print('-'*60)

def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.cla()
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    print('-'*60)
    print("Logistic Regression")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    plot_learning_curve_per_ax(estimator1, ax1, X, y)

    # Second Estimator 
    print('-'*60)
    print("Knears neighbors ...")
    ax2.set_title("K Nearest Neighbors Learning Curve", fontsize=14)
    plot_learning_curve_per_ax(estimator2, ax2, X, y)

    # Third Estimator
    print('-'*60)
    print("Support vector classifier ...")
    ax3.set_title("Support vector classifier Learning Curve", fontsize=14)
    plot_learning_curve_per_ax(estimator3, ax3, X, y)

    # Fourth Estimator
    print('-'*60)
    print("Decision Tree Classifier")
    ax4.set_title("Decision Tree Classifier Learning Curve", fontsize=14)
    plot_learning_curve_per_ax(estimator4, ax4, X, y)

    plt.savefig("/home/hoangnv68/BankFraudDetection/image_evaluation/learningCurve.png")
    print("visualize learning in BankFraudDetection/image_evaluation/learningCurve.png")

def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, \
        tree_fpr, tree_tpr, y_train, log_reg_pred, knears_pred, svc_pred, tree_pred):
    plt.cla()
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'\
            .format(roc_auc_score(y_train, log_reg_pred)))
    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'\
            .format(roc_auc_score(y_train, knears_pred)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'\
            .format(roc_auc_score(y_train, svc_pred)))
    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'\
            .format(roc_auc_score(y_train, tree_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    plt.savefig("/home/hoangnv68/BankFraudDetection/image_evaluation/ROCCurve.png")
    print("Visualize ROC Curve in /BankFraudDetection/image_evaluation/ROCCurve.png")

def undersample(df:pd.core.frame.DataFrame):
    sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    undersample_X = df.drop('Class', axis=1)
    undersample_y = df['Class']

    for train_index, test_index in sss.split(undersample_X, undersample_y):
        print("Train:", train_index, "Test:", test_index)
        undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
        undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]
        
    undersample_Xtrain = undersample_Xtrain.values
    undersample_Xtest = undersample_Xtest.values
    undersample_ytrain = undersample_ytrain.values
    undersample_ytest = undersample_ytest.values 

    undersample_accuracy = []
    undersample_precision = []
    undersample_recall = []
    undersample_f1 = []
    undersample_auc = []

    X_nearmiss, y_nearmiss = NearMiss().fit_sample(undersample_X.values, undersample_y.values)
    print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))
    
    for train, test in sss.split(undersample_Xtrain, undersample_ytrain):
        undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), log_reg)
        undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])
        undersample_prediction = undersample_model.predict(undersample_Xtrain[test])
        
        undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))
        undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))
        undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))
        undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))
        undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))  
    
    return  