import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split, StratifiedKFold, \
                ShuffleSplit, learning_curve
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

def plot_learning_curve_per_ax():
    print("Learning curve completely.")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print("train size: ", train_sizes)
    print("train_scores_mean: ", train_scores_mean)
    print("train_scores_std: ", train_scores_std)
    print("test_scores_mean: ", test_scores_mean)
    print("test_scores_std: ", test_scores_std)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    print('-'*60)
    
def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20,14), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    print('-'*60)
    print("Logistic Regression")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    print("Learning curve completely.")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print("train size: ", train_sizes)
    print("train_scores_mean: ", train_scores_mean)
    print("train_scores_std: ", train_scores_std)
    print("test_scores_mean: ", test_scores_mean)
    print("test_scores_std: ", test_scores_std)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    print('-'*60)

    # Second Estimator 
    print('-'*60)
    print("Knears neighbors ...")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    print("Learning curve completely.")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print("train size: ", train_sizes)
    print("train_scores_mean: ", train_scores_mean)
    print("train_scores_std: ", train_scores_std)
    print("test_scores_mean: ", test_scores_mean)
    print("test_scores_std: ", test_scores_std)
    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax2.set_title("Knears Neighbors Learning Curve", fontsize=14)
    ax2.set_xlabel('Training size (m)')
    ax2.set_ylabel('Score')
    ax2.grid(True)
    ax2.legend(loc="best")
    print('-'*60)

    # Third Estimator
    print('-'*60)
    print("Suport Vector Classifier ...")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    print("Learning curve completely.")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print("train size: ", train_sizes)
    print("train_scores_mean: ", train_scores_mean)
    print("train_scores_std: ", train_scores_std)
    print("test_scores_mean: ", test_scores_mean)
    print("test_scores_std: ", test_scores_std)
    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax3.set_title("Support Vector Classifier \n Learning Curve", fontsize=14)
    ax3.set_xlabel('Training size (m)')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    ax3.legend(loc="best")
    print('-'*60)

    # Fourth Estimator
    print('-'*60)
    print("Secision Tree Classifier ...")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    print("Learning curve completely.")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print("train size: ", train_sizes)
    print("train_scores_mean: ", train_scores_mean)
    print("train_scores_std: ", train_scores_std)
    print("test_scores_mean: ", test_scores_mean)
    print("test_scores_std: ", test_scores_std)
    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax4.set_title("Decision Tree Classifier \n Learning Curve", fontsize=14)
    ax4.set_xlabel('Training size (m)')
    ax4.set_ylabel('Score')
    ax4.grid(True)
    ax4.legend(loc="best")
    print('-'*60)
    plt.savefig("/home/hoangnv68/BankFraudDetection/image_evaluation/learningCurve.png")
    plt.cla()
    print("visualize learning in BankFraudDetection/image_evaluation/learningCurve.png")

def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, \
        tree_fpr, tree_tpr, y_train, log_reg_pred, knears_pred, svc_pred, tree_pred):
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
    plt.cla()
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