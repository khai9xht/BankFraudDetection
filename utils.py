import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split


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