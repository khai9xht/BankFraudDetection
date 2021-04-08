import sys
sys.path.append("/home/hoangnv68/BankFraudDetection")

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from utils import Downsample_data, Remove_ouliers, read_data

def Clustering(X:pd.core.frame.DataFrame):
    print("\nClustering.\n")
    #T-SNE implementation
    t0 = time.time()
    X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
    t1 = time.time()
    print("T-SNE took {:.2} s".format(t1 - t0))
    
    # PCA Implementation
    t0 = time.time()
    X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
    t1 = time.time()
    print("PCA took {:.2} s".format(t1 - t0))

    # TruncatedSVD
    t0 = time.time()
    X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
    t1 = time.time()
    print("Truncated SVD took {:.2} s".format(t1 - t0))
    print("-"*80)
    return X_reduced_tsne, X_reduced_pca, X_reduced_svd

def visualize_clustering(X_reduced_tsne, X_reduced_pca, X_reduced_svd, y):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
    f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


    blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
    red_patch = mpatches.Patch(color='#AF0000', label='Fraud')
    # t-SNE scatter plot
    ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
    ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
    ax1.set_title('t-SNE', fontsize=14)

    ax1.grid(True)

    ax1.legend(handles=[blue_patch, red_patch])


    # PCA scatter plot
    ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
    ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
    ax2.set_title('PCA', fontsize=14)

    ax2.grid(True)

    ax2.legend(handles=[blue_patch, red_patch])

    # TruncatedSVD scatter plot
    ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
    ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
    ax3.set_title('Truncated SVD', fontsize=14)

    ax3.grid(True)

    ax3.legend(handles=[blue_patch, red_patch])

    plt.savefig("/home/hoangnv68/BankFraudDetection/image_evaluation/clustering.png")

if __name__ == "__main__":
    path = "/home/hoangnv68/BankFraudDetection/creditcard.csv"
    df = read_data(path)
    sub_df = Downsample_data(df)
    sub_df = Remove_ouliers(sub_df, ["V14", "V12", "V10"], 25, 75, 1.5)
    X = sub_df.drop("Class", axis=1)
    y = sub_df["Class"]
    
    X_reduced_tsne, X_reduced_pca, X_reduced_svd = Clustering(X)
    visualize_clustering(X_reduced_tsne, X_reduced_pca, X_reduced_svd, y)
