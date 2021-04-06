import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import downsample_data

def read_data(path):
    df = pd.read_csv(path)
    return df

def analysis_basic(df:pd.core.frame.DataFrame):
    print("example instances in data:\n", df.head())
    print("\ndescribe:\n", df.describe())
    print("\nNull value: ", df.isnull().sum().max())
    print("\nNumber of fields: \n", df.columns)

    class_dis = [x/len(df)*100 for x in df["Class"].value_counts()]
    print("\nclass distibution:\n\tNo Fraud: {:.2f}\t\t\tFraud: {:.2f}"\
                .format(class_dis[0], class_dis[1]))

    colors = ["#0101DF", "#DF0101"]
    sns.countplot('Class', data=df, palette=colors)
    plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
    plt.savefig("image_evaluation/class_distributions.png")
    plt.cla()

    sub_df = downsample_data(df)
    # print(len(sub_df.loc[sub_df["Class"]==1]))
    sns.countplot('Class', data=sub_df, palette=colors)
    plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
    plt.savefig("image_evaluation/subdata_class_distributions.png")
    plt.cla()

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

    # Entire DataFrame
    corr = df.corr()
    sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
    ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)

    sub_sample_corr = sub_df.corr()
    sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
    ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
    plt.savefig("image_evaluation/correlation_matrix.png")
    plt.cla()

if __name__ == "__main__":
    path = "/home/hoangnv68/BankFraudDetection/creditcard.csv"
    df = read_data(path)
    analysis_basic(df)