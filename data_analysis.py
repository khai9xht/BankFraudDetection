import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import Downsample_data, read_data, Remove_ouliers, convert_data

def analysis_basic(df:pd.core.frame.DataFrame):
    print("example instances in data:\n", df.head())
    print("\ndescribe:\n", df.describe())
    print("\nNull value: ", df.isnull().sum().max())
    print("\nNumber of fields: \n", df.columns)

    class_dis = [x/len(df)*100 for x in df["Class"].value_counts()]
    print("\nclass distibution:\n\tNo Fraud: {:.2f}\t\t\tFraud: {:.2f}"\
                .format(class_dis[0], class_dis[1]))
    
    plt.cla()
    colors = ["#0101DF", "#DF0101"]
    sns.countplot('Class', data=df, palette=colors)
    plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
    plt.savefig("image_evaluation/class_distributions.png")
    plt.cla()

    sub_df = Downsample_data(df)
    # print(len(sub_df.loc[sub_df["Class"]==1]))
    sns.countplot('Class', data=sub_df, palette=colors)
    plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
    plt.savefig("image_evaluation/subdata_class_distributions.png")
    
def corr_matrix(df, sub_df):
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

def neg_corel(df, fields, Y_name):
    print('-'*80)
    print("Start visualizing Negative Correlation of fields vs Class")
    plt.cla()
    _, axes = plt.subplots(ncols=len(fields), figsize=(20, len(fields)))
    colors = ["#0101DF", "#DF0101"]
    for i, field in enumerate(fields):
        sns.boxplot(x=Y_name, y=field, data=df, palette=colors, ax=axes[i])
        axes[i].set_title(f"{field} vs {Y_name} Negative Correlation.")
    plt.savefig("/home/hoangnv68/BankFraudDetection/image_evaluation/Class_Neg_Correl.png")
    print("finish !\n visualization image in BankFraudDetection/image_evaluation/Class_Neg_Correl.png")
    print('-'*80)


def pos_corel(df, fields, Y_name):
    print('-'*80)
    print("Start visualizing Positive Correlationof fields vs Class")
    plt.cla()
    _, axes = plt.subplots(ncols=len(fields), figsize=(20, len(fields)))
    colors = ["#0101DF", "#DF0101"]
    for i, field in enumerate(fields):
        sns.boxplot(x=Y_name, y=field, data=df, palette=colors, ax=axes[i])
        axes[i].set_title(f"{field} vs {Y_name} Positive Correlation.")
    plt.savefig("/home/hoangnv68/BankFraudDetection/image_evaluation/Class_Pos_Correl.png")
    print("finish !\n visualization image in BankFraudDetection/image_evaluation/Class_Pos_Correl.png")
    print('-'*80)


if __name__ == "__main__":
    path = "/home/hoangnv68/BankFraudDetection/creditcard.csv"
    df = read_data(path)
    
    analysis_basic(df)
    

    sub_df = Downsample_data(df)
    sub_df = Remove_ouliers(sub_df, ["V14", "V12", "V10"], 25, 75, 1.5)

    X = sub_df.drop("Class", axis=1)
    y = sub_df["Class"]

    corr_matrix(df, sub_df)
    Y_name = "Class"
    neg_fields = ["V17", "V14", "V12", "V10"]
    pos_fields = ["V11", "V4", "V2", "V19"]
    neg_corel(sub_df, neg_fields, Y_name)
    pos_corel(sub_df, pos_fields, Y_name)