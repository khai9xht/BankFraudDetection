import pandas as pd
import seaborn as sns


def downsample_data(df:pd.core.frame.DataFrame):
    # multi = int(len(df.loc[df["Class"]==0]) / len(df.loc[df["Class"]==1]))
    Fraud = df.loc[df["Class"]==1]
    NonFraud = df.loc[df["Class"]==0].sample(len(Fraud))
    new_df = pd.concat([Fraud, NonFraud])
    new_df = new_df.sample(frac=1, random_state=42)
    return new_df
