import pandas as pd
import numpy as np


def csv_string():
    df = pd.read_csv("src/data/twitter_data/dataset.csv")

    for i in range(0, len(df)):
        if df["Id"][i][0] == "'":
            df["Id"][i] = df["Id"][i]+"'"
        else:
            df["Id"][i] = np.nan

    print(df.head())

    df = df.dropna()

    df.to_csv("src/data/twitter_data/dataset_str.csv", index=False)


def remove_index():
    # Remove first three columns
    df = pd.read_csv("src/data/twitter_data/labels_and_tweets.csv")
    df = df.drop(df.columns[[0, 1, 2]], axis=1)
    df.to_csv("src/data/twitter_data/labels_and_tweets.csv", index=False)

# rename column to Tweet and remove rows with nan

def rename_column():
    df = pd.read_csv("src/data/twitter_data/labels_and_tweets.csv")
    df = df.rename(columns={"tweet": "Tweet"})
    df = df.dropna()
    df.to_csv("src/data/twitter_data/labels_tweets_.csv", index=False)

