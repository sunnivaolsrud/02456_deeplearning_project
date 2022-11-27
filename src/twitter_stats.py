import pandas as pd
import numpy as np

# get general statistics about "twitter_data/labels_tweets_dropna.csv"

df = pd.read_csv(r'src/data/twitter_data/labels_tweets_dropna.csv')

# get the number of tweets per label
print(df.iloc[:,1:-1].sum())


print(df.iloc[:,1:-1].sum(axis=1).value_counts())

pos = df.iloc[:,1:-1].sum(axis=1) == 5

print(pos[pos>0])

print(df.iloc[24332])