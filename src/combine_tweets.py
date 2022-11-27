# from twitter_test import create_url
from twitter_test import bearer_oauth
from twitter_test import connect_to_endpoint
import pandas as pd
import numpy as np
import json
import csv
# from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import pickle 
# import re

# with open('Emoji_Dict.p', 'rb') as fp:
#     Emoji_Dict = pickle.load(fp)
# Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}

# def convert_emojis_to_word(text):
#     for emot in Emoji_Dict:
#         text = re.sub(r'('+emot+')', "_".join(Emoji_Dict[emot].replace(",","").replace(":","").split()), text)
#     return text

# def remove_emoji(string):
#     emoji_pattern = re.compile("["
#                            u"\U0001F600-\U0001F64F" # emoticons
#                            u"\U0001F300-\U0001F5FF" # symbols & pictographs
#                            u"\U0001F680-\U0001F6FF" # transport & map symbols
#                            u"\U0001F1E0-\U0001F1FF" # flags (iOS)
#                            u"\U00002702-\U000027B0"
#                            u"\U000024C2-\U0001F251"
#                            "]+", flags=re.UNICODE)
#     return emoji_pattern.sub(r'', string)
import time

df = pd.read_csv("src/data/twitter_data/dataset_tweets.csv")

# if there is not a column for tweets, create one
if "tweet" not in df.columns:
    print("CREATING NEW COLUMN")
    df["tweet"] = np.nan

# create string of 100 ids from csv

def create_id_string(n):
    ids = ""
    for i in range(n,n+100):
        ids += df["Id"][i].strip("'") + ","
    
    return ids[:-1]

def create_url(ids):

    tweet_fields = "tweet.fields=lang,author_id"
    id = "ids=" + ids
    # You can adjust ids to include a single Tweets.
    # Or you can add to up to 100 comma-separated IDs
    url = "https://api.twitter.com/2/tweets?{}&{}".format(id, tweet_fields)
    return url

def append_tweet(n1,n2):

    tweets = {}

    for i in range(n1,n2,100):
        
        ids = create_id_string(i)
        url = create_url(ids)

        json_response = connect_to_endpoint(url)
        
        data = json_response["data"]
        for tweet in data:
            text = tweet["text"]
      
            tweets[tweet["id"]] = text

    return tweets

# append to dataframe 

def append_to_df(n1,n2):

    tweets = append_tweet(n1,n2)

    np.save(f"src/data/twitter_data/numpydict/tweets{n1}_{n2}.npy", tweets)

    # match key value with id in df

    for i in range(n1, n2):
        if df["Id"][i].strip("'") in tweets:
            df["tweet"][i] = tweets[df["Id"][i].strip("'")]

# intervals = [(0,10000), (10000, 20000), (20000,30000), (30000,40000),(40000,50000),(50000,60000)]
intervals = [(59974, 62574)]
for i in intervals:
    append_to_df(i[0],i[1])
    # time.sleep(500)
    print(i[0],i[1])

df.to_csv("src/data/twitter_data/dataset_tweets_soon.csv", index = False)


"""
        for response in json_response:
            if "data" in response:
                print(response)
                # print(json.dumps(json_response, indent=4, sort_keys=True))
                # tweet = response["data"][0]["text"]
                # tweets += tweet,
            else:
                tweets += np.nan,

        break


    df["tweet"][:len(tweets)] = tweets


    print(df.head())

    df.to_csv("src/data/twitter_data/dataset_tweets.csv", index=False)
"""