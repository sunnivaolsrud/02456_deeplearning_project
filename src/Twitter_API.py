import requests
import pandas as pd
import numpy as np
import time
import os

# Twitter authentication
bearer_token = "aint none of yo business"

def bearer_oauth(r):
    """Authentication header"""
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    # r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

def connect_to_endpoint(url):
    """Call to API with json response"""
    response = requests.get(url, auth=bearer_oauth)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def get_all_tweets():
    data = pd.read_csv('df.csv')

    tweets = []
    
    for i, id in enumerate(data['id']):
        url = f"https://api.twitter.com/2/tweets/{id}"
        try: 
            json_response = connect_to_endpoint(url)
            tweets.append(json_response['data']['text'])
        except:
            tweets.append(np.nan)
            print(f"error at {i}")

    data['tweet'] = tweets
    data.dropna(inplace=True)
    data.to_csv('data/twitter_data/Tweets.csv', index=False)
    
    return True

if __name__ == "__main__":
    get_all_tweets()
