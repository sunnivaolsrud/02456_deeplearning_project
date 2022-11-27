import requests
import os
import json

# To set your enviornment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
# bearer_token = os.environ.get("BEARER_TOKEN")

api_key = "Mr9hJSpj5urLM6GJ1KREuS5hD"
api_key_secret = "VfTIBQK6yjJCMwJDgz6jQrxylk5l5JWzG5I390B138OwNV9wN4"
bearer_token = "AAAAAAAAAAAAAAAAAAAAAKlRXgEAAAAAaBO7QTi%2BGUZjOrpSmn8Cxk6w2YA%3DHreYQOVG03Ea17jKjAJwRYmvnrBzEewHoFfcWWMlkCg2Bfh1kC"
access_token = "4333141396-ZUiIlJXVrXOhwBfXmTDM4IphThzcAesVPYroluA"
acces_token_secret =  "PPRvRJ3vXC9So6ny9yEJSxVRNopnk37R4Io9HcHLRVWui"


def create_url():
    tweet_fields = "tweet.fields=lang,author_id"
    ids = "ids=1132713774773751808"
    # You can adjust ids to include a single Tweets.
    # Or you can add to up to 100 comma-separated IDs
    url = "https://api.twitter.com/2/tweets?{}&{}".format(ids, tweet_fields)
    return url


def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r


def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    # print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()

def main():
    url = create_url()
    json_response = connect_to_endpoint(url)
    print(json.dumps(json_response, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()
