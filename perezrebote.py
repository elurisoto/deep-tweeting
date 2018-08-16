import configparser
import tweepy
from tqdm import tqdm
from pandas.io.json import json_normalize

def get_tweets(twitter_handle, num_tweets, csv_file=None, key_cfg="api-key.cfg"):

    keys = configparser.ConfigParser()
    keys.read(key_cfg)

    access_token = keys.get("general", "access_token")
    access_token_secret = keys.get("general", "access_token_secret")
    consumer_key = keys.get("general", "consumer_key")
    consumer_secret = keys.get("general", "consumer_secret")

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    tweets = []
    last_tweet = None
    with tqdm(total=num_tweets) as pbar:
        while len(tweets) < num_tweets:
            new_tweets = api.user_timeline(twitter_handle, 
                                           count=(num_tweets-len(tweets))%200,
                                           max_id=last_tweet,
                                           tweet_mode='extended')
            if len(new_tweets) > 0:
                tweets.extend(new_tweets)
                pbar.update(len(new_tweets))
                last_tweet = tweets[-1]._json["id"]
            else:
                break

    tweets = [tweet._json for tweet in tweets]

    df = json_normalize(tweets)
    if csv_file is not None:
        df.to_csv(csv_file)
    
    return df

if __name__ == "__main__":
    print(get_tweets("perezreverte", 20, "reverte.csv").head())