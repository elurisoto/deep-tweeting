import configparser
import tweepy
import json
from tqdm import tqdm

def api_login(key_cfg):
    keys = configparser.ConfigParser()
    keys.read(key_cfg)

    access_token = keys.get("general", "access_token")
    access_token_secret = keys.get("general", "access_token_secret")
    consumer_key = keys.get("general", "consumer_key")
    consumer_secret = keys.get("general", "consumer_secret")

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    return tweepy.API(auth)

def get_tweets(twitter_handle, num_tweets, outfile=None, key_cfg="api-key.cfg"):
    api = api_login(key_cfg)
    tweets = []
    last_tweet = None
    with tqdm(total=num_tweets) as pbar:
        while len(tweets) < num_tweets:
            new_tweets = api.user_timeline(twitter_handle, 
                                           count=(num_tweets-len(tweets))%500,
                                           max_id=last_tweet,
                                           tweet_mode='extended')
            if len(new_tweets) > 0:
                tweets.extend(new_tweets)
                pbar.update(len(new_tweets))
                last_tweet = tweets[-1]._json["id"] - 1
            else:
                break

    print("Downloaded {} tweets.".format(len(tweets)))
    tweets = [tweet._json for tweet in tweets]

    if outfile:
        with open(outfile, 'w') as fout:
            json.dump(tweets, fout)

    return tweets