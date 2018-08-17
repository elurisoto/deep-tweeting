import re
import configparser
import tweepy
import json
from tqdm import tqdm
from deepwriter import DeepWriter

def api_login(key_cfg):
    "Logs on twitter with the given api keys"
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
    """Gets the last tweets from an user. 
    Twitter only allows downloading the last ~3400 tweets
    """
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


def clean(tweet, fields={'full_text', 'id'}):
    """Strips metadata that won't be used and removes @mentions from all tweets"""
    clean_tweet = {k:v for k,v in tweet.items() if k in fields}
    clean_tweet['is_retweet'] = 'retweeted_status' in tweet
    clean_tweet['text'] = re.sub(r"@(\w){1,15}", '', clean_tweet['full_text']).strip()
    return clean_tweet


def main():
    # get_tweets("perezreverte", 4000, "reverte.json")

    with open("reverte.json", "r") as fin:
        tweets = json.loads(fin.read())

    clean_tweets = [clean(tweet) for tweet in tweets]
    corpus = [tweet['text'] for tweet in clean_tweets if not tweet['is_retweet']]
    full_text = '\n'.join(corpus).lower()

    model_config = DeepWriter.vectorize(full_text)

    model_config.update({
        "layers": 1,
        "neurons": 150,
        "modelname": "reverte"
    })

    model = DeepWriter(**model_config)
    model.train(epochs=60)


if __name__ == "__main__":
    main()