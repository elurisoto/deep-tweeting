import re
import configparser
import tweepy
import json
import click
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

def get_tweets(twitter_handle, num_tweets, key_cfg="api-key.cfg"):
    """Gets the last tweets from an user. 
    Twitter only allows downloading the last ~3400 tweets
    """
    try:
        api = api_login(key_cfg)
    except tweepy.error.TweepError as e:
        print("[ERROR] Problem authenticating")
        print(e)
        exit(1)

    print("Donwnloading last tweets from @{}".format(twitter_handle))
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

    with open("data/{}.json".format(twitter_handle), 'w') as fout:
        json.dump(tweets, fout)

    return tweets


def clean(tweet, fields={'full_text', 'id'}):
    "Strips metadata that won't be used and removes @mentions from all tweets"
    clean_tweet = {k:v for k,v in tweet.items() if k in fields}
    clean_tweet['is_retweet'] = 'retweeted_status' in tweet
    clean_tweet['text'] = re.sub(r"@(\w){1,15}", '', clean_tweet['full_text']).strip()
    return clean_tweet


def print_help_msg(command):
    "Prints CLI help message"
    with click.Context(command) as ctx:
        click.echo(command.get_help(ctx))

@click.command()
@click.option("--username", "-u", 
              help=("Twitter handle to learn from (without @)."
                    "Tweets will be stored in a json file"), 
              default=None)
@click.option("--tweets-json", "-t", 
              help="List of tweets already donloaded. Overrides --username", 
              type=click.File(),
              default=None)
@click.option("--layers", "-l", 
              help="Number of layers of the neural network", 
              default=1, 
              type=int,
              show_default=True)
@click.option("--neurons", "-n", 
              help="Number of neurons in each layer", 
              default=150, 
              type=int,
              show_default=True)
@click.option("--epochs", "-e", 
              help="Number of epochs to train for", 
              default=60, 
              type=int,
              show_default=True)
@click.option("--keys", "-k", 
              help="File where the api keys are stored", 
              default="api-key.cfg",
              show_default=True)
def main(username, tweets_json, layers, neurons, epochs, keys):
    if tweets_json:
        tweets = json.loads(tweets_json.read())
        username, _ = tweets_json.name.split('.')
    elif username:
        tweets = get_tweets(username, 4000, keys)
    else:
        print("\n[ERROR] You must specify a twitter username or a tweets json file.\n")
        print_help_msg(main)
        exit(1)

    clean_tweets = [clean(tweet) for tweet in tweets]
    corpus = [tweet['text'] for tweet in clean_tweets if not tweet['is_retweet']]
    full_text = '\n'.join(corpus).lower()

    model_config = DeepWriter.vectorize(full_text)

    model_config.update({
        "layers": layers,
        "neurons": neurons,
        "modelname": username
    })

    model = DeepWriter(**model_config)
    model.train(epochs=epochs)


if __name__ == "__main__":
    main()