import re
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.utils import np_utils
from get_tweets import get_tweets

# Neural network part based on https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/

def clean(tweet, fields={'full_text', 'id'}):
    clean_tweet = {k:v for k,v in tweet.items() if k in fields}
    clean_tweet['is_retweet'] = 'retweeted_status' in tweet
    clean_tweet['text'] = re.sub(r"@(\w){1,15}", '', clean_tweet['full_text']).strip()
    return clean_tweet
    
def train_model(X, Y, layers=2, neurons=400, epochs=1, modelname="reverte", batch_size=100):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    for i in range(layers-1):
        model.add(LSTM(neurons))
        model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(Y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(X, Y, epochs=epochs, batch_size=batch_size)

    filename = [modelname] + ["{}_0.2".format(neurons)]*layers
    filename = '_'.join(filename)
    model.save_weights('models/{}.h5'.format(filename))
    return model

if __name__ == "__main__":
    # get_tweets("perezreverte", 4000, "reverte.json")

    with open("reverte.json", "r") as fin:
        tweets = json.loads(fin.read())

    clean_tweets = [clean(tweet) for tweet in tweets]
    corpus = [tweet['text'] for tweet in clean_tweets if not tweet['is_retweet']]
    full_text = '\n'.join(corpus).lower()

    characters = sorted(list(set(full_text)))
    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}

    X = []
    Y = []
    length = len(full_text)
    seq_length = 100

    for i in range(0, length-seq_length, 1):
        sequence = full_text[i:i + seq_length]
        label =full_text[i + seq_length]
        X.append([char_to_n[char] for char in sequence])
        Y.append(char_to_n[label])

    X_modified = np.reshape(X, (len(X), seq_length, 1))
    X_modified = X_modified / float(len(characters))
    Y_modified = np_utils.to_categorical(Y)

    model_config = {
        "layers": 2,
        "neurons": 512,
        "epochs": 100
    }
    model = train_model(X_modified, Y_modified, **model_config)

    string_mapped  = X[99]
    full_string = [n_to_char[value] for value in string_mapped]
    # generating characters
    for i in range(280):
        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
        x = x / float(len(characters))

        pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [n_to_char[value] for value in string_mapped]
        full_string.append(n_to_char[pred_index])

        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]

    print(''.join(full_string))