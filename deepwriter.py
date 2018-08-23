import json
import time
import random
import re
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import Callback

# based on https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

class DeepWriter(Callback):
    def __init__(self, x, y, layers, neurons, max_len, char_indices, indices_char, chars, modelname, text, ):
        self.x = x
        self.y = y
        self.layers = layers
        self.neurons = neurons
        self.max_len = max_len
        self.char_indices = char_indices
        self.indices_char = indices_char
        self.chars = chars
        self.modelname = modelname
        self.text = text

        model = Sequential()
        model.add(LSTM(self.neurons, input_shape=(self.x.shape[1], self.x.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        for _ in range(self.layers-1):
            model.add(LSTM(self.neurons, return_sequences=True))
            model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(self.y.shape[1], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print("\nModel summary:")
        model.summary()
        self.model = model

    def train(self, epochs, batch_size=100):
        "Trains the model for the specified number of epochs"
        self.model.fit(self.x, self.y, epochs=epochs, batch_size=batch_size, callbacks=[self])
        filename = ([time.strftime("%Y%m%d-%H%M%S"), self.modelname] + 
                    ["{}_0.2".format(self.neurons)]*self.layers + 
                    ["{}_ep".format(epochs)])
        filename = '_'.join(filename)
        self.model.save_weights('models/{}.h5'.format(filename))

    @staticmethod
    def vectorize(text, max_len=40, step=1):
        "Converts the corpus to the structure required by the model"
        chars = sorted(list(set(text)))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        # cut the text in semi-redundant sequences of maxlen characters
        sentences = []
        next_chars = []
        for i in range(0, len(text) - max_len, step):
            sentences.append(text[i: i + max_len])
            next_chars.append(text[i + max_len])

        x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        return {
            'char_indices': char_indices, 
            'indices_char': indices_char, 
            'chars': chars,
            'x': x, 
            'y': y,
            'max_len': max_len,
            "text": text
        }

    @staticmethod
    def sample(preds, temperature=1.0):
        "Helper function to sample an index from a probability array"
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    def on_epoch_end(self, epoch, logs):
        "Function invoked at end of each epoch. Prints generated text."
        print()
        print('----- Generating text after Epoch: %d' % epoch)
        self.talk(400)

    def get_seed(self):
        "Gets a seed sentence. Only picks beginnings of lines"
        line_starts = [m.start() for m in re.finditer(r"^.", self.text, flags=re.M|re.S) 
                       if m.start() < len(self.text) - self.max_len - 1]
        start_index = random.choice(line_starts)
        return self.text[start_index: start_index + self.max_len]

    def talk(self, n_chars, diversity=1):
        "Generates a message of n_chars"
        if isinstance(diversity, list):
            diversities = diversity
        else:
            diversities = [diversity]

        sentence = self.get_seed()
        for diversity in diversities:
            generated = sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for _ in range(400):
                x_pred = np.zeros((1, self.max_len, len(self.chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, self.char_indices[char]] = 1.

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = DeepWriter.sample(preds, diversity)
                next_char = self.indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()