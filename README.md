# Perez Rebote
Script that downloads (or uses an already downloaded list) tweets from a twitter user and tries to imitate their tweeting style using LSTM networks.

## Installation
Considering python 3.6 is already installed, this should be enough:

`pip install -r requirements.txt`

## Usage
Run `perezrebote.py` with the following parameters:
```
Options:
  -u, --username TEXT         Twitter handle to learn from (without @).Tweets
                              will be stored in a json file
  -t, --tweets-json FILENAME  List of tweets already donloaded. Overrides
                              --username
  -l, --layers INTEGER        Number of layers of the neural network
                              [default: 1]
  -n, --neurons INTEGER       Number of neurons in each layer  [default: 150]
  -e, --epochs INTEGER        Number of epochs to train for  [default: 60]
  -k, --keys TEXT             File where the api keys are stored  [default:
                              api-key.cfg]
  --help                      Show this message and exit.
```

### Example:

`python perezrebote.py --user perezreverte --layers 2 --neurons 150`

## Setting up the api-key.cfg file
The script uses cfg file containing Twitter credentials.
The file's format should be:
```
[general]
consumer_key: <consumer_key>
consumer_secret: <consumer_secret>
access_token: <access_token>
access_token_secret: <access_token_secret>
```