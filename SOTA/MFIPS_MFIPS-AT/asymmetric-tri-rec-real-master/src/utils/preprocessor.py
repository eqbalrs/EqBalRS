"""
Codes for preprocessing datasets used in the real-world experiments
in the paper "Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback".
"""

import codecs
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_datasets(data: str, seed: int = 0) -> Tuple:
    print("Inside Preprocess datasets: "+str(data))
    """Load and preprocess raw datasets (Yahoo! R3 or Coat)."""

    #if data == 'yahoo':
       # with codecs.open(f'../data/{data}/train.txt', 'r', 'utf-8', errors='ignore') as f:
       #     data_train = pd.read_csv(f, delimiter='\t', header=None)
       #     data_train.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
        #with codecs.open(f'../data/{data}/test.txt', 'r', 'utf-8', errors='ignore') as f:
        #    data_test = pd.read_csv(f, delimiter='\t', header=None)
        #    data_test.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
        #for _data in [data_train, data_test]:
        #    _data.user, _data.item = _data.user - 1, _data.item - 1
   # elif data == 'coat':
      #  col = {'level_0': 'user', 'level_1': 'item', 2: 'rate', 0: 'rate'}
      #  with codecs.open(f'../data/{data}/train.ascii', 'r', 'utf-8', errors='ignore') as f:
       #     data_train = pd.read_csv(f, delimiter=' ', header=None)
       #     data_train = data_train.stack().reset_index().rename(columns=col)
       #     data_train = data_train[data_train.rate.values != 0].reset_index(drop=True)
       # with codecs.open(f'../data/{data}/test.ascii', 'r', 'utf-8', errors='ignore') as f:
        #    data_test = pd.read_csv(f, delimiter=' ', header=None)
         #   data_test = data_test.stack().reset_index().rename(columns=col)
         #   data_test = data_test[data_test.rate.values != 0].reset_index(drop=True)

    if data=='movie':
        data_train = pd.read_csv('../data/movie/train_movieLens.csv',delimiter=',')
        #data_train = data_train.drop(data_train.columns[[0]],axis=1)
        data_train = data_train
        print(data_train)
        data_test = pd.read_csv('../data/movie/valid_movieLens.csv',delimiter=',')
        #data_test = data_test.drop(data_test.columns[[0]],axis=1)
        data_test = data_test
        print(data_test)
    if data=='yahoo':
        data_train = pd.read_csv('../data/yahoo/train_yahoo.csv',delimiter=',')
        #data_train = data_train.drop(data_train.columns[[0]],axis=1)
        #data_train = data_train[:10000]
        print(data_train)
        data_test = pd.read_csv('../data/yahoo/valid_yahoo.csv',delimiter=',')
        #data_test = data_test.drop(data_test.columns[[0]],axis=1)
        print(data_test)

    if data=='pantry':
        data_train = pd.read_csv('../data/pantry/train_pantry.csv',delimiter=',')
        #data_train = data_train.drop(data_train.columns[[0]],axis=1)
        #data_train = data_train[:10000]
        print(data_train)
        data_test = pd.read_csv('../data/pantry/valid_pantry.csv',delimiter=',')
        #data_test = data_test.drop(data_test.columns[[0]],axis=1)
        print(data_test)
   
    if data=='gift_cards':
        data_train = pd.read_csv('../data/gift_cards/train_gift_cards.csv',delimiter=',')
        #data_train = data_train.drop(data_train.columns[[0]],axis=1)
        #data_train = data_train[:10000]
        print(data_train)
        data_test = pd.read_csv('../data/gift_cards/valid_gift_cards.csv',delimiter=',')
        #data_test = data_test.drop(data_test.columns[[0]],axis=1)
        print(data_test)

    #data_test=data_test[:10000]

    test = data_test.values
    train, val = data_train.values,data_test.values#train_test_split(data_train.values, test_size=0.1, random_state=seed)#data_train.values,data_test.values#
    num_users, num_items = train[:, 0].max() + 1, train[:, 1].max() + 1
    num_users_val, num_items_val = val[:, 0].max() + 1, val[:, 1].max() + 1
    num_users_test, num_items_test = test[:, 0].max() + 1, test[:, 1].max() + 1
    print("Train: "+str(num_users)+" "+str(num_items)+" Val: "+str(num_users_val)+" "+str(num_items_val)+" Test: "+str(num_users_test)+" "+str(num_items_test))

    return train, val, test, num_users, num_items
