import os, sys
import numpy as np
from random import shuffle
import argparse
from math import log, floor
import pandas as pd

def load_data():
    X_train = pd.read_csv(sys.argv[1], sep=',', header=0)
    X_train = np.array(X_train.values)
    X_train = np.delete(X_train , 1 , 1)
    X_train = np.concatenate((X_train,X_train**2), axis=1) #square
    #print(X_train.shape)
    Y_train = pd.read_csv(sys.argv[2], sep=',', header=0)
    Y_train = np.array(Y_train.values)
    #print(Y_train)
    X_test = pd.read_csv(sys.argv[3], sep=',', header=0)
    X_test = np.array(X_test.values)
    X_test = np.delete(X_test , 1 , 1)
    X_test = np.concatenate((X_test,X_test**2), axis=1)
    
    return (X_train, Y_train, X_test)

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

def valid(w, b, X_valid, Y_valid):
    valid_data_size = len(X_valid)

    z = (np.dot(X_valid, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Validation acc = %f' % (float(result.sum()) / valid_data_size))
    return

def infer(X_test, save_dir, output_dir):
    test_data_size = len(X_test)

    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    w = np.loadtxt(os.path.join(save_dir, 'w'))
    b = np.loadtxt(os.path.join(save_dir, 'b'))

    # predict
    z = (np.dot(X_test, np.transpose(w)) + b)
    y = sigmoid(z)
    y_ = np.around(y)

    print('=====Write output to %s =====' % output_dir)
    #if not os.path.exists(output_dir):
    #    os.mkdir(output_dir)
    #output_path = os.path.join(output_dir, 'log_prediction.csv')
    with open(output_dir, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return



X_all, Y_all, X_test = load_data()
#print(X_all.shape)
#print(Y_all.shape)
#print(X_test.shape)
X_all, X_test = normalize(X_all, X_test)
#train(X_all, Y_all, './output/')
infer(X_test , './log_parameters/' , sys.argv[4])
