import os, sys
import pandas as pd
import numpy as np
from random import shuffle
from math import log, floor

def load_data():
    X_train = pd.read_csv(sys.argv[1], sep=',', header=0)
    X_train = np.array(X_train.values)
    X_train = np.delete(X_train , 1 , 1)
    X_train = np.concatenate((X_train,X_train**2), axis=1) #square
    #print(X_train.shape)
    Y_train = pd.read_csv(sys.argv[2], sep=',', header=0)
    Y_train = np.array(Y_train.values)
    X_test = pd.read_csv(sys.argv[3], sep=',', header=0)
    X_test = np.array(X_test.values)
    X_test = np.delete(X_test , 1 , 1)
    X_test = np.concatenate((X_test,X_test**2), axis=1)

    return (X_train, Y_train, X_test)

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
    
    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    
    return X_train, Y_train, X_valid, Y_valid

def sigmoid(z):    
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2):
    #print(np.array(shared_sigma).shape)
    sigma_inverse = np.linalg.pinv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_valid.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
    return

def infer(X_test, save_dir, output_dir):
    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    mu1 = np.loadtxt(os.path.join(save_dir, 'mu1'))
    mu2 = np.loadtxt(os.path.join(save_dir, 'mu2'))
    shared_sigma = np.loadtxt(os.path.join(save_dir, 'shared_sigma'))
    N1 = np.loadtxt(os.path.join(save_dir, 'N1'))
    N2 = np.loadtxt(os.path.join(save_dir, 'N2'))

    # Predict
    sigma_inverse = np.linalg.pinv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)

    print('=====Write output to %s =====' % output_dir)
    # Write output
    #if not os.path.exists(output_dir):
#os.mkdir(output_dir)
    #output_path = output_dir
    with open(output_dir, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return

# Load feature and label
X_all, Y_all, X_test = load_data()
# Normalization
X_all, X_test = normalize(X_all, X_test)


infer(X_test, './gen_parameters/', sys.argv[4])