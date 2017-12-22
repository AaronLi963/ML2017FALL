import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.layers import LSTM
from numpy import argmax
import re
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Flatten, Reshape, Concatenate,Merge
from keras.layers import Dot , Add
from keras.utils import np_utils # this will make the label to one-hot encoding 
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.layers import Dropout, Input
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model
import tensorflow as tf
import csv
import sys


if __name__ == "__main__":
    model = load_model('latent128')

    test = pd.read_csv(sys.argv[1] , encoding = 'big5')
    test = test.iloc[:,1:3].values

    mean_std = np.load('mean_std.npy')
    mean = mean_std[0]
    std = mean_std[1]
    
    sol = model.predict([test[:,0],test[:,1]])
    sol = (sol*std)+mean

    prediction=[]
    prediction.append(list(sol))

    prediction_write = []
    temp=[]
    temp.append('TestDataID')
    temp.append('Rating')
    prediction_write.extend([temp])

    for i in range(sol.shape[0]): #test data size
        temp=[('%s' %(i+1),sol[i,0])] 
        prediction_write.extend(temp)#temp.append(sol[i])


    f = open(sys.argv[2],"w" , newline='')
    w = csv.writer(f)
    for row in prediction_write:
        w.writerow(row)

    f.close()
