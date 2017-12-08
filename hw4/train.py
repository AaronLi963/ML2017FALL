# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import *
from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint

vector_len = 128
sentence_len = 64
"""insert = []
for i in range(vector_len):
    insert.append(0.)"""
insert = np.zeros(vector_len)
embedding = Word2Vec.load('./model128.bin')
X_train = []

f = pd.read_csv(sys.argv[1] , sep = '\+\+\+\$\+\+\+' , header = None)
f = np.array(f)
data = []
label = []

for i in range(f.shape[0]):
    label.append(f[i][0])
    data.append(f[i][1])
label = np.array(label)
"""
Y_train = []
for i in range(label.shape[0]):
    if label[i] == 0:
        Y_train.append(np.array([1 , 0]))
    else:
        Y_train.append(np.array([0 , 1]))
Y_train = np.array(Y_train)"""
"""   
for i in range(len(data)):
    temp = data[i].split(',' , 1)
    data[i] = temp[1].strip('\n')
data = np.array(data)
"""

for i in range(len(data)):
    data[i] = data[i].split(' ')
    
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = embedding[data[i][j]]

for i in range(len(data)):
    for j in range(sentence_len-len(data[i])):
        data[i].append(insert)

X_train = np.array(data)
print(X_train.shape)
#X_train = sequence.pad_sequences(X_train , maxlen=100 , dtype = 'float' , padding = 'pre' , value=0.)


model = Sequential()
model.add(LSTM(1024 , input_shape = (sentence_len , vector_len) , recurrent_dropout=0.5))
model.add(Dropout(0.5))
"""
model.add(Dense(512 ,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(512 ,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(512 ,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(512 ,activation='sigmoid'))
model.add(Dropout(0.5))"""

model.add(Dense(1 ,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

filepath="new_best.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
print(model.summary())
model.fit(X_train, label ,validation_split=0.4 , batch_size=64, epochs=10 ,callbacks=callbacks_list)
model.save('model')