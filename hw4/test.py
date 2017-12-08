# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
from keras.models import Sequential , load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers.advanced_activations import *
from gensim.models import Word2Vec
vector_len = 128
sentence_len = 64
insert = np.zeros(vector_len)

embedding = Word2Vec.load('./model128.bin')
X_test = []

f = open(sys.argv[1] , 'r' , encoding="utf-8")
data = []
for line in open(sys.argv[1] , 'r' , encoding="utf-8"):
    line = f.readline()
    data.append(line)
f.close()
del data[0]


for i in range(len(data)):
    temp = data[i].split(',' , 1)
    data[i] = temp[1].strip('\n')

for i in range(len(data)):
    data[i] = data[i].split(' ')
    
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = embedding[data[i][j]]

for i in range(len(data)):
    for j in range(sentence_len - len(data[i])):
        data[i].append(insert)

X_test = np.array(data)
print(X_test.shape)
model = load_model('model')
prediction = model.predict(X_test)

ans = []
for i in range(prediction.shape[0]):
    if prediction[i] > 0.5:
        ans.append('1')
    else:
        ans.append('0')
ans = np.array(ans)
f = open(sys.argv[2], 'w')
f.write('id,label\n')
for i in range(ans.shape[0]):
    f.write('{0},{1}\n'.format(i , ans[i]))

print(prediction)

f.close()
