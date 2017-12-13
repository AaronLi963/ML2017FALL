from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
model = KeyedVectors.load_word2vec_format('./wiki.zh/wiki.zh.vec')


f = open('./data/train.caption' , 'r' , encoding = 'UTF-8')
data = []
for line in f.readlines():
    data.append(line)
insert = np.zeros(300)

Y_train = []
for i in range(len(data)):
    Y_train.append([])
    sentence = data[i].rstrip(' \n').split(' ')
    for j in range(len(sentence)):
        try:
            Y_train[i].append(model[sentence[j]])
        except :
            Y_train[i].append(insert)
    for k in range(30-len(sentence)):
        Y_train[i].append(insert)
        
Y_train = np.array(Y_train)
np.save('Y_train.npy' , Y_train)