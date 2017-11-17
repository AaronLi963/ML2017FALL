import sys
import pandas as pd
import numpy as np
from keras.models import Sequential , load_model
from keras.layers.core import Dense , Dropout , Activation
from keras.layers import Convolution2D , Conv2D , MaxPooling2D , Flatten
from keras.optimizers import SGD , Adam
from keras.utils import np_utils



def load_data(filename):
    test = pd.read_csv(filename , sep = ',')

    test_x = test['feature']
    X_test = []
    for i in range(len(test_x)):
        X_test.append(test_x[i].split(' '))
    X_test = np.array(X_test)
    X_test = X_test.astype('float32')
    X_test = X_test.reshape(X_test.shape[0] , 48 , 48 , 1)/255

    return X_test
def load_data1():
    X_test = np.load('X_test.npy')
    return X_test
    

X_test = load_data(sys.argv[1])

attr = np.load('attr.npy')
X_test = (X_test - attr[0]) / (attr[1] + 1e-20)

model = load_model('model')

prediction = model.predict(X_test)

ans = []

for i in range(prediction.shape[0]):
    ans.append(np.argmax(prediction[i]))

f = open(sys.argv[2] , 'w')
f.write('id,label\n')
for i in range(len(ans)):
    f.write('{0},{1}\n'.format(i , ans[i]))

#print(ans)
