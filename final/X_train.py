import pickle
import numpy as np
data = pickle.load(open('./data/train.data' , 'rb'))
insert = np.zeros(39)
X_train = []

for i in range(len(data)):
    X_train.append([])
    for j in range(len(data[i])):
        X_train[i].append(data[i][j])
    
    for k in range(250-len(data[i])):
        X_train[i].append(insert)

X_train = np.array(X_train)
print(X_train)
print(X_train.shape)
np.save('X_train.npy' , X_train)