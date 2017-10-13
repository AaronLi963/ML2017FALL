import csv
import numpy as np
from numpy.linalg import inv
import sys
import os
import math

w = np.load('model_best_square10000000.npy')
f = open(sys.argv[1] , 'r')
row = csv.reader(f ,delimiter = ',')
counter = 0
x = []
for r in row:
    if counter % 18 == 0:
        x.append([])
        for i in range(2 , 11):
            x[counter//18].append(float(r[i]))

    else:
        for i in range(2 ,11):
            if r[i] == 'NR':
                x[counter//18].append(0)
            else:
                x[counter//18].append(float(r[i]))
        
    counter = counter + 1
f.close()
x = np.array(x)

x = np.concatenate((x,x**2), axis=1)
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

ans = []
for i in range(len(x)):
    ans.append(['id_'+str(i)])
    y = np.dot(w , x[i])
    ans[i].append(y)

filename = sys.argv[2]
f = open(filename, "w+")
s = csv.writer(f,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
f.close()