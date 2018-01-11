import numpy as np
from keras.datasets import mnist
from keras.models import Model , load_model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import sys

pic = np.load(sys.argv[1])
#pic = (pic - np.mean(pic)) / np.std(pic)
pic = pic/255

encoder = load_model('encoder97.85')

# plotting
encoded_imgs = encoder.predict(pic)

kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)

import pandas as pd
Test = pd.read_csv(sys.argv[2], encoding = "big5")
test = Test.iloc[:,0:3].values

look_up_table = kmeans.labels_
look_up_table.sum()

ans = []

for i in range(test.shape[0]):
    if look_up_table[test[i,1]] == look_up_table[test[i,2]]:
        ans.append(1)
    else:
        ans.append(0)

        import csv
prediction=[]
prediction.append(list(ans))

prediction_write = []
temp=[]
temp.append('ID')
temp.append('Ans')
prediction_write.extend([temp])

for i in range(len(ans)): #test data size
  temp=[('%s' %(i),ans[i])] 
  prediction_write.extend(temp)#temp.append(sol[i])


f = open(sys.argv[3],"w" , newline='')
w = csv.writer(f)
for row in prediction_write:
    w.writerow(row)

f.close()
