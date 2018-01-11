import numpy as np
import sys
from skimage import io
import os
from skimage import transform

direct = sys.argv[1]
pics = os.listdir(direct)
X = []

for name in pics:
    im = io.imread(os.path.join(direct,name))
    #im = transform.resize(im , (400 , 400 , 3))
    X.append(im.flatten())

X = np.array(X)
X = X/255

pic = io.imread( os.path.join(direct, sys.argv[2]))
#pic = transform.resize(pic, (400,400,3))
pic = pic.flatten()
k = 50
X_mean = np.mean(X , axis = 0)
#pic_mean = pic - X_mean

U, s, V = np.linalg.svd(np.transpose(X - X_mean), full_matrices=False)
temp = []
for i in range(4):
    temp.append(np.dot(pic - X_mean , U[: , i]))

res = np.zeros(600*600*3)
#res = np.zeros(400*400*3)


for i in range(4):
    res += temp[i] * U[: , i]
res += X_mean
res -= np.min(res)
res /= np.max(res)
res = (res * 255).astype(np.uint8)
res = res.reshape((600 , 600 , 3))
#res = res.reshape((400,400,3))

io.imsave('reconstruction.jpg',res)