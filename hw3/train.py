import sys
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.layers.advanced_activations import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

def load_data(filename):
    all = pd.read_csv(filename , sep = ',')
    """
    test = pd.read_csv('test.csv' , sep = ',')
    test_x = test['feature']
    X_test = []
    for i in range(len(test_x)):
        X_test.append(test_x[i].split(' '))
    X_test = np.array(X_test)
    X_test = X_test.astype('float32')
    X_test = X_test.reshape(X_test.shape[0] , 48 , 48 , 1)/255"""

    X = all['feature']
    X_train = []
    for i in range(len(X)):
        X_train.append(X[i].split(' '))
    X_train = np.array(X_train)
    X_train = X_train.astype('float32')
    X_train = X_train.reshape(X_train.shape[0] , 48 , 48 , 1)/255

    Y = np.array(all['label'])
    Y = Y.astype('int')
    S = (Y.shape[0] , 7)
    Y_train = np.zeros(S)
    for y in range(Y.shape[0]):
        Y_train[y][Y[y]] = 1
    Y_train = Y_train.astype('float32')

    #return X_train , Y_train , X_test
    return X_train , Y_train


def main():
    leaky = 0.05
    X, Y = load_data(sys.argv[1])

    epochs = 80

    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    np.save('attr.npy', [mean, std])

    X = (X - mean) / (std + 1e-20)

    X_train, X_valid = X[:-5000], X[-5000:]
    Y_train, Y_valid = Y[:-5000], Y[-5000:]

    datagen = ImageDataGenerator(
            rotation_range=60,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.8, 1.2],
            shear_range=0.2,
            horizontal_flip=True)

    model = Sequential()

    model.add(Conv2D(125, kernel_size=(5, 5), input_shape=(48 , 48 , 1), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=leaky))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(250, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=leaky))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Conv2D(500, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=leaky))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(500, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=leaky))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(500, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit_generator(
            datagen.flow(X_train, Y_train, batch_size=128), 
            steps_per_epoch=5*len(X_train)//128,
            epochs=epochs,
            validation_data=(X_valid, Y_valid)
            )

    model.save(sys.argv[2])    

if __name__ == '__main__':
    main()