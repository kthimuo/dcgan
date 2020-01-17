import warnings
warnings.filterwarnings('ignore')
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Conv2D,Flatten,Activation,MaxPool2D,Dropout,Dense
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import np_utils

(x_train,y_train),(x_test,y_test) = mnist.load_data()
y_train = np_utils.to_categorical(y_train) 
y_test = np_utils.to_categorical(y_test) 
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(y_train)

model = Sequential()
model.add(Conv2D(32,3,input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(Conv2D(32,3))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(1.0))

model.add(Dense(10, activation='softmax'))

adam = Adam(lr=1e-4)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train)

