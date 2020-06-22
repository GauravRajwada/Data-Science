# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:55:09 2020

@author: Gaurav
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

train=pd.read_csv("E:/Kaggel compitiion/Digit Recognizer/train.csv")
test=pd.read_csv('E:/Kaggel compitiion/Digit Recognizer/test.csv')
sub=pd.read_csv("E:/Kaggel compitiion/Digit Recognizer/sample_submission.csv")

x=train.drop("label",axis=1)
y=train['label']


x.shape
test.shape
sub.shape

# Normalizing
x/=255.0
test/=255.0

# Reshape in to height =28, weidth=28
x=x.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)

k=len(set(y))

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
i=4
plt.imshow(x_train[i][:,:,0])

model=Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3),activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(3,3)))
model.add(Dropout(0.20))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(k,activation='softmax'))

model.compile(optimizer="RMSprop",loss="categorical_crossentropy",metrics=['accuracy'])


datagen=ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0.1,
    width_shift_range=0.01,
    height_shift_range=0.01,
    brightness_range=None,
    shear_range=0.01,
    zoom_range=0.01,
    channel_shift_range=0.01,
    fill_mode="nearest",
    cval=0.01,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.01,
    dtype=None,
)

datagen.fit(x_train)

history=model.fit_generator(datagen.flow(x_train,y_train,batch_size=20),epochs=5,validation_data=(x_test,y_test))





from skimage import transform
from PIL import Image


def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (28, 28, 1))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

image = load("E:/Python Project/Paint Save/im/3.png")

# plt.imshow(image[0][:,:,0])

result=model.predict(image)
result=np.argmax(result,axis = 1)
print(result)

test_pred=model.predict(test)
test_pred=np.argmax(test_pred,axis = 1) 

ID=sub['ImageId']
Sur=test_pred
submission = pd.DataFrame({'ImageId':ID,'Label':Sur})
submission.to_csv(r'E:/Kaggel compitiion/Digit Recognizer/sub.csv',index=False)


"""Saving model"""
from sklearn.externals import joblib
joblib.dump(model, "Digit_reconizer")

