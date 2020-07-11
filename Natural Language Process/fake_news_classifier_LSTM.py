# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 09:08:01 2020

@author: Gaurav
"""
import pandas as pd


train=pd.read_csv("E:/All Data Set/Achived/NLP Fake_news/train.csv")
test=pd.read_csv("E:/All Data Set/Achived/NLP Fake_news/test.csv")
corpus1=pd.read_csv("E:/All Data Set/Achived/NLP Fake_news/train_corpus.csv")
corpus2=pd.read_csv("E:/All Data Set/Achived/NLP Fake_news/test__corpus.csv")

train=train.dropna()

train_corpus=[]
for i in range(len(train)):
    train_corpus.append(str(corpus1['corpus'][i]))
test_corpus=[]
for i in range(len(test)):
    test_corpus.append(str(corpus2['corpus'][i]))
    
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot

#Vocabulary Size
voc_size=10000

"""Onehot Representation"""
onehot_train=[one_hot(words,voc_size) for words in train_corpus]
onehot_test=[one_hot(words,voc_size) for words in test_corpus]

#Making one hot fixed size
sent_length=700
embedded_docs_train=pad_sequences(onehot_train,padding='pre',maxlen=sent_length )
embedded_docs_test=pad_sequences(onehot_test,padding='pre',maxlen=sent_length )


import numpy as np
from sklearn.model_selection import train_test_split

x=np.array(embedded_docs_train)
y=np.array(train['label'])
test_x=np.array(embedded_docs_test)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)



#Creating model
embedding_vector_features=200
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

#Training
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)

"""
loss: 0.0120 - accuracy: 0.9964 - val_loss: 0.4118 - val_accuracy: 0.9191
loss: 0.0229 - accuracy: 0.9930 - val_loss: 0.3471 - val_accuracy: 0.9167
"""


"""Adding dropout layers"""
embedding_vector_features=200
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.4))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,batch_size=90)


"""loss: 0.1042 - accuracy: 0.9693 - val_loss: 0.2782 - val_accuracy: 0.9114"""
