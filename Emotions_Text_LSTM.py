# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:23:49 2020

@author: Gaurav
"""
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
import numpy as np
from sklearn.model_selection import train_test_split



df=pd.read_csv("E:/All Data Set/NLP Emotions_Text/NLP Emotions_Text.csv")
df["emotion"].unique()

df["emotion"][df["emotion"] == "anger"] = 0
df["emotion"][df["emotion"] == "sadness"] = 1
df["emotion"][df["emotion"] == "fear"] = 2
df["emotion"][df["emotion"] == "surprise"] = 3
df["emotion"][df["emotion"] == "joy"] = 4
df["emotion"][df["emotion"] == "love"] = 5

df['emotion']=pd.to_numeric(df['emotion'])
df["emotion"].dtype

ps=PorterStemmer()
lm=WordNetLemmatizer()

def corp(df):
    corpus=[]
    for i in range(0,len(df['text'])):
      review=re.sub('[^a-zA-Z]'," ",df['text'][i])
      review=review.lower()
      review=review.split()
      review=[lm.lemmatize(i) for i in review if not i in set(stopwords.words('english'))]
      review=" ".join(review)
      corpus.append(review)
    return corpus
      
corpus=corp(df)

#Vocabulary Size
voc_size=5000

"""Onehot Representation"""
onehot_rep=[one_hot(words,voc_size) for words in corpus]

#Making one hot fixed size
sent_length=20
embedded_docs=pad_sequences(onehot_rep,padding='pre',maxlen=sent_length )


x=np.array(embedded_docs)
y=np.array(df['emotion'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)




#Creating model
embedding_vector_features=30
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(6,activation='softmax'))
# model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.compile(optimizer="RMSprop",loss='sparse_categorical_crossentropy',metrics=['accuracy'])


model.summary()

#Training
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=30,batch_size=10)


