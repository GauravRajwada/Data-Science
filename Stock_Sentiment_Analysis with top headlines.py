# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:14:04 2020

@author: Gaurav
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("E:/All Data Set/Stock-Sentiment-Analysis-master/Data.csv",encoding="ISO-8859-1")

# train = df[df['Date'] < '20150101']
# test = df[df['Date'] > '20141231']

data=df.iloc[:,2:]

# data.replace("[^a-zA-Z]"," ", regex=True,inplace=True)

# for i in data.columns:
#     data[i]=data[i].str.lower()
    
headlines = []
for row in range(len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))

ps=PorterStemmer()
corpus=[]
for i in range(len(headlines)):
      review=re.sub('[^a-zA-Z]'," ",headlines[i])
      review=review.lower()
      review=review.split()
      review=[ps.stem(i) for i in review if not i in set(stopwords.words('english'))]
      review=" ".join(review)
      corpus.append(review) 

cv=CountVectorizer(max_features=5000,ngram_range=(2,2))
x=cv.fit_transform(corpus).toarray()

x_train,x_test,y_train,y_test=train_test_split(x,df['Label'],test_size=0.25,random_state=0)


classifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))

