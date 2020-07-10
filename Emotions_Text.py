# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:55:30 2020

@author: Gaurav
"""
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import confusion_matrix

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

cv=CountVectorizer(max_features=3000,ngram_range=(0,2))
x=cv.fit_transform(corpus).toarray()
y=df['emotion']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#Fitting the model

# from sklearn.linear_model import PassiveAggressiveClassifier
# classifier=PassiveAggressiveClassifier(n_iter_no_change=50)

classifier=MultinomialNB()
classifier.fit(x_train,y_train)

#Testing the accuracy 
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
cm
true=(cm[2][2]+cm[1][1]+cm[3][3]+cm[4][4]+cm[5][5]+cm[0][0])
total=0
for i in range(6):
    for j in range(6):
        total+=cm[i][j]
        
print(str((true/total)*100)+"%")



import pickle
file = open("sentiment_model1.pkl", 'wb')
pickle.dump(classifier,file)

file = open("sentiment_vectorizer.pkl", 'wb')
pickle.dump(cv,file)



"""
    Best I have used PassiveAggressiveClassifier i got 88.375% accuracy

    1. I have used Lemmitizer by CountVetrize
    
        Maximum Features=2200 gives acc=86.475%
        Maximum Features=2400 gives acc=86.9%
        
        Best- 
        Maximum Features=3000 gives acc=87.325%
    
    
    2. I have used porter stemer by CountVetrize
    
        Maximum Features=1500 gives acc=84.325%
        Maximum Features=1700 and 1800 gives acc=84.625%
        Maximum Features=1900 gives acc=84.7%
        Maximum Features=2000 gives acc=85%
        Maximum Features=2200 gives acc=85.075%
        
        Best gives-
        Maximum Features=2200 gives acc=85.95%
        
        Maximum Features=2500 gives acc=83.875%
        Maximum Features=5000 gives acc=83.84%
    
    3. I have used TfidfVectorizer but accuracy didnt increase

"""


var=[]
result="You are so romantic"
result=re.sub('[^a-zA-Z]'," ",result)
result=result.lower()
result=result.split()
result=[lm.lemmatize(i) for i in result if i not in set(stopwords.words('english'))]
result=" ".join(result)
var.append(result)
x=cv.transform(var).toarray()
z=classifier.predict_proba(x)
classifier.predict_log_proba(x)

print(classifier.predict(x))
