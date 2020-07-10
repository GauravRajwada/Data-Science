# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:29:19 2020

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

train=pd.read_csv("E:/All Data Set/fake-news/train.csv")
test=pd.read_csv("E:/All Data Set/fake-news/test.csv")
sub=pd.read_csv("E:/All Data Set/fake-news/submit.csv")

train=train.dropna()
train.reset_index(inplace=True)
test=test.fillna(" ")

ps=PorterStemmer()
lm=WordNetLemmatizer()

# corpus=[]
# for i in range(0,len(train['text'])):
#      review=re.sub('[^a-zA-Z]'," ",train['text'][i])
#      review=review.lower()
#      review=review.split()
#      review=[ps.stem(i) for i in review if not i in set(stopwords.words('english'))]
#      review=" ".join(review)
#      corpus.append(review) 



cor=pd.read_csv("E:/All Data Set/fake-news/train_corpus.csv")
corpus_train=[]
for i in range(len(cor)):
    corpus_train.append(str(cor["corpus"][i]))

cv=CountVectorizer(max_features=5000,ngram_range=(1,3))
x=cv.fit_transform(corpus_train).toarray()
y=train.iloc[:,5]

#spliting the datasets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)

#Fitting the model
classifier=MultinomialNB()
classifier.fit(x_train,y_train)

#Testing the accuracy 
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100)+"%")

"""Using naive bayes with Tfidf got accuracy of  0.9021144732045207"""

#Going to use Passive Aggressive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier
linear_classifier=PassiveAggressiveClassifier(n_iter_no_change=50)
linear_classifier.fit(x_train,y_train)

y_pred=linear_classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100)+"%")

"""Using PassiveAggressiveClassifier with Tfidf got accuracy of  93.693036820998915"""


var=[]
result="Long before Facebook, Twitter or even Google existed, the fact checking website Snopes.com was running down the half-truths, misinformation, and outright lies that ricochet across the Internet. Today it remains a widely respected clearinghouse of all things factual and not."
result=re.sub('[^a-zA-Z]'," ",result)
result=result.lower()
result=result.split()
result=[ps.stem(i) for i in result if i not in set(stopwords.words('english'))]
result=" ".join(result)
var.append(result)
x=cv.transform(var).toarray()

print(classifier.predict(x))






