# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:23:56 2020

@author: Gaurav
"""


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

df=pd.read_csv("E:/All Data Set/SMS Spam/SMSSpamCollection",delimiter='\t',names=['label','msg'])
df["label"].value_counts()

ps=PorterStemmer()
lm=WordNetLemmatizer()

corpus=[]
for i in range(len(df['label'])):
    review=re.sub("[^a-zA-Z]"," ",df["msg"][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(i) for i in review if i not in set(stopwords.words('english'))]
    review=" ".join(review)
    corpus.append(review)
    
cv=TfidfVectorizer(max_features=2500)
x=cv.fit_transform(corpus).toarray()
y=pd.get_dummies(df['label'])
y=y.iloc[:,1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

classifier=SVC()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
cm
print("Accuracy:",((cm[0][0]+cm[1][1])/1393)*100,"%")

"""
    Without balancing the data I have got the accuracy of  98.49246231155779 %
"""