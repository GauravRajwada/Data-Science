# -*- coding: utf-8 -*-
"""
Created on Fri May  8 07:57:56 2020

@author: Gaurav
"""
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import string
import numpy as np

Data_train=pd.read_csv("E:/Kaggel compitiion/titanic/train.csv")
Data_test=pd.read_csv("E:/Kaggel compitiion/titanic/test.csv")
df=Data_train
df1=Data_test

#Creating new family_size column
df['Family_Size']=df['SibSp']+df['Parch']
df1['Family_Size']=df1['SibSp']+df1['Parch']

df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
df1['Fare_Per_Person']=df1['Fare']/(df1['Family_Size']+1)

df.drop(["PassengerId","Name", "SibSp","Sex","Fare", "Parch","Ticket","Cabin","Embarked"], inplace = True,axis = 1) 
df1.drop(["PassengerId","Name", "SibSp","Sex","Fare", "Parch","Ticket","Cabin","Embarked"], inplace = True,axis = 1) 

sb.set(style="ticks", color_codes=True)
sb.pairplot(data=df,hue='Survived')

df1=df1.fillna(df1.mean())
df1.isnull().any()


x=df.iloc[:,1:].values
y=df.iloc[:,0].values
test=df1.iloc[:,:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
# Scaling the data
from sklearn.preprocessing import StandardScaler
scale_x=StandardScaler()
x_train=scale_x.fit_transform(x_train)
x_test=scale_x.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.svm import SVC

knn=Pipeline([('scaler',StandardScaler()),
              ('knn',KNeighborsClassifier(n_jobs=-1))])

dt=Pipeline([('scaler1',StandardScaler()),
              ('dt',DecisionTreeClassifier())])

rf=Pipeline([('scaler2',StandardScaler()),
              ('rf',RandomForestClassifier(n_jobs=-1))])

# xgb=Pipeline([('scale3',StandardScaler()),
#               ("xgb",XGBClassifier(n_jobs=-1))])

svm=Pipeline([('scale4',StandardScaler()),
              ("svm",SVC())])

pipe=[knn,dt,rf,svm]

model_dict={0:'KNN',1:'Decision Tree',2:'Random Forest',3:'SVM',4:'SVM'}

best_accuracy=0
best_classifier=0
best_pipeline=""


for i in pipe:
    i.fit(x_train,y_train)
    

for i,model in enumerate(pipe):
    print("{} accuracy is: {}".format(model_dict[i],model.score(x_test,y_test)))


for i,model in enumerate(pipe):
    if best_accuracy<model.score(x_test,y_test):
        best_accuracy=model.score(x_test,y_test)
        best_classifier=model
        best_pipeline=i
        
print("{} is best model with {}".format(model_dict[best_pipeline],best_accuracy))


from sklearn.model_selection import GridSearchCV
degree=[i for i in range(1,5)]
param_grid = {'C':[0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],  
              'gamma':[1e-2, 1e-3, 1e-4, 1e-5], 
              'degree':degree,
              'kernel': ['rbf','sigmoid','linear','poly']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 

best_model=grid.fit(x_train,y_train)

print(grid.best_params_, grid.best_score_)

print(best_model.estimator)
print(best_model.score(x_train,y_train))
