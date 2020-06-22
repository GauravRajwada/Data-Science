# -*- coding: utf-8 -*-
"""
Created on Sun May 10 17:13:04 2020

@author: Gaurav
"""
import pandas as pd
import string
import numpy as np
from statistics import mode
import matplotlib.pyplot as ply
import seaborn as sb

Data_train=pd.read_csv("E:/Kaggel compitiion/titanic/train.csv")
Data_test=pd.read_csv("E:/Kaggel compitiion/titanic/test.csv")
df=Data_train.copy()
df1=Data_test.copy()

df.isnull().sum().sort_values(ascending = False)
df1.isnull().sum().sort_values(ascending = False)

df["Embarked"] = df["Embarked"].fillna(mode(df["Embarked"]))
df1["Embarked"] = df1["Embarked"].fillna(mode(df1["Embarked"]))

df["Sex"][df["Sex"] == "male"] = 0
df["Sex"][df["Sex"] == "female"] = 1

df1["Sex"][df1["Sex"] == "male"] = 0
df1["Sex"][df1["Sex"] == "female"] = 1


df["Embarked"][df["Embarked"] == "S"] = 0
df["Embarked"][df["Embarked"] == "C"] = 1
df["Embarked"][df["Embarked"] == "Q"] = 2


df1["Embarked"][df1["Embarked"] == "S"] = 0
df1["Embarked"][df1["Embarked"] == "C"] = 1
df1["Embarked"][df1["Embarked"] == "Q"] = 2


sb.heatmap(df.corr(), annot = True)
# Grouping by Pclass and using a lambda to impute the Age median
df['Age'] = df.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))
# Grouping by Pclass and using a lambda to impute the Age median
df['Age'] = df.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))

df1['Age'] = df1.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))
# Grouping by Pclass and using a lambda to impute the Age median
df1['Age'] = df1.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))


df.isnull().sum().sort_values(ascending = False)
df1.isnull().sum().sort_values(ascending = False)

df['Cabin'] = df['Cabin'].fillna('Missing')
df1['Cabin'] = df1['Cabin'].fillna('Missing')

df1['Fare'] = df1['Fare'].fillna(df1['Fare'].mean())

df.isnull().mean().sort_values(ascending = False)
df1.isnull().mean().sort_values(ascending = False)

df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
df1['Title'] = df1.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

df['Title'].value_counts(normalize = True) * 100
df1['Title'].value_counts(normalize = True) * 100

df['Title'] = df['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 
                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')

df1['Title'] = df1['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 
                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')


title_category = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Other':5}

df['Title']=df['Title'].replace(['Mr', 'Miss', 'Mrs', 'Master', 'Other'],[1,2,3,4,5])
df1['Title']=df1['Title'].replace(['Mr', 'Miss', 'Mrs', 'Master', 'Other'],[1,2,3,4,5])

df['familySize'] = df['SibSp'] + df['Parch'] + 1
df1['familySize'] = df1['SibSp'] + df1['Parch'] + 1

df = df.drop(['Name', 'SibSp', 'Parch', 'Ticket'], axis = 1)
df1 = df1.drop(['Name', 'SibSp', 'Parch', 'Ticket'], axis = 1)


df[['Sex', 'Embarked']] = df[['Sex', 'Embarked']].apply(pd.to_numeric) 
df1[['Sex', 'Embarked']] = df1[['Sex', 'Embarked']].apply(pd.to_numeric) 

import re
df['Cabin'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
df1['Cabin'] = df1['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'Missing':9}

df['Cabin'] = df['Cabin'].map(cabin_category)
df1['Cabin'] = df1['Cabin'].map(cabin_category)

x=df.iloc[:,2:].values
y=df.iloc[:,1].values
test=df1.iloc[:,1:].values

from sklearn.model_selection import train_test_split

# Here is out local validation scheme!
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,random_state = 2)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state = 2)
# Set our parameter grid
param_grid = { 
    'criterion' : ['gini', 'entropy'],
    'n_estimators': [100, 300, 500,1000],
    'max_features': ['auto', 'log2'],
    'max_depth' : [3, 5, 7,9]    
}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5)
grid.fit(x_train, y_train)

grid.best_params_


rf= RandomForestClassifier(random_state=2,criterion='gini',max_depth=9,max_features='log2',n_estimators =300)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)

rf.score(x_test,y_test)

test_pred=rf.predict(test)


Data_test=pd.read_csv("E:/Kaggel compitiion/titanic/test.csv")
ID=Data_test['PassengerId']
Sur=test_pred
submission = pd.DataFrame({'PassengerId':ID,'Survived':Sur})
submission.to_csv(r'E:/Kaggel compitiion/Titanic Predictions 2.csv.csv',index=False)
