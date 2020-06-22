# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:54:22 2020

@author: Gaurav
"""
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
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

df.isnull().sum()
df1.isnull().sum()



df['Cabin']=df['Cabin'].fillna('Missing')
df1['Cabin']=df1['Cabin'].fillna('Missing')
df = df. fillna(df['Embarked']. value_counts(). index[0])
df1['Age']=df1['Age'].fillna(df1['Age'].median())
df1['Fare']=df1['Fare'].fillna(df1['Fare'].median())
df1['Fare_Per_Person']=df1['Fare_Per_Person'].fillna(df1['Fare_Per_Person'].median())

sb.countplot(x = 'Survived', hue = 'Sex', data =df)
sb.countplot(x = 'Survived', hue = 'Pclass', data =df)
sb.countplot(x = 'Survived', hue = 'Embarked', data =df)

df['Sex']=df['Sex'].replace(['male','female'],[0,1])
df1['Sex']=df1['Sex'].replace(['male','female'],[0,1])

df['Embarked']=df['Embarked'].replace(['S','C','Q'],[0,1,2])
df1['Embarked']=df1['Embarked'].replace(['S','C','Q'],[0,1,2])

df=df.drop(['PassengerId',"Name","SibSp","Parch","Ticket","Cabin","Fare"],axis=1)
df1=df1.drop(['PassengerId',"Name","SibSp","Parch","Ticket","Cabin","Fare"],axis=1)

plt.scatter(df["Fare_Per_Person"], df['Survived'])
plt.scatter(df["Family_Size"], df['Survived'])

sb.set(style="ticks", color_codes=True)
sb.pairplot(data=df,hue='Survived',diag_kind='hist')

x=df.iloc[:,1:].values
y=df.iloc[:,0].values
test=df1.iloc[:,:].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scale_x=StandardScaler()
x_train=scale_x.fit_transform(x_train)
x_test=scale_x.transform(x_test)

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

knn=KNeighborsClassifier(n_jobs=-1)

dt=DecisionTreeClassifier()

rf=RandomForestClassifier(n_jobs=-1)

xgb=XGBClassifier(n_jobs=-1)

svm=SVC()

sgd = SGDClassifier(n_jobs=-1)

gbc = GradientBoostingClassifier()

pipe=[knn,dt,rf,svm,sgd,gbc]

model_dict={0:'KNN',1:'Decision Tree',2:'Random Forest',3:'XGBOOST',4:'SVM',4:'sgd',5:'gbc'}

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

score=cross_val_score(xgb, x_train,y_train,cv=5,n_jobs=-1)
print(score)
print(score.mean())

# from sklearn.model_selection import GridSearchCV
# parameter={"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
#  "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
#  "min_child_weight" : [ 1, 3, 5, 7 ],
#  "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
#  "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

# grid=GridSearchCV(XGBClassifier(n_jobs=-1), param_grid=parameter)
# best_model=grid.fit(x_train, y_train)

classifier=XGBClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

test_pred=classifier.predict(test)

ID=Data_test['PassengerId']
Sur=test_pred
submission = pd.DataFrame({'PassengerId':ID,'Survived':Sur})
submission.to_csv(r'E:/Kaggel compitiion/Titanic Predictions 2.csv.csv',index=False)


