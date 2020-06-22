# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:21:31 2020

@author: Gaurav
"""

import pandas as pd
import string
import numpy as np

Data_train=pd.read_csv("E:/Kaggel compitiion/titanic/train.csv")
Data_test=pd.read_csv("E:/Kaggel compitiion/titanic/test.csv")
df=Data_train
df1=Data_test

# def substrings_in_string(big_string, substrings):
#     for substring in substrings:
#         if str.find(big_string, substring) != -1:
#             return substring
#     print(big_string)
#     return np.nan

# title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
#                     'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
#                     'Don', 'Jonkheer']

# df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))

# df1['Title']=df1['Name'].map(lambda x: substrings_in_string(x, title_list))

# def replace_titles(x):
#     title=x['Title']
#     if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
#         return 'Mr'
#     elif title in ['Countess', 'Mme']:
#         return 'Mrs'
#     elif title in ['Mlle', 'Ms']:
#         return 'Miss'
#     elif title =='Dr':
#         if x['Sex']=='Male':
#             return 'Mr'
#         else:
#             return 'Mrs'
#     else:
#         return title
# df['Title']=df.apply(replace_titles, axis=1)
# df['Title']=df1.apply(replace_titles, axis=1)

# #Turning cabin number into Deck
# cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
# df['Deck']=df['Cabin'].map(lambda x: substrings_in_string(str(x), cabin_list))
# df1['Deck']=df1['Cabin'].map(lambda x: substrings_in_string(str(x), cabin_list))

#Creating new family_size column
df['Family_Size']=df['SibSp']+df['Parch']
df1['Family_Size']=df1['SibSp']+df1['Parch']

df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
df1['Fare_Per_Person']=df1['Fare']/(df1['Family_Size']+1)

df.drop(["PassengerId","Name", "SibSp", "Parch","Fare","Sex","Ticket","Cabin","Embarked"], inplace = True,axis = 1) 
df1.drop(["PassengerId","Name", "SibSp", "Parch","Fare","Sex","Ticket","Cabin","Embarked"], inplace = True,axis = 1) 

# #Dummy variable
# df=pd.get_dummies(df,columns=['Sex'],drop_first=True)
# df1=pd.get_dummies(df1,columns=["Sex"],drop_first=True)

k=df.var()
cor=df.corr()


x=df.iloc[:,1:].values
y=df.iloc[:,0].values
test=df1.iloc[:,:].values

from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
imputer=imputer.fit(x[:,:])
x[:,:]=imputer.transform(x[:,:])
test[:,:]=imputer.transform(test[:,:])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# # Necessary imports 
# from scipy.stats import randint 
# from sklearn.tree import DecisionTreeClassifier 
# from sklearn.model_selection import RandomizedSearchCV 

# # Creating the hyperparameter grid 
# param_dist = {"max_depth": [3, None], 
#  			"max_features": randint(1, 4), 
#  			"min_samples_leaf": randint(1, 9), 
#  			"criterion": ["gini", "entropy"]} 

# # Instantiating Decision Tree classifier 
# tree = DecisionTreeClassifier() 

# # Instantiating RandomizedSearchCV object 
# tree_cv = RandomizedSearchCV(tree, param_dist, cv = 5) 

# tree_cv.fit(x_train, y_train) 

# # Print the tuned parameters and score 
# print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_)) 
# print("Best score is {}".format(tree_cv.best_score_)) 




# from sklearn.preprocessing import StandardScaler
# scale_x=StandardScaler()
# x_train=scale_x.fit_transform(x_train)
# x_test=scale_x.transform(x_test)
# test=scale_x.transform(test)


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy",max_depth=3,max_features= 3,min_samples_leaf= 6)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
test_pred=classifier.predict(test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

(102+48)/179


# test=df1.iloc[:,:].values


# from sklearn.impute import SimpleImputer 
# imputer=SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer=imputer.fit(test[:,:])
# test[:,:]=imputer.transform(test[:,:])

# from sklearn.preprocessing import StandardScaler
# scale_x=StandardScaler()
# test=scale_x.fit_transform(test)

# test_pred=classifier.predict(test)


ID=Data_test['PassengerId']
Sur=test_pred
submission = pd.DataFrame({'PassengerId':ID,'Survived':Sur})
submission.to_csv(r'E:/Kaggel compitiion/Titanic Predictions 2.csv.csv',index=False)









