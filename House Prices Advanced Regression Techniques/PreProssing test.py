# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:06:53 2020

@author: Gaurav
"""
"""Advance House price prediction"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1=pd.read_csv("E:/Kaggel compitiion/House Prices Advanced Regression Techniques/train.csv")
df=pd.read_csv("E:/Kaggel compitiion/House Prices Advanced Regression Techniques/test.csv")

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(df,df['SalePrice'],test_size=0.1,random_state=0)

""" Missing Values"""

# Missing values in categorical features
missing_features=[i for i in df.columns if df[i].isnull().sum()>1 and df[i].dtype=='O']
for i in missing_features:
    print(i,np.round(df[i].isnull().mean(),4))
    
# Replacing Missing value with new label
def fillNa(df,missing_features):
    data=df.copy()
    data[missing_features]=data[missing_features].fillna("Missing")
    return data
df=fillNa(df,missing_features)
print(df[missing_features].isnull().sum())


#Missing values in numerical variable
missing_features1=[i for i in df.columns if df[i].isnull().sum()>1 and df[i].dtype!='O']
for i in missing_features1:
    print(i,np.round(df[i].isnull().mean(),4))
    
# Now we will replace missing value in numerical by median because some values are very large
# And creating new feature if missing then 1 else 0

for i in missing_features1:
    med=df[i].median()
    #Creatiing new feature 
    df[i+"Nan"]=np.where(df[i].isnull(),1,0)
    #Filling missing value with median
    df[i].fillna(med,inplace=True)


df[missing_features1].isnull().sum()

"""Converting year to how many year"""
yr_features=["GarageYrBlt","YearRemodAdd","YearBuilt"]

for i in yr_features:
        df[i]=df['YrSold']-df[i]

"""Appling gaussian distribution"""
# Appling to them which features do not have 0
# numerical_features=[i for i in df.columns if df[i].dtype !="O" if 0 not in df[i].unique()]

numerical_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']
for i in numerical_features:
    df[i]=np.log(df[i])
    
"""Handeling Rare categorical Features"""
"""In this we skip that categorical featres which has less than 1%"""    

categorical_features=[i for i in df.columns if df[i].dtype =="O" ]
for i in categorical_features:
    temp=df.groupby(i)["Id"].count()/len(df)
    temp_df=temp[temp>0.01].index
    df[i]=np.where(df[i].isin(temp_df),df[i],"Rare_var")


"""Changing Categorical features"""
for feature in categorical_features:
    labels_ordered=df.groupby([feature])['Id'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df[feature]=df[feature].map(labels_ordered)
    

"""Appling scaling either min max scalong or stanfard scaling"""
scale_features=[i for i in df.columns if i not in ['Id']]

from sklearn.preprocessing import MinMaxScaler,StandardScaler
scale=MinMaxScaler()
scale.fit(df[scale_features])
# Doing this concat beacuse the scale.transorfm gives array we have to change into dataframe
data = pd.concat([df[['Id']].reset_index(drop=True),
                    pd.DataFrame(scale.transform(df[scale_features]), columns=scale_features)],
                    axis=1)
data.to_csv(r'E:/Kaggel compitiion/house-prices-advanced-regression-techniques/After analizing/test.csv',index=False)
