# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:11:21 2020

@author: Gaurav
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

train=pd.read_csv("E:/Kaggel compitiion/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("E:/Kaggel compitiion/house-prices-advanced-regression-techniques/test.csv")

test=test.drop('Id',axis=1)
train=train.drop('Id',axis=1)

sb.distplot(train["SalePrice"])
plt.show()
sb.boxenplot(train["SalePrice"])
plt.show()
"""To make normalize"""
train["SalePrice"]=np.log(train["SalePrice"])
sb.distplot(train["SalePrice"])
plt.show()

cor = train.corr()
plt.figure(figsize=(15,10))
sb.heatmap(cor,cmap="Blues", vmax=0.9)
plt.show()

cor[abs(cor['SalePrice'].values) >= 0.5]['SalePrice'].sort_values(ascending=False)


plt.scatter(train["OverallQual"], train["SalePrice"])
plt.xlabel("OverallQual")
plt.ylabel("SalePrice")

plt.scatter(train["GrLivArea"], train["SalePrice"])
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")

train=train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<12.5)].index)
train.reset_index(drop = True, inplace = True)
"""
plt.scatter(train["GarageCars"], train["SalePrice"])
plt.xlabel("GarageCars")
plt.ylabel("SalePrice")


plt.scatter(train["GarageArea"], train["SalePrice"])
plt.xlabel("GarageArea")
plt.ylabel("SalePrice")


plt.scatter(train["TotalBsmtSF"], train["SalePrice"])
plt.xlabel("TotalBsmtSF")
plt.ylabel("SalePrice")

plt.scatter(train["1stFlrSF"], train["SalePrice"])
plt.xlabel("1stFlrSF")
plt.ylabel("SalePrice")

plt.scatter(train["FullBath"], train["SalePrice"])
plt.xlabel("FullBath")
plt.ylabel("SalePrice")
"""

import matplotlib.style as style
style.use('ggplot')
sb.set_style('whitegrid')
plt.subplots(figsize = (30,20))
## Plotting heatmap. 
# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sb.heatmap(train.corr(), cmap=sb.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0, );
## Give title. 
plt.title("Heatmap of all the Features", fontsize = 30)


"""Missing Value"""

data=pd.concat((train,test)).reset_index(drop=True)

#Total Missing Values
total_missing=data.isnull().sum().sort_values(ascending = False)
percent_missing=((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending = False)
missing=pd.concat([total_missing,percent_missing],axis=1,keys=["Total","%"])

# Diffrencing categorial and int
data_categorical=data.select_dtypes("O")
data_int=data.select_dtypes(["int64","float64"])

#Missing value in categorical
k=data_categorical.isnull().sum().sort_values(ascending=False)
cat_column1=[i for i in k.index if k[i]>4]
for i in cat_column1:    
    data_categorical[i].fillna("None",inplace=True)

cat_column2=[i for i in k.index if k[i]<=4 and k[i]>0]
for i in cat_column2:
    data_categorical[i].fillna(data[i].mode()[0],inplace = True)

#Missing value in int
k1=data_int.isnull().sum().sort_values(ascending=False)
k1=k1.drop("SalePrice")
int_column1=[i for i in k1.index]

for i in int_column1:
    data_int[i].fillna(0,inplace=True)

df=pd.concat([data_categorical,data_int],axis=1)

df['MSSubClass'] = df['MSSubClass'].apply(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
df.drop('SalePrice', axis=1, inplace = True)


#Again Checking any missing value
total_missing=df.isnull().sum().sort_values(ascending = False)
percent_missing=((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False)
missing=pd.concat([total_missing,percent_missing],axis=1,keys=["Total","%"])


numeric_feats = df.dtypes[df.dtypes != "object"].index
skewed_feats = df[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending=False)
skewed_feats
#Box Cox Transformation on Skewed Features
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

high_skew = skewed_feats[skewed_feats > 0.5]
skew_index = high_skew.index

# Normalise skewed features
for i in skew_index:
    df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1))

#Encoding the finalized features
final_features = pd.get_dummies(df).reset_index(drop=True)
print('Features size:', df.shape)
final_features.head()

"""Spliting data to trin and test"""
nrow_train = train.shape[0]
x_train = final_features[:nrow_train]
x_test = final_features[nrow_train:]
y = train['SalePrice']

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# Kernel Ridge Regression : made robust to outliers
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

# LASSO Regression : made robust to outliers
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 
                    alphas=alphas2,random_state=42, cv=kfolds))

# Elastic Net Regression : made robust to outliers
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 
                         alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))

#Fit the training data x_train,y 
elastic_model = elasticnet.fit(x_train, y)
lasso_model = lasso.fit(x_train, y)
ridge_model = ridge.fit(x_train, y)

# model blending function using fitted models to make predictions
def blend_models(X):
    return ((elastic_model.predict(X)) + (lasso_model.predict(X)) + (ridge_model.predict(X)))/3

submission = pd.read_csv("E:/Kaggel compitiion/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = np.expm1(blend_models(x_test))

# Fix outleir predictions
q1 = submission['SalePrice'].quantile(0.0045)
q2 = submission['SalePrice'].quantile(0.99)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("E:/Kaggel compitiion/house-prices-advanced-regression-techniques/House Price submission.csv", index=False)