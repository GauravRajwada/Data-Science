# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 17:31:47 2020

@author: Gaurav
"""
import pandas as pd
import numpy as np

df=pd.read_csv("E:/Kaggel compitiion/House Prices Advanced Regression Techniques/After analizing/train.csv")
df1=pd.read_csv("E:/Kaggel compitiion/House Prices Advanced Regression Techniques/After analizing/test.csv")



"""Features Selection"""
y=df["SalePrice"]
x=df.drop(["Id","SalePrice"],axis=1)
test=df1.drop(["Id"],axis=1)
test=test.fillna(test.mean())
test=test.drop(["BsmtFullBathNan", "BsmtHalfBathNan"],axis=1)

"""Features Selection"""
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) 
feature_sel_model.fit(x,y)
selected_features=x.columns[(feature_sel_model.get_support())]

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

parameters_for_testing = {
    'colsample_bytree':[0.4,0.6,0.8],
    'gamma':[0,0.03,0.1,0.3],
    'min_child_weight':[1.5,6,10],
    'learning_rate':[0.1,0.07],
    'max_depth':[3,5],
    'n_estimators':[10000],
    'reg_alpha':[1e-5, 1e-2,  0.75],
    'reg_lambda':[1e-5, 1e-2, 0.45],
    'subsample':[0.6,0.95]  
}

                    
xgb_model = XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)

gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, n_jobs=6,iid=False, verbose=10,scoring='neg_mean_squared_error')
gsearch1.fit(x,y)
print (gsearch1.grid_scores_)
print('best params')
print (gsearch1.best_params_)
print('best score')
print (gsearch1.best_score_)






regressor=XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
regressor.fit(x,y)
y_pred=regressor.predict(test)


ID=df1['Id']
Sur=y_pred
submission = pd.DataFrame({'Id':ID,'SalePrice':Sur})
submission.to_csv(r'E:/Kaggel compitiion/house-prices-advanced-regression-techniques/House Price submission.csv',index=False)


