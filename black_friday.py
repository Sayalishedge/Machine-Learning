# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 15:01:18 2024

@author: dbda
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer 
from xgboost import XGBRegressor,XGBRFRegressor
from sklearn.model_selection import KFold, GridSearchCV

train = pd.read_csv(r"C:\Users\dbda\Desktop\PML\competitions\blackfriday\train.csv")
test = pd.read_csv(r"C:\Users\dbda\Desktop\PML\competitions\blackfriday\test.csv")
ss = pd.read_csv(r"C:\Users\dbda\Desktop\PML\competitions\blackfriday\sample_submission_V9Inaty.csv")

train.columns
test.columns

train.isnull().sum()
test.isnull().sum()

train['Product_Category_2'].value_counts()
train[['Product_ID','Product_Category_2']]

X_train = train.drop(['Purchase','User_ID','Product_ID'], axis=1)
X_test = test.drop(['User_ID','Product_ID'], axis=1)

X_train = X_train.astype(str)
X_test = X_test.astype(str)

#concat the two
whole_data = pd.concat([X_train,X_test])
whole_data['Product_Category_2'].fillna('missing',inplace=True)
whole_data['Product_Category_3'].fillna('missing',inplace=True)

print(whole_data.isnull().sum())
print(whole_data.info())

#Columns transforming with One hot encoding
ohe = OneHotEncoder()

ohe.fit(whole_data)

trans_trn = ohe.transform(X_train)
trans_tst = ohe.transform(X_test)
y_train = train['Purchase']

#%%
from xgboost import XGBRegressor,XGBRFRegressor
from sklearn.model_selection import KFold, GridSearchCV
x_gbm = XGBRegressor(random_state=23)
print(x_gbm.get_params())

params = {'learning_rate' : np.linspace(0.001, 0.1, 3),
          'max_depth' : [3,5],
          'n_estimators' : [25,50]
          }
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
gcv_x_gbm = GridSearchCV(x_gbm, param_grid=params,verbose=3, cv = kfold)
gcv_x_gbm.fit(trans_trn,y_train)
print(gcv_x_gbm.best_score_)
#0.6397555038069276
print(gcv_x_gbm.best_params_)
#{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 50}

best_xgb = gcv_x_gbm.best_estimator_
y_pred = best_xgb.predict(trans_tst)

ss = test[['User_ID','Product_ID']]
ss.loc[:,'Purchase'] = y_pred
ss.to_csv("blackfriday_xgb_Jan9.csv", index=False)









