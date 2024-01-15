# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:48:36 2023

@author: dbda
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

boston = pd.read_csv(r"F:\PML\Datasets\Boston.csv") 
boston.shape

train, test = train_test_split(boston, test_size=0.3)
train.shape, test.shape

X_train = train.drop('medv', axis=1)
y_train = train['medv']
X_test = test.drop('medv', axis=1)
y_test = test['medv']

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))

poly = PolynomialFeatures(degree=2)
X_poly_trn = poly.fit_transform(X_train)
X_poly_trn.shape

poly = PolynomialFeatures(degree=2)
X_poly_tst = poly.transform(X_test)
X_poly_tst.shape


lr.fit(X_poly,y_train)
ycap = le.predict()