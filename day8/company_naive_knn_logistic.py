# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:46:04 2024

@author: dbda
"""

import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB,GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

company = pd.read_csv(r"F:\PML\Cases\Company_Bankruptcy\data.csv")
company.columns

X = company.drop('Bankrupt?', axis=1)
y = company['Bankrupt?']

####### Naive Bayes #########

params = {'var_smoothing' : np.linspace(1e-9, 10, 20)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
nb = GaussianNB()
gcv = GridSearchCV(nb,param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
#{'var_smoothing': 0.5263157904210526}
#-0.13904767219658373]

######## KNN ###########

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(nb,param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
#{'var_smoothing': 0.5263157904210526}
#-0.13904767219658373

#Using both scaler
std_scl = StandardScaler()
knn = KNeighborsRegressor()
min_max_scl = MinMaxScaler()
pipe = Pipeline([('SCL', min_max_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors' : np.arange(1,30),
          'SCL' : [min_max_scl,std_scl ]}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='r2')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
#{'KNN__n_neighbors': 4, 'SCL': MinMaxScaler()}
#0.9820927671330439

########## Logistic Regression ##############
params = {'penalty' : [None, 'l1', 'l2', 'elasticnet'],
          'C': np.linspace(0,10,10),
          'l1_ratio' : np.linspace(0,1,5)}

lr = LogisticRegression(random_state=23, solver='saga')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(lr, param_grid=params, cv=kfold, scoring='neg_log_loss',verbose=3)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
#{'C': 1.1111111111111112, 'l1_ratio': 0.0, 'penalty': 'l1'}
#-0.17995413083592615

