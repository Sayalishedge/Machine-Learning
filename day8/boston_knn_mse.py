# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:21:30 2024

@author: dbda
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import train_test_split,KFold, GridSearchCV
from sklearn.metrics import log_loss, accuracy_score, r2_score, mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import Pipeline

boston = pd.read_csv(r"F:\PML\Datasets\Boston.csv")
boston.info()
boston.head(1)

X = boston.drop('medv', axis=1).values
y = boston['medv'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=23)

knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(mse(y_test, y_pred))
#52.43243421052631
print(r2_score(y_test, y_pred))
#0.3689542355727533

######################### Grid Search CV #########################

params = {'n_neighbors': np.arange(1,20)}
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsRegressor()
gcv = GridSearchCV(knn, param_grid=params, cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

#{'n_neighbors': 3}
#0.544002500199357


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

#{'KNN__n_neighbors': 2, 'SCL': StandardScaler()}
#0.7793947733827284


################## Randomized Search ####################
from sklearn.model_selection import RandomizedSearchCV

std_scl = StandardScaler()
knn = KNeighborsRegressor()
min_max_scl = MinMaxScaler()
pipe = Pipeline([('SCL', min_max_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors' : np.arange(1,10),
          'SCL' : [min_max_scl,std_scl, 'passthrough' ]}
rgcv = RandomizedSearchCV(pipe, param_distributions=params, cv=kfold,n_iter=60, random_state=23)
rgcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

#{'KNN__n_neighbors': 2, 'SCL': StandardScaler()}
#0.7793947733827284




















