# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 08:13:58 2024

@author: dbda
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

bank = pd.read_csv(r"F:\PML\Cases\Bankruptcy\Bankruptcy.csv")
#print(bank.head(3))
#print(bank.columns)

X = bank.drop(['NO','YR','D'], axis=1).values
y = bank['D'].values


################## Grid Search CV ###########################
params = {'n_neighbors': np.arange(1,20)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(knn, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
#{'n_neighbors': 15}
#-0.6100792547642208

##################### With Standard Scaling ###########################

std_scl = StandardScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('SCL', std_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors' : np.arange(1,21)}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
#{'KNN__n_neighbors': 19}
#-0.45951383263289236

##################### With MinMax Scaler ###########################

min_max_scl = MinMaxScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('SCL', min_max_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors' : np.arange(1,21)}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

#{'KNN__n_neighbors': 20}
#-0.5086037973088182

##################### With Both scaling ###########################

knn = KNeighborsClassifier()
pipe = Pipeline([('SCL', min_max_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors' : np.arange(1,21),
          'SCL' : [min_max_scl,std_scl]}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

#{'KNN__n_neighbors': 19, 'SCL': StandardScaler()}
#-0.45951383263289236


#################### Glass Identification #############################

glass = pd.read_csv(r"F:\PML\Cases\Glass_Identification\Glass.csv")
print(glass.columns)

X = glass.drop('Type', axis=1)
y = glass['Type']


params = {'n_neighbors': np.arange(1,20)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)


std_scl = StandardScaler()
knn = KNeighborsClassifier()
min_max_scl = MinMaxScaler()
pipe = Pipeline([('SCL', min_max_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors' : np.arange(1,30),
          'SCL' : [min_max_scl,std_scl]}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


