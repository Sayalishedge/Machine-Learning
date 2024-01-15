# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 08:46:11 2024

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

img = pd.read_csv(r"F:\PML\Cases\Image_Segmentation\Image_Segmention.csv")
img.columns

print(img.isnull().sum())

X = img.drop('Class', axis=1).values
y = img['Class'].values

#Without Scaling
params = {'n_neighbors': np.arange(1,30)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(knn, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

#{'n_neighbors': 22}
#-0.8193756232998135

#Using both scaler
std_scl = StandardScaler()
knn = KNeighborsClassifier()
min_max_scl = MinMaxScaler()
pipe = Pipeline([('SCL', min_max_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors' : np.arange(1,30),
          'SCL' : [min_max_scl,std_scl ]}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

#{'KNN__n_neighbors': 18, 'SCL': StandardScaler()}
#-0.5249851414233093
