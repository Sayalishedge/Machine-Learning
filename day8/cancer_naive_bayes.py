# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 12:16:03 2024

@author: dbda
"""

import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


cancer = pd.read_csv(r"F:\PML\Cases\Cancer\Cancer.csv")
cancer.info()
cancer['Class'].unique()
cancer.isnull().sum()

cancer_dum = pd.get_dummies(cancer,drop_first=True)
cancer_dum.isnull().sum()

X = cancer_dum.drop('Class_recurrence-events',axis=1)
y = cancer_dum['Class_recurrence-events']

params = {'alpha' : np.linspace(0,3, 10)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
nb = BernoulliNB()
gcv = GridSearchCV(nb,param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

#{'alpha': 3.0}
#-0.6055659570913339

####### KNN ######
cancer_dum = pd.get_dummies(cancer,drop_first=True)
cancer_dum.isnull().sum()

X = cancer_dum.drop('Class_recurrence-events',axis=1)
y = cancer_dum['Class_recurrence-events']

params = {'alpha' : np.linspace(0,3, 10)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(knn,param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

###### Logistic ###########
params = {'penalty' : [None, 'l1', 'l2', 'elasticnet'],
          'C': np.linspace(0,10,10),
          'l1_ratio' : np.linspace(0,1,5)}

lr = LogisticRegression(random_state=23, solver='saga')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(lr, param_grid=params, cv=kfold, scoring='neg_log_loss')

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

#{'C': 1.1111111111111112, 'l1_ratio': 0.0, 'penalty': None}
#-0.621611205149576