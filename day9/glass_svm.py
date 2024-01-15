# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 13:02:19 2024

@author: dbda
"""

from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB,GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

glass = pd.read_csv(r"F:\PML\Cases\Glass_Identification\Glass.csv")
glass.columns

X = glass.drop(['Type'], axis=1)
y = glass['Type']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=23)


#WITH SCALING : LINEAR, POLY AND RADIAL
#LINEAR
svm_lin = SVC(kernel='linear',probability=True, random_state=23)
std_scl = StandardScaler()
min_max_scl = MinMaxScaler()
pipe_lin = Pipeline([('SCL', min_max_scl),('SVM', svm_lin)])

params = {'SVM__C':np.linspace(0.001, 6, 20),
          'SVM__funciton_shape' : ['ovo','ovr'],
          'SCL' : [None,std_scl ,min_max_scl]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_lin = GridSearchCV(pipe_lin, param_grid=params, cv=kfold, scoring='neg_log_loss', verbose=3)
gcv_lin.fit(X,y)
print(gcv_lin.best_params_)
print(gcv_lin.best_score_)
#{'SCL': None, 'SVM__C': 4.4213157894736845, 'SVM__coef0': -1.0, 'SVM__degree': 2}
#-0.47671452136438486



#POLY
svm_poly = SVC(kernel='poly',probability=True, random_state=23)
std_scl = StandardScaler()
min_max_scl = MinMaxScaler()
pipe_poly = Pipeline([('SCL', min_max_scl),('SVM', svm_poly)])

params = {'SVM__C':np.linspace(0.001, 6, 20),
          'SVM__degree' : [2,3,5,6,7],
          'SVM__coef0' : np.linspace(-1,2,10),
          'SCL' : [None,std_scl ,min_max_scl]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_poly = GridSearchCV(pipe_poly, param_grid=params, cv=kfold, scoring='neg_log_loss', verbose=3)
gcv_poly.fit(X,y)
print(gcv_poly.best_params_)
print(gcv_poly.best_score_)
#{'SCL': StandardScaler(), 'SVM__C': 3.1583684210526313, 'SVM__coef0': -0.33333333333333337, 'SVM__degree': 2}
#-0.4163006032972584



#RADIAL
svm_rbf = SVC(kernel='rbf',probability=True, random_state=23)
std_scl = StandardScaler()
min_max_scl = MinMaxScaler()
pipe_rbf = Pipeline([('SCL', min_max_scl),('SVM', svm_rbf)])

params = {'SVM__C':np.linspace(0.001, 6, 20),
          'SVM__gamma' : np.linspace(0.001, 5, 10),
          'SCL' : [None,std_scl ,min_max_scl]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_rbf = GridSearchCV(pipe_rbf, param_grid=params, cv=kfold, scoring='neg_log_loss', verbose=3)
gcv_rbf.fit(X,y)
print(gcv_rbf.best_params_)
print(gcv_rbf.best_score_)
#{'SCL': StandardScaler(), 'SVM__C': 6.0, 'SVM__gamma': 3.3336666666666663}
#-0.40911508335755914
