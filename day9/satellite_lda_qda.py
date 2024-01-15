# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:46:37 2024

@author: dbda
"""

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

satellite = pd.read_csv(r"F:\PML\Cases\Satellite_Imaging\Satellite.csv", sep=';')
satellite.columns

X = satellite.drop(['classes'], axis=1)
y = satellite['classes']


lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=23)

#predict using lda
lda.fit(X_train, y_train)
lda_y_pred = lda.predict(X_test)
print(accuracy_score(y_test, lda_y_pred))  #0.8420507509062661
y_pred_prob = lda.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))    #0.6298212434256166

#predict using qda
qda.fit(X_train, y_train)
qda_y_pred = qda.predict(X_test)
print(accuracy_score(y_test, qda_y_pred))  #0.8487830139823925
y_pred_prob = qda.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))    #0.9725297849896979


#Accuracy is slightly increased in qda and logloss is also bad. LDA is prefered
######################################################################

#GRID SEARCH USING LDA
lda = LinearDiscriminantAnalysis()
params = {'solver' : ['svd','lsqr','eigen']}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(lda, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)  #{'solver': 'svd'}
print(gcv.best_score_)   #-0.5872905454633512


#GRID SEARCH USING QDA
qda = QuadraticDiscriminantAnalysis()
params = {'reg_param' : np.linspace(0,1,10)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(qda, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)  #{'reg_param': 0.0}
print(gcv.best_score_)   #-0.8814849658082482

#LDA is better again.

#########################################################################

#LOGISTIC REGRESSION
params = {'penalty' : [None, 'l1', 'l2', 'elasticnet'],
          'C': np.linspace(0,10,10),
          'l1_ratio' : np.linspace(0,1,5)}

lr = LogisticRegression(random_state=23, solver='saga')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(lr, param_grid=params, cv=kfold, scoring='neg_log_loss')

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)
 
