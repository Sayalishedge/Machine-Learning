# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 08:52:49 2024

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

banking = pd.read_csv(r"F:\PML\Cases\Bankruptcy\Bankruptcy.csv")
banking.columns

X = banking.drop(['NO','D','YR'], axis=1)
y = banking['D']

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=23)

#predict using lda
lda.fit(X_train, y_train)
lda_y_pred = lda.predict(X_test)
print(accuracy_score(y_test, lda_y_pred))  #0.65
y_pred_prob = lda.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))    #0.6935379378951569

#predict using qda
qda.fit(X_train, y_train)
qda_y_pred = qda.predict(X_test)
print(accuracy_score(y_test, qda_y_pred))  #0.775
y_pred_prob = qda.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))    #4.426586124499626


#Accuracy is better in qda but logloss is worse.
######################################################################

#GRID SEARCH USING LDA
lda = LinearDiscriminantAnalysis()
params = {'solver' : ['svd','lsqr','eigen']}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(lda, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)  #{'solver': 'svd'}
print(gcv.best_score_)   #-0.9337131138729617


#GRID SEARCH USING QDA
qda = QuadraticDiscriminantAnalysis()
params = {'reg_param' : np.linspace(0,1,10)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(qda, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)  #{'reg_param': 0.2222222222222222}
print(gcv.best_score_)   #-1.360081357096572

#LDA is better again.









