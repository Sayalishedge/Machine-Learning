# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 11:32:29 2024

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

tel = pd.read_csv(r"F:\PML\Datasets\tel_bayes.csv")
tel.info()
tel.head()

dum_tel = pd.get_dummies(tel, drop_first=True)
dum_tel

X = dum_tel.drop('Response_not bought',axis=1)
y = dum_tel['Response_not bought']

nb = BernoulliNB(alpha=0, force_alpha=True)
nb.fit(X,y)

#########################################
telecom = pd.read_csv(r"F:\PML\Datasets\Telecom.csv")
telecom.info()
tel_dum = pd.get_dummies(telecom, drop_first=True)

nb = BernoulliNB(alpha=0, force_alpha=True)
X = tel_dum.drop('Response_Y', axis=1)
y = tel_dum['Response_Y']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=23)

nb.fit(X_train, y_train)
y_pred_proba = nb.predict_proba(X_test)
y_pred = nb.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_proba))



#book
print(metrics.classification_report(y_test, y_pred))
cm = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='.2f')

#############Grid Search CV ########################
params = {'alpha' : np.linspace(0,3, 10)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
nb = BernoulliNB()
gcv = GridSearchCV(nb,param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

#{'alpha': 3.0}
#-0.42947379653203416










