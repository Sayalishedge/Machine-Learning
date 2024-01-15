# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:55:00 2024

@author: dbda
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

kyphosis = pd.read_csv(r"F:\PML\Cases\Kyphosis\Kyphosis.csv")
kyphosis.head()

X = kyphosis.drop('Kyphosis', axis=1)
y = kyphosis['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=23,stratify=y)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_trn_scl = scaler.transform(X_train)
X_tst_scl = scaler.transform(X_test)

scores = dict()
for k in [3,5,7]:
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_proba = knn.predict_proba(X_test)
    scores_std[str(k)] = log_loss(y_test, y_pred_proba)




#y_pred = knn.predict(X_tst_scl)
#print(accuracy_score(X_test, y_pred))




