# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 15:18:12 2024

@author: dbda
"""

import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB,GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold,KFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression,ElasticNet
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

med = pd.read_csv(r"F:\PML\Cases\Medical Cost Personal\insurance.csv")
med_dum = pd.get_dummies(med,drop_first=True)

X = med_dum.drop('charges', axis=1)
y = med_dum['charges']

#### Elastic Net #####
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=23)
elastic = ElasticNet()
elastic.fit(X_train, y_train)

params = {'alpha': np.linspace(0,10,20),
         'l1_ratio': np.linspace(0.001,1,10)}
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(elastic, cv=kfold, param_grid=params)
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)

#{'alpha': 10.0, 'l1_ratio': 1.0}
#0.7473839422806123



##### KNN Regressor ########
params = {'n_neighbors': np.arange(1,20)}
kfold = KFold(n_splits=5, shuffle=True, random_state=23)
knn = KNeighborsRegressor()
gcv = GridSearchCV(knn, param_grid=params, cv=kfold, scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
#{'n_neighbors': 10}
#0.16067563081947606











