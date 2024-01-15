# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:48:14 2024

@author: dbda
"""

import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

bank = pd.read_csv(r"F:\PML\Cases\Bankruptcy\Bankruptcy.csv")
#print(bank.head(3))
#print(bank.columns)

X = bank.drop(['NO','YR','D'], axis=1).values
y = bank['D'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.3, random_state=23)

svm = SVC(probability=True, random_state=23)
lr = LogisticRegression()
dtc = DecisionTreeClassifier(random_state=23)
gbm = GradientBoostingClassifier(random_state=23)

stack = StackingClassifier([('LR', lr),('SVM',svm),('TREE', dtc),('Gradient_Boosting',gbm)],
                           passthrough=True,
                           final_estimator=gbm)

stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
print(accuracy_score(y_test, y_pred))   #0.8

y_pred_proba = stack.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba))  #0.6189231345620148

##########################################################

print(stack.get_params())

params = {'SVM__C' : [0.5,1,1.5],
          'TREE__max_depth' : [None,3,5],
          'Gradient_Boosting__learning_rate' : [0.1,0.5],
          'passthrough' : [True,False]}
kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=23)
gcv_stack = GridSearchCV(stack, param_grid=params, verbose=3, cv=kfold, scoring='neg_log_loss')
gcv_stack.fit(X,y)
print(gcv_stack.best_params_)
print(gcv_stack.best_score_)

#{'Gradient_Boosting__learning_rate': 0.1, 'SVM__C': 0.5, 'TREE__max_depth': 3, 'passthrough': False}
#-0.5985667899510808









