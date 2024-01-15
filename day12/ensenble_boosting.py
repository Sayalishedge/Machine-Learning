# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:25:53 2024

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
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


bank = pd.read_csv(r"F:\PML\Cases\Bankruptcy\Bankruptcy.csv")
#print(bank.head(3))
#print(bank.columns)

X = bank.drop(['NO','YR','D'], axis=1)
y = bank['D']

dtc = DecisionTreeClassifier(random_state=23, max_depth=1)
lr = LogisticRegression()
svm = SVC(probability=True, random_state=23)

###############################################
# Ada boost
ada = AdaBoostClassifier(random_state=23)

params = {'estimator':[dtc, lr, svm],
          'n_estimators':[25,50,100]}

ada.fit(X, y)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_ada = GridSearchCV(ada, param_grid=params, cv = kfold, scoring = 'neg_log_loss')
gcv_ada.fit(X,y)
print(gcv_ada.best_score_)
#-0.5848542284033642
print(gcv_ada.best_params_)
#{'estimator': DecisionTreeClassifier(max_depth=1, random_state=23), 'n_estimators': 25}

################################

# GBM

gbm =GradientBoostingClassifier(random_state=23)
print(gbm.get_params())

params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5, None],
          'n_estimators' : [50,100,150]}


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_gbm = GridSearchCV(gbm, param_grid=params, cv = kfold, scoring = 'neg_log_loss')
gcv_gbm.fit(X,y)
print(gcv_gbm.best_score_)
#-0.5146374231822894
print(gcv_gbm.best_params_)
#{'learning_rate': 0.112, 'max_depth': 3, 'n_estimators': 50}

#for max_depth = 1
#-0.39816572682894985
#{'learning_rate': 0.112, 'max_depth': 1, 'n_estimators': 50}


##########################################################################

# XG Boost

from xgboost import XGBClassifier
x_gbm = XGBClassifier(random_state=23)
print(x_gbm.get_params())

params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5, None],
          'n_estimators' : [50,100,150],
          'tree_method' : ['hist','approx','exact']}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_x_gbm = GridSearchCV(x_gbm, param_grid=params,verbose=3, cv = kfold, scoring = 'neg_log_loss')
gcv_x_gbm.fit(X,y)
print(gcv_x_gbm.best_score_)
# using tree method : -0.3844250084112245
#without using tree method : -0.3975700557967484
print(gcv_x_gbm.best_params_)
# using tree method :{'learning_rate': 0.112, 'max_depth': 1, 'n_estimators': 50}
# without using tree method :{'learning_rate': 0.112, 'max_depth': 1, 'n_estimators': 50, 'tree_method': 'exact'}


###########################################################################

# Light GBM
from lightgbm import LGBMClassifier
l_gbm = LGBMClassifier(random_state=23)

print(l_gbm.get_params())
params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5, None],
          'n_estimators' : [50,100,150]}

gcv_l_gbm = GridSearchCV(l_gbm, param_grid=params,verbose=3, cv = kfold, scoring = 'neg_log_loss')
gcv_l_gbm.fit(X,y)
print(gcv_l_gbm.best_score_)
#-0.3847377117620936
print(gcv_l_gbm.best_params_)
#{'learning_rate': 0.112, 'max_depth': 1, 'n_estimators': 50}

###########################################################################

# Catboost

from catboost import CatBoostClassifier
c_gbm = CatBoostClassifier(random_state=23)

print(c_gbm.get_params())
params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5, None],
          'n_estimators' : [50,100,150]}

gcv_c_gbm = GridSearchCV(c_gbm, param_grid=params,verbose=3, cv = kfold, scoring = 'neg_log_loss')
gcv_c_gbm.fit(X,y)
print(gcv_c_gbm.best_score_)
#-0.3885479938473245
print(gcv_c_gbm.best_params_)
#{'learning_rate': 0.112, 'max_depth': 1, 'n_estimators': 50}




