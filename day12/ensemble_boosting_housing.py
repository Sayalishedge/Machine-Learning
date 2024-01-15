# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:01:15 2024

@author: dbda
"""
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
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor,StackingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

housing = pd.read_csv(r"F:\PML\Datasets\Housing.csv")
housing.isnull().sum()
housing.columns
dum_house = pd.get_dummies(housing,drop_first=True)

X = dum_house.drop(['price'], axis=1).values
y = dum_house['price'].values

KFold(n_splits=5, shuffle=True, random_state=23)

dtc = DecisionTreeRegressor(random_state=23, max_depth=1)
lr = LogisticRegression()
svm = SVC(probability=True, random_state=23)

###############################################
# Ada boost             #error max_iter
ada = AdaBoostRegressor(random_state=23)

params = {'estimator':[dtc, lr, svm],
          'n_estimators':[25,50,100]}


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_ada = GridSearchCV(ada, param_grid=params, cv = kfold,verbose=3)
gcv_ada.fit(X,y)
print(gcv_ada.best_score_)
#0.3322625850544519
print(gcv_ada.best_params_)
#{'estimator': DecisionTreeRegressor(max_depth=1, random_state=23), 'n_estimators': 25}

################################

# GBM

gbm =GradientBoostingRegressor(random_state=23)
print(gbm.get_params())

params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5, None],
          'n_estimators' : [50,100,150]}


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_gbm = GridSearchCV(gbm, param_grid=params, cv = kfold,verbose=3)
gcv_gbm.fit(X,y)
print(gcv_gbm.best_score_)
#0.6267222599115818
print(gcv_gbm.best_params_)
#{'learning_rate': 0.556, 'max_depth': 1, 'n_estimators': 100}




##########################################################################

# XG Boost

from xgboost import XGBRFRegressor
x_gbm = XGBRFRegressor(random_state=23)
print(x_gbm.get_params())

params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5, None],
          'n_estimators' : [50,100,150]
          }
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_x_gbm = GridSearchCV(x_gbm, param_grid=params,verbose=3, cv = kfold)
gcv_x_gbm.fit(X,y)
print(gcv_x_gbm.best_score_)
# 0.5763527890377531

print(gcv_x_gbm.best_params_)
# {'learning_rate': 1.0, 'max_depth': None, 'n_estimators': 100}

###########################################################################

# Light GBM
from lightgbm import LGBMRegressor
l_gbm = LGBMRegressor(random_state=23)

print(l_gbm.get_params())
params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5, None],
          'n_estimators' : [50,100,150]}

gcv_l_gbm = GridSearchCV(l_gbm, param_grid=params,verbose=3, cv = kfold)
gcv_l_gbm.fit(X,y)
print(gcv_l_gbm.best_score_)
#0.6322606183305712
print(gcv_l_gbm.best_params_)
#{'learning_rate': 0.778, 'max_depth': 1, 'n_estimators': 150}

###########################################################################

# Catboost

from catboost import CatBoostRegressor
c_gbm = CatBoostRegressor(random_state=23)

print(c_gbm.get_params())
params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5, None],
          'n_estimators' : [50,100,150]}

gcv_c_gbm = GridSearchCV(c_gbm, param_grid=params,verbose=3, cv = kfold)
gcv_c_gbm.fit(X,y)
print(gcv_c_gbm.best_score_)
#0.6311047248736175
print(gcv_c_gbm.best_params_)
#{'learning_rate': 0.112, 'max_depth': 3, 'n_estimators': 150}


##########################################################

#w/o hot encoding

X = dum_house.drop(['price'], axis=1)
y = dum_house['price']

all_cats = list(X.dtypes[X.dtypes==object].index)
c_gbm = CatBoostRegressor(random_state=23)
params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5, None],
          'n_estimators' : [50,100,150],
          'cat_features' : [all_cats]}

gcv_c_gbm = GridSearchCV(c_gbm, param_grid=params,verbose=3, cv = kfold)
gcv_c_gbm.fit(X,y)
print(gcv_c_gbm.best_score_)
#0.6311047248736175
print(gcv_c_gbm.best_params_)
#{'cat_features': [], 'learning_rate': 0.112, 'max_depth': 3, 'n_estimators': 150}


##############################################################
# stacking

svm = SVC(probability=True, random_state=23)
lr = LogisticRegression()
dtc = DecisionTreeRegressor(random_state=23)
gbm = GradientBoostingRegressor(random_state=23)
rf = RandomForestRegressor(random_state=23)
el = ElasticNet(random_state=23)
stack = StackingRegressor([('LR', lr),('SVM',svm),('TREE', dtc),('Gradient_Boosting',gbm),('EL',el)],
                           passthrough=True,
                           final_estimator=rf)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=23)

kfold = KFold(n_splits=5, shuffle=True, random_state=23)

print(stack.get_params())
params = {''}
