# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:27:21 2024

@author: dbda
"""

from sklearn.linear_model import SGDRegressor,SGDClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

############################## SGD REGRESSOR ##################################
pizza = pd.read_csv(r"F:\PML\Datasets\pizza.csv")
sc = MinMaxScaler().set_output(transform='pandas')


sgd = SGDRegressor(random_state=23,penalty=None,
                   learning_rate='constant',
                   eta0=0.2)

pipe = Pipeline([('SCL', sc),('SGD',sgd)])

X = pizza[['Promote']]
y = pizza[['Sales']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=23)


pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test, y_pred))  #0.9932594209164927

############################## SGD CLASSIFIER ##################################

bank = pd.read_csv(r"F:\PML\Cases\Bankruptcy\Bankruptcy.csv")
sgd = SGDClassifier(loss='log_loss')

pipe = Pipeline([('SCL', sc),('SGD',sgd)])
print(pipe.get_params())
params = {'SGD__learning_rate' : ['constant','optimal','invscaling','adaptive'],
          'SGD__penalty' : ['l2', 'l1', 'elasticnet', None],
          'SGD__eta0' : [0.01,0.2,0.3]
          }

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
#{'SGD__eta0': 0.2, 'SGD__learning_rate': 'adaptive', 'SGD__penalty': 'elasticnet'}

print(gcv.best_score_)
#-0.4630184775711762
