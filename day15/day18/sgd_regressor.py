# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:27:21 2024

@author: dbda
"""

from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

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