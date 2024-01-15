# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:27:04 2024

@author: dbda
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,accuracy_score,log_loss

le = LabelEncoder()

tiny = pd.read_csv(r"F:\PML\Datasets\tinydata.csv")
print(tiny)
tiny.info()
X = tiny[['Salt','Fat']]
y = le.fit_transform(tiny['Acceptance'])

mlp = MLPClassifier(random_state=23,hidden_layer_sizes=(3,))
mlp.fit(X,y)
print(mlp.coefs_)  #weights
print(mlp.intercepts_)  #biases

##################################################################

bank = pd.read_csv(r"F:\PML\Cases\Bankruptcy\Bankruptcy.csv")
bank.columns
X =bank.drop(['NO','D','YR'],axis=1)
y = bank['D']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=23)
mlp = MLPClassifier(random_state=23, hidden_layer_sizes= (10, 5,),activation='logistic')


#without scaling
mlp.fit(X,y)
y_pred = mlp.predict(X_test)
print(accuracy_score(y_test, y_pred))   #0.475
y_pred_proba = mlp.predict_proba(X_test)
print(log_loss(y_test,y_pred))     #18.922918029286507


#with scaling
sc = MinMaxScaler().set_output(transform='pandas')
pipe = Pipeline([('SCL', sc),('MLP',mlp)])
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))  #0.475 with and without pipeline








