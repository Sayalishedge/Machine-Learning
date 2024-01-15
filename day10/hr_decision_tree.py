# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:03:37 2024

@author: dbda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, log_loss

hr = pd.read_csv(r"F:\PML\Cases\human-resources-analytics\HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)

dum_hr.columns
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=23)


dtc = DecisionTreeClassifier(random_state=23, max_depth=2)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred))   #0.840408979773283
y_pred_proba = dtc.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba))   #0.29865775782872894

#######depth = none

dtc = DecisionTreeClassifier(random_state=23, max_depth=None)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred))  
y_pred_proba = dtc.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba))  

#0.9777728384085352
#0.8011481082266539

#######depth = 4

dtc = DecisionTreeClassifier(random_state=23, max_depth=4)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred))  
y_pred_proba = dtc.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba)) 

#0.9695487886196933
#0.12464601022150539

#######depth = 6

dtc = DecisionTreeClassifier(random_state=23, max_depth=6)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred))  
y_pred_proba = dtc.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba)) 

#0.9771060235607912
#0.11004442168808944






