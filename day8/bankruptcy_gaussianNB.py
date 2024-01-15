# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:00:51 2024

@author: dbda
"""

import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB,GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

banking = pd.read_csv(r"F:\PML\Cases\Bankruptcy\Bankruptcy.csv")
banking.columns

X = banking.drop(['NO','D','YR'], axis=1)
y = banking['D']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, stratify=y, random_state=23)

nb = GaussianNB()

