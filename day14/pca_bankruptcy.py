# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:13:52 2024

@author: dbda
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold

bank = pd.read_csv(r"F:\PML\Cases\Bankruptcy\Bankruptcy.csv")
X = bank.drop(['NO','YR','D'], axis=1).values
y = bank['D'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,random_state=23)

scaler = StandardScaler().set_output(transform='pandas')
prcomp = PCA(n_components=0.75).set_output(transform='pandas')

pipe = Pipeline([('SCL',scaler),('PCA',prcomp)])

pca_data = pipe.fit_transform(X)

print(np.cumsum(prcomp.explained_variance_ratio_*100))


# GCV
print(pipe.get_params())
params = {'SVM__C': np.linspace(0.001, 6, 20),
          'PCA__n_components':[0.75,0.8,0.85,0.90,0.95]}
scaler = StandardScaler().set_output(transform='pandas')
prcomp = PCA().set_output(transform='pandas')
svm = SVC(kernel='linear', probability=True,
          random_state=23)
pipe = Pipeline([('SCL',scaler),('PCA',prcomp),('SVM',svm)])
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_lin = GridSearchCV(pipe, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', verbose=3)
gcv_lin.fit(X, y)
print(gcv_lin.best_params_)
#{'PCA__n_components': 0.9, 'SVM__C': 0.9482105263157895}

print(gcv_lin.best_score_)
#-0.4550871562757785











