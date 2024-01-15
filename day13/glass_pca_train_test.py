# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 19:18:58 2024

@author: dbda
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


glass = pd.read_csv(r"F:\PML\Cases\Glass_Identification\Glass.csv")
glass.columns

#glass.isnull().sum()

X = glass.drop('Type', axis=1)
y = glass['Type']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,stratify=y, random_state=23)


#glass -> train -> scl -> pca -> svm.fit

sc = StandardScaler().set_output(transform='pandas')
prcomp = PCA().set_output(transform='pandas')

scl_trn = sc.fit_transform(X_train)
trn_pca = prcomp.fit_transform(scl_trn)


total_var = np.sum(prcomp.explained_variance_)#sum of all the pc columns
percentage = (prcomp.explained_variance_/total_var)*100
print(percentage)
print(np.cumsum(percentage))

#c=5
svm = SVC(kernel='linear')
svm.fit(trn_pca.iloc[:,:5],y_train)


#glass -> test -> scl -> pca -> svm.predict() ->accuracy

scl_tst = sc.transform(X_test)
tst_pca = prcomp.transform(scl_tst)
y_pred = svm.predict(tst_pca.iloc[:,:5])
print(accuracy_score(y_test, y_pred))
#0.5846153846153846


###############################################################
# with pipeline
from sklearn.pipeline import Pipeline

sc = StandardScaler().set_output(transform='pandas')
prcomp = PCA(n_components=5).set_output(transform='pandas')
svm = SVC(kernel='linear')

pipe = Pipeline([('SCL',sc),
                 ('PCA',prcomp),
                 ('SVM',svm)])
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))
#0.5846153846153846

######################################################################
#refer sir code
from sklearn.model_selection import GridSearchCV

print(pipe.get_params())
params = {'SVM__C' : np.linspace(0.001,6,20),
          'PCA__n_components' : [4,3,5,6],
          }
sc = StandardScaler().set_output(transform='pandas')
prcomp = PCA(n_components=5).set_output(transform='pandas')
svm = SVC(kernel='linear')

pipe = Pipeline([('SCL',sc),
                 ('PCA',prcomp),
                 ('SVM',svm)])

kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=23)
gcv_lin = GridSearchCV(pipe, param_grid=params, verbose=3, cv=kfold)
gcv_lin.fit(X,y)
print(gcv_lin.best_params_)
#{'PCA__n_components': 3, 'SVM__C': 1}
print(gcv_lin.best_score_)
#0.6446290143964564











