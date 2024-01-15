# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:56:05 2024

@author: dbda
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

bank = pd.read_csv(r"F:\PML\Cases\Bankruptcy\Bankruptcy.csv")
bank.columns

bank.drop(['NO','D','YR'], axis=1, inplace=True)
print(bank.columns)


# 1. Scaling
sc = StandardScaler().set_output(transform='pandas')
bank_scaled = sc.fit_transform(bank)


#2 .Analysis using silhouette score
clusters = [2,3,4,5]
score = []
for c in clusters:
    clust = KMeans(n_clusters=c,random_state=23)
    clust.fit(bank_scaled)
    sc = silhouette_score(bank_scaled, clust.labels_)
    score.append(sc)

pd_scores = pd.DataFrame({'Number' : clusters,
                          'Score' : score})
print(pd_scores.sort_values('Score',ascending=False))

# 3. Best k
kn = KMeans(n_clusters=2, random_state=23)
kn.fit(bank_scaled)
print(kn.labels_)
bank_clust = bank.copy()
bank_clust['Clust'] = kn.labels_

print(bank_clust.groupby('Clust').mean())
#print(bank_clust.sort_values('Clust',ascending=False))

print(bank_clust['Clust'].value_counts()) #one point in one set ---> outlier

#the outlier 
bank_clust[bank_clust['Clust']==1]
print(bank_clust.columns)

########################################################################
# GCV for with outliers
#refer sir code
X = bank
y = bank_clust['Clust']
X_wo = bank[bank_clust['Clust']==0]
y_wo = y[bank_clust['Clust']==0]

from xgboost import XGBRegressor,XGBRFRegressor
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
x_gbm = XGBRegressor(random_state=23)
print(x_gbm.get_params())

params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5,None],
          'n_estimators' : [25,50,100]
          }
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
x_gbm = GridSearchCV(x_gbm, param_grid=params,verbose=3, cv = kfold)
x_gbm.fit(X_wo,y_wo)
print(x_gbm.best_score_)

print(x_gbm.best_params_)

########################################################################
# GCV for without outliers

from xgboost import XGBRegressor,XGBRFRegressor
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
x_gbm = XGBRegressor(random_state=23)
print(x_gbm.get_params())

params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5,None],
          'n_estimators' : [25,50,100]
          }
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
x_gbm = GridSearchCV(x_gbm, param_grid=params,verbose=3, cv = kfold)
x_gbm.fit(X_wo,y_wo)
print(x_gbm.best_score_)

print(x_gbm.best_params_)













