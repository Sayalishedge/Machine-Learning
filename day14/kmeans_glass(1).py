# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:24:33 2024

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

glass = pd.read_csv(r"F:\PML\Cases\Glass_Identification\Glass.csv")
glass.columns
y = glass['Type']
X = glass.drop('Type',axis=1)


# 1. Scaling
sc = StandardScaler().set_output(transform='pandas')
glass_scaled = sc.fit_transform(X)

#2 .Analysis using silhouette score
clusters = [2,3,4,5]
score = []
for c in clusters:
    clust = KMeans(n_clusters=c,random_state=23)
    clust.fit(glass_scaled)
    sc = silhouette_score(glass_scaled, clust.labels_)
    score.append(sc)

pd_scores = pd.DataFrame({'Number' : clusters,
                          'Score' : score})
print(pd_scores.sort_values('Score',ascending=False))

# 3. Best k
kn = KMeans(n_clusters=2, random_state=23)
kn.fit(glass_scaled)
print(kn.labels_)
glass_clust = X.copy()
glass_clust['Clust'] = kn.labels_

print(glass_clust.groupby('Clust').mean())
print(glass_clust['Clust'].value_counts()) 

#crosstab
pd.crosstab(index=y, columns=glass_clust['Clust'])

####################################################
# Xg boost for whole data
from xgboost import XGBRegressor,XGBRFRegressor
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

x_gbm = XGBRegressor(random_state=23)
print(x_gbm.get_params())

params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5,None],
          'n_estimators' : [25,50,100]
          }
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
x_gbm = GridSearchCV(x_gbm, param_grid=params,verbose=3, cv = kfold)

le = LabelEncoder()
le_y = le.fit_transform(y)
x_gbm.fit(X,le_y)
print(x_gbm.best_score_)

print(x_gbm.best_params_)
'''
# Xg boost for x==0
x_0 = X[glass_clust['Clust']==0]
y_0 = le_y[glass_clust['Clust']==0]
from xgboost import XGBRegressor,XGBRFRegressor
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

x_gbm = XGBRegressor(random_state=23)
print(x_gbm.get_params())

params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5,None],
          'n_estimators' : [25,50,100]
          }
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
x_gbm = GridSearchCV(x_gbm, param_grid=params,verbose=3, cv = kfold)

le = LabelEncoder()
le_y = le.fit_transform(y)
x_gbm.fit(x_0,y_0)
print(x_gbm.best_score_)
#0.4752397218744018
print(x_gbm.best_params_)
#{'learning_rate': 0.112, 'max_depth': 1, 'n_estimators': 25}

# Xg boost for x==1
x_1 = X[glass_clust['Clust']==1]
y_1 = le_y[glass_clust['Clust']==1]
from xgboost import XGBRegressor,XGBRFRegressor
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

x_gbm = XGBRegressor(random_state=23)
print(x_gbm.get_params())

params = {'learning_rate' : np.linspace(0.001, 1, 10),
          'max_depth' : [1,3,5,None],
          'n_estimators' : [25,50,100]
          }
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
x_gbm = GridSearchCV(x_gbm, param_grid=params,verbose=3, cv = kfold)

le = LabelEncoder()
le_y = le.fit_transform(y)
x_gbm.fit(x_1,y_1)
print(x_gbm.best_score_)
#0.07626734378997797
print(x_gbm.best_params_)
#{'learning_rate': 0.334, 'max_depth': 3, 'n_estimators': 25}
'''