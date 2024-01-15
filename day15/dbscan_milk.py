# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:10:22 2024

@author: dbda
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

milk = pd.read_csv(r"F:\PML\Datasets\milk.csv",index_col=0)
sc = StandardScaler().set_output(transform='pandas')
milk_scaled = sc.fit_transform(milk)

clust = DBSCAN(eps=0.8, min_samples=3)
clust.fit(milk_scaled)

print(clust.labels_)

########################################################
# Calculating with the outliers removed
clust = DBSCAN(eps=0.5, min_samples=4)
clust.fit(milk_scaled)
c_data = milk_scaled.copy()
if len(np.unique(clust.labels_)) > 2:
    c_data['labels'] = clust.labels_
    inliers = c_data[c_data['labels'] != -1]
    score  = silhouette_score(inliers.iloc[:,:-1], inliers['labels'])
    print(score)

# eps = 0.8, min = 3 : 0.5330377521235296
# eps = 0.8, min = 2 : 0.46467391846010603
# eps = 0.8, min = 4 : 0.4571511833668322

# eps = 0.6, min = 3 : 0.5344431042454363
# eps = 0.6, min = 2 : 0.5934459505692155
# eps = 0.6, min = 4 : 0.5519747727201489

# eps = 0.5, min = 3 : 0.5344431042454363
# eps = 0.5, min = 2 : 0.5934459505692155
# eps = 0.5, min = 4 : Error as silhouette score requires 2 clusters

########################################
# apply loop 
episilons = np.linspace(0.5,1,20)
min_points = [3,2,4]
scores = []
for e in episilons:
    for m in min_points:
        clust = DBSCAN(eps=e, min_samples=m)
        clust.fit(milk_scaled)
        c_data = milk_scaled.copy()
        if len(np.unique(clust.labels_)) > 2:
            c_data['labels'] = clust.labels_
            inliers = c_data[c_data['labels'] != -1]
            sil_score  = silhouette_score(inliers.iloc[:,:-1], inliers['labels'])
            scores.append([e,m,sil_score])
pd_scores = pd.DataFrame(scores, columns=['Episilon','Min_points','Silhouette_score']) .sort_values(by='Silhouette_score', ascending=False)       
print(pd_scores)
print(pd_scores.iloc[0]) # top one
pd_scores.columns

eps_1 = pd_scores['Episilon'].iloc[0]
min_pt = pd_scores['Min_points'].iloc[0]
clust = DBSCAN(eps=eps_1, min_samples=min_pt)
clust.fit(milk_scaled)
print(clust.labels_)

milk_scaled.columns
milk_scaled['Outliers'] = clust.labels_
outlier_animals = milk_scaled[milk_scaled['Outliers'] == -1]
print(outlier_animals.index)

