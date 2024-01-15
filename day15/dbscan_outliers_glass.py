
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:09:15 2024

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


glass = pd.read_csv(r"F:\PML\Cases\Glass_Identification\Glass.csv")
glass.columns
glass.set_index('Type',inplace=True)


# 1. Scaling
sc = StandardScaler().set_output(transform='pandas')
glass_scaled = sc.fit_transform(glass)



episilons = np.linspace(0.5,1,20)
min_points = [3,2,4]
scores = []
for e in episilons:
    for m in min_points:
        clust = DBSCAN(eps=e, min_samples=m)
        clust.fit(glass_scaled)
        c_data = glass_scaled.copy()
        if len(np.unique(clust.labels_)) > 2:
            c_data['labels'] = clust.labels_
            inliers = c_data[c_data['labels'] != -1]
            sil_score  = silhouette_score(inliers.iloc[:,:-1], inliers['labels'])
            scores.append([e,m,sil_score])
pd_scores = pd.DataFrame(scores, columns=['Episilon','Min_points','Silhouette_score']) .sort_values(by='Silhouette_score', ascending=False)       
#print(pd_scores)
#print(pd_scores.iloc[0]) # top one
pd_scores.columns

eps_1 = pd_scores['Episilon'].iloc[0]
min_pt = pd_scores['Min_points'].iloc[0]
clust = DBSCAN(eps=eps_1, min_samples=min_pt)
clust.fit(glass_scaled)
#print(clust.labels_)

glass_scaled.columns
glass_scaled['Outliers'] = clust.labels_

# No. of clusters
cl = glass_scaled['Outliers'].unique() # no. of clusters
print("No. of clusters: ",len(cl)-1)

# Inliers and Outliers
outliers = glass_scaled[glass_scaled['Outliers'] == -1]
#print(outliers.index)
print("Outliers: ",len(outliers.index))
print("Inliers : ",len(clust.labels_ >=0))  #inliers
glass_scaled['Outliers'].value_counts()
