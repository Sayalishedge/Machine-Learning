# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:45:22 2024

@author: dbda
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

milk = pd.read_csv(r"F:\PML\Datasets\milk.csv")
milk.set_index('Animal',inplace=True)


###################################################
# 1. Scaling
sc = StandardScaler().set_output(transform='pandas')
milk_scaled = sc.fit_transform(milk)


###################################################
# 2. Plotting dendrogram

#calculating the linkage ( average )
merging = linkage(milk_scaled, method='average')
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams['figure.figsize'] = fig_size

#plot dendrogram using varieties as lables
dendrogram(merging,
           labels = np.array(milk_scaled.index),
           leaf_rotation=45,
           leaf_font_size=10)
'''
#calculating the linkage ( single )
merging = linkage(milk_scaled, method='single')
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams['figure.figsize'] = fig_size

#plot dendrogram using varieties as lables
dendrogram(merging,
           labels = np.array(milk.index),
           leaf_rotation=45,
           leaf_font_size=10)


#calculating the linkage ( complete )
merging = linkage(milk_scaled, method='complete')
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams['figure.figsize'] = fig_size

#plot dendrogram using varieties as lables
dendrogram(merging,
           labels = np.array(milk.index),
           leaf_rotation=45,
           leaf_font_size=10)
'''
#################################################
# 3. Agglomerative clustering

clust = AgglomerativeClustering(n_clusters=4,linkage='average')
clust.fit(milk_scaled)
print(clust.labels_)
print(silhouette_score(milk_scaled, clust.labels_))

milk_clust = milk.copy()
milk_clust['Cluster'] = clust.labels_
milk_clust.sort_values('Cluster')

#################################################
# 4. Evaluating with silhouette score

clusters = [2,3,4,5,6]
score = []
for c in clusters:
    clust = AgglomerativeClustering(n_clusters=c,linkage='ward')
    clust.fit(milk_scaled)
    sc = silhouette_score(milk_scaled, clust.labels_)
    score.append(sc)

pd_scores = pd.DataFrame({'Number' : clusters,'Score' : score})
print(pd_scores.sort_values('Score',ascending=False))


#you can try linkage with methods as simple,average(ward is default)

#####################################################














