# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:55:09 2024

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

milk = pd.read_csv(r"F:\PML\Datasets\milk.csv")
milk.set_index('Animal',inplace=True)

###################################################
# 1. Scaling
sc = StandardScaler().set_output(transform='pandas')
milk_scaled = sc.fit_transform(milk)

###################################################
# 2. KMeans
kn = KMeans(n_clusters=3, random_state=23)
kn.fit(milk_scaled)
print(kn.labels_)

###################################################
#3.Analysis using silhouette score
clusters = [2,3,4,5,6]
score = []
for c in clusters:
    clust = KMeans(n_clusters=c,random_state=23)
    clust.fit(milk_scaled)
    sc = silhouette_score(milk_scaled, clust.labels_)
    score.append(sc)

pd_scores = pd.DataFrame({'Number' : clusters,
                          'Score' : score})
pd_scores.sort_values('Score',ascending=False)

######################################################
# 4. Best k
milk_clust = milk.copy()
milk_clust['Clust'] = kn.labels_
print(milk_clust.groupby('Clust').mean())

