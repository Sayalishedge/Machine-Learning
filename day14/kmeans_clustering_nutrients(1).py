# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:16:58 2024

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


nutrient = pd.read_csv(r"F:\PML\Datasets\nutrient.csv")
nutrient.columns
nutrient.set_index('Food_Item',inplace=True)

###################################################
# 1. Scaling
sc = StandardScaler().set_output(transform='pandas')
nutrient_scaled = sc.fit_transform(nutrient)

###################################################
# 2. KMeans
kn = KMeans(n_clusters=3, random_state=23)
kn.fit(nutrient_scaled)
print(kn.labels_)

###################################################
#3.Analysis using silhouette score
clusters = [2,3,4,5,6]
score = []
for c in clusters:
    clust = KMeans(n_clusters=c,random_state=23)
    clust.fit(nutrient_scaled)
    sc = silhouette_score(nutrient_scaled, clust.labels_)
    score.append(sc)

pd_scores = pd.DataFrame({'Number' : clusters,
                          'Score' : score})
pd_scores.sort_values('Score',ascending=False)

######################################################
# 4. Best k
nutrient_clust = nutrient.copy()
nutrient_clust['Clust'] = kn.labels_
print(nutrient_clust.groupby('Clust').mean())
