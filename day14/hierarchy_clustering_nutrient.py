# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 17:32:35 2024

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

nutrient = pd.read_csv(r"F:\PML\Datasets\nutrient.csv")
nutrient.columns
nutrient.set_index('Food_Item',inplace=True)

###############################################
# 1. Scaling
sc = StandardScaler().set_output(transform='pandas')
nutrient_scaled = sc.fit_transform(nutrient)

# 2. Dendrogram
#calculating the linkage ( ward )
merging = linkage(nutrient_scaled, method='average')
fig_size = plt.rcParams['figure.figsize']
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams['figure.figsize'] = fig_size

#plot dendrogram using varieties as lables
dendrogram(merging,
           labels = np.array(nutrient_scaled.index),
           leaf_rotation=45,
           leaf_font_size=10)

#3.Analysis using silhouette score
clusters = [2,3,4,5,6]
score = []
for c in clusters:
    clust = AgglomerativeClustering(n_clusters=c,linkage='average')
    clust.fit(nutrient_scaled)
    sc = silhouette_score(nutrient_scaled, clust.labels_)
    score.append(sc)

pd_scores = pd.DataFrame({'Number' : clusters,
                          'Score' : score})
pd_scores.sort_values('Score',ascending=False)

#ward = 4 clusters for 0.415 score
#average = 3 clusters for 0.445 score
#simple = 2 clusters for 0.448 score <-- best

