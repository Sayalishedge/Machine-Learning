# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 18:32:49 2024

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

rfm = pd.read_csv(r"F:\PML\Cases\Recency Frequency Monetary/rfm_data_customer.csv")
rfm.columns
rfm.set_index('customer_id', inplace=True)
rfm.drop('most_recent_visit',axis=1, inplace=True)

# 1. Scaling
sc = StandardScaler().set_output(transform='pandas')
rfm_scaled = sc.fit_transform(rfm)

# 2. KMeans
kn = KMeans(n_clusters=3, random_state=23)
kn.fit(rfm_scaled)
print(kn.labels_)

#3.Analysis using silhouette score
clusters = [2,3,4,5]
score = []
for c in clusters:
    clust = KMeans(n_clusters=c,random_state=23)
    clust.fit(rfm_scaled)
    sc = silhouette_score(rfm_scaled, clust.labels_)
    score.append(sc)

pd_scores = pd.DataFrame({'Number' : clusters,
                          'Score' : score})
print(pd_scores.sort_values('Score',ascending=False))

# 4. Best k
rfm_clust = rfm.copy()
rfm_clust['Clust'] = kn.labels_

print(rfm_clust.groupby('Clust').mean())
#print(rfm_clust.sort_values('Clust',ascending=False))
