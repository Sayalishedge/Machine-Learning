# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:43:34 2024

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
from sklearn.decomposition import PCA

milk = pd.read_csv(r"F:\PML\Datasets\milk.csv")
sc = StandardScaler().set_output(transform='pandas')

X = milk.drop('Animal',axis=1)
y = milk['Animal']


milk.set_index('Animal',inplace=True)
#apply scaler
m_scaled = sc.fit_transform(milk)





##############################################
# kmeans
# 1. Scaling
sc = StandardScaler().set_output(transform='pandas')
milk_scaled = sc.fit_transform(X)

#2 .Analysis using silhouette score
clusters = [2,3,4,5]
score = []
for c in clusters:
    clust = KMeans(n_clusters=c,random_state=23)
    clust.fit(milk_scaled)
    sc = silhouette_score(milk_scaled, clust.labels_)
    score.append(sc)

pd_scores = pd.DataFrame({'Number' : clusters,
                          'Score' : score})
print(pd_scores.sort_values('Score',ascending=False))

# 3. Best k
kn = KMeans(n_clusters=3, random_state=23)
kn.fit(milk_scaled)
print(kn.labels_)
milk_clust = X.copy()
milk_clust['Clust'] = kn.labels_
milk_clust['Clust'] = milk_clust['Clust'].astype(str)

print(milk_clust.groupby('Clust').mean())
print(milk_clust['Clust'].value_counts()) 
#################################################
#  pca transform
prcomp = PCA().set_output(transform='pandas')
components = prcomp.fit_transform(m_scaled)

pc_data = components.iloc[:,:2]
pc_data['Clust'] = kn.labels_
pc_data['Clust'] = pc_data['Clust'].astype(str)

sns.scatterplot(data=pc_data, x='pca0', y='pca1',hue='Clust')

for i in np.arange(0, milk.shape[0]): 
    plt.text(pc_data.values[i,0],
            pc_data.values[i,1],
            list(milk.index)[i])
plt.show()    