# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:44:28 2024

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

milk = pd.read_csv(r"F:\PML\Datasets\milk.csv",index_col=0)
sc = StandardScaler().set_output(transform='pandas')
milk_scaled = sc.fit_transform(milk)

clusts = np.arange(2,25)
wss = []
for c in clusts:
    km = KMeans(random_state=23, n_clusters=c)
    km.fit(milk_scaled)
    wss.append(km.inertia_)
    
plt.scatter(clusts,wss,c='red')    
plt.plot(clusts, wss)
plt.xlabel("No of clusters")
plt.ylabel("WSS")
plt.show()


###############################################################
arrests = pd.read_csv(r"F:\PML\Datasets\USArrests.csv",index_col=0)
sc = StandardScaler().set_output(transform='pandas')
arrests_scaled=sc.fit_transform(arrests)


clusts = np.arange(2,25)
wss = []
for c in clusts:
    km = KMeans(random_state=23, n_clusters=c)
    km.fit(arrests_scaled)
    wss.append(km.inertia_)
    
plt.scatter(clusts,wss,c='red')    
plt.plot(clusts, wss)
plt.xlabel("No of clusters")
plt.ylabel("WSS")
plt.show()