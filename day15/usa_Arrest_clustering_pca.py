# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 13:10:19 2024

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

arrests = pd.read_csv(r"F:\PML\Datasets\USArrests.csv",index_col=0)
sc = StandardScaler().set_output(transform='pandas')
arrests_scaled=sc.fit_transform(arrests)

clusters = [2,3,4,5,6]
score = []
for c in clusters:
    clust = KMeans(random_state=23, n_clusters=c)
    clust.fit(arrests_scaled)
    sc = silhouette_score(arrests_scaled, clust.labels_)
    score.append(sc)
    
pd_score = pd.DataFrame({'Number':clusters,
                         'Score':score})
pd_score.sort_values('Score', ascending=False)

#### best k
km = KMeans(random_state=23, n_clusters=2)
km.fit(arrests_scaled)

arrest_clust = arrests.copy()
arrest_clust['Clust'] = km.labels_
print(arrest_clust.groupby('Clust').mean())

######## PCA
from sklearn.decomposition import PCA

prcomp = PCA().set_output(transform='pandas')
PC_data = prcomp.fit_transform(arrests_scaled)
PC_data = PC_data.iloc[:,:2]
PC_data['Clust'] = km.labels_
PC_data['Clust'] = PC_data['Clust'].astype(str)

sns.scatterplot(data=PC_data, x='pca0', y='pca1',
                hue='Clust')
for i in np.arange(0, arrests.shape[0] ):
    plt.text(PC_data.values[i,0], 
             PC_data.values[i,1], 
             list(arrests.index)[i])
plt.show()