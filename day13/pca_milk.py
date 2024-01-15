# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:16:56 2024

@author: dbda
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA

milk = pd.read_csv(r"C:\Users\dbda\Desktop\PML\Datasets\milk.csv",index_col='Animal')

sc = StandardScaler().set_output(transform='pandas')

#apply scaler
m_scaled = sc.fit_transform(milk)

#pca transform
prcomp = PCA().set_output(transform='pandas')
components = prcomp.fit_transform(m_scaled)

print(components.var()) #the variance of components are decreasing
print(prcomp.explained_variance_)
#var_cov = np.cov(m_scaled.T)
#values,vectors = LA.eig(var_cov)
#print(values) # this is same as the prcomp.explained_variance_

total_var = np.sum(prcomp.explained_variance_)#sum of all the pc columns
percentage = (prcomp.explained_variance_/total_var)*100
print(percentage)
print(np.cumsum(percentage))

#####################################################################
from pca import pca
import matplotlib.pyplot as plt

model = pca() 
results = model.fit_transform(m_scaled, col_labels=milk.columns,row_labels=list(milk.index))
model.biplot(label=True, legend=True)
for i in np.arange(0, milk.shape[0]):
    plt.text(components.values[i,0],
             components.values[i,1],
             list(milk.index)[i])
plt.show()    



















