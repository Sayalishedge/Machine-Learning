# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:52:44 2024

@author: dbda
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA

train = pd.read_csv(r"C:\Users\dbda\Desktop\PML\competitions\blackfriday\train.csv")
train.columns

purch_cat = train[['Product_Category_1','Purchase']]

tot_pur_grp = pd.read_csv(r"C:\Users\dbda\Desktop\PML\competitions\blackfriday\tot_pur_age_cat.csv",index_col='Row Labels')

sc = StandardScaler().set_output(transform='pandas')
pur_scaled = sc.fit_transform(tot_pur_grp)

prcomp = PCA().set_output(transform='pandas')
components = prcomp.fit_transform(pur_scaled)

from pca import pca
import matplotlib.pyplot as plt

model = pca() 
results = model.fit_transform(pur_scaled, col_labels=pur_scaled.columns,row_labels=list(pur_scaled.index))
model.biplot(label=True, legend=True)
for i in np.arange(0, pur_scaled.shape[0]):
    plt.scatter(components.values[])
    plt.text(components.values[i,0],
             components.values[i,1],
             list(pur_scaled.index)[i])
plt.show() 

