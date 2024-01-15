# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:25:27 2024

@author: dbda
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA

glass = pd.read_csv(r"F:\PML\Cases\Glass_Identification\Glass.csv")
glass.columns

sc = StandardScaler().set_output(transform='pandas')
g_scaled = sc.fit_transform(glass.iloc[:,:-1])

prcomp = PCA().set_output(transform='pandas')
components = prcomp.fit_transform(g_scaled)

components['Type'] = glass['Type']

sns.scatterplot(data=components,
                x='pca0', y='pca1',
                hue='Type')
plt.legend(loc='best')
plt.show()