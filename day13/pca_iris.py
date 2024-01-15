# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:13:02 2024

@author: dbda
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA

iris = pd.read_csv(r"C:\Users\dbda\Desktop\PML\Datasets\iris.csv")


sc = StandardScaler().set_output(transform='pandas')
i_scaled = sc.fit_transform(iris.iloc[:,:-1])

prcomp = PCA().set_output(transform='pandas')
components = prcomp.fit_transform(i_scaled)
components['Species'] = iris['Species']

sns.scatterplot(data=components,
                x='pca0', y='pca1',
                hue='Species')
plt.legend(loc='best')
plt.show()



