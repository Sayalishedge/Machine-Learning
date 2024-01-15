# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:27:04 2024

@author: dbda
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()

tiny = pd.read_csv(r"F:\PML\Datasets\tinydata.csv")
print(tiny)
tiny.info()
X = tiny[['Salt','Fat']]
y = le.fit_transform(tiny['Acceptance'])

mlp = MLPClassifier(random_state=23,hidden_layer_sizes=(3,))
mlp.fit(X,y)
print(mlp.coefs_)
print(mlp.intercepts_)












