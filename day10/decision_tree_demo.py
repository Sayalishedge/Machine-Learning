# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:05:37 2024

@author: dbda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

x = np.array([[2,4],
              [3,5],
              [12,18],
              [15,20],
              [34,56],
              [35,60],
              [78, 26],
              [80, 23]])
y = np.array(["1","1","0","0","1","1","0","0"])

p_df = pd.DataFrame(x, columns=['x1', 'x2'])
p_df['y'] = y

sns.scatterplot(data=p_df, x='x1',y='x2',
                hue='y')
plt.show()

X = p_df[['x1','x2']]
y = p_df['y']

dtc = DecisionTreeClassifier(random_state=23, max_depth=3)
dtc.fit(X, y)

plt.figure(figsize=(15,10))
plot_tree(dtc,feature_names=list(X.columns),
               class_names=['0','1'],
               filled=True,fontsize=18)
plt.show() 










