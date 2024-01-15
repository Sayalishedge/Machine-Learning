# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:59:16 2024

@author: dbda
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:33:19 2024

@author: dbda
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

df = np.array([[4,5,1,2],
               [2,4,4,5],
               [1,3,5,5],
               [4,4,2,1]])



column_name = ['User1', 'User2', 'User3', 'User4']
df = pd.DataFrame(df, columns=column_name,index=['u1','u2','u3','u4'])
print(df)

df = df.T
print(df.corr())

#cosine similarity
print(cosine_similarity(df.T))
#########################################################
#should item1 be recommended to user3 -> no

def cosine(a,b) : 
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return np.dot(a,b)/(norm_a*norm_b)


user3_1 = cosine([3,5,5,2],[5,1,2,5])
user3_2 = cosine([4,5,1],[3,5,2])
user3_4 = cosine([5,5,2],[2,1,4])
user3_5 = cosine([3,5,2],[2,4,2])

print(user3_1)
print(user3_2)
print(user3_4)
print(user3_5)

X = df[['u1','u2','u3']]
y = df['u3']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)




















