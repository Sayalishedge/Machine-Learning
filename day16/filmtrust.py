# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:46:35 2024

@author: dbda
"""

import surprise
import numpy as np
import pandas as pd
import os

ratings = pd.read_csv(r"C:\Users\dbda\Desktop\PML\recommender_systems\filmtrust\ratings.txt",
                     sep=' ', names=['uid','iid','rating'])
ratings.head()

lowest_rating = ratings['rating'].min()
highest_rating = ratings['rating'].max()


reader = surprise.Reader(rating_scale = (lowest_rating,highest_rating))
data = surprise.Dataset.load_from_df(ratings, reader)
type(data)


similarity_options = {'name' : 'cosine','user_based' : True}
#default k=40
algo = surprise.KNNBasic(sim_options=similarity_options)
output = algo.fit(data.build_full_trainset())


#total items
iids = ratings['iid'].unique()
print(iids)

#the list of items rated by user 50
u_iid = ratings[ratings['uid']==50]['iid'].unique()
print(u_iid)

