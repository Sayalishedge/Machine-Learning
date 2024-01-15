# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 15:52:44 2024

@author: dbda
"""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv(r"F:\PML\Datasets\monthly-milk-production-pounds-p.csv")
df.head()
df.plot()
plt.show()

series = df['Milk']
result = seasonal_decompose(series, model='additive',period=12)
result.plot()
plt.title('Additive Decomposition')
plt.show()

result = seasonal_decompose(series, model='multiplicative', period=12)
result.plot()
plt.title("Multiplicative Decomposition")
plt.show()

##################################################################
# Air Passengers 
air = pd.read_csv(r"F:\PML\Datasets\AirPassengers.csv")
air.head()
air.plot()
plt.show()

series = air['']