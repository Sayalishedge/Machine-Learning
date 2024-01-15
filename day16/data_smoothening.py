# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:12:26 2024

@author: dbda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv(r"F:\PML\Datasets\monthly-milk-production-pounds-p.csv")
df.head()
df.plot()
plt.show()

y = df['Milk']

## Centered Moving Average

fcast= y.rolling(3,center=True).mean()
plt.plot(y, label='Data')
plt.plot(fcast, label='Centered Moving Average')
plt.legend(loc='best')
plt.show()

#########################################################
air = pd.read_csv(r"F:\PML\Datasets\AirPassengers.csv")
air.head()
air.plot()
plt.show()
y = air['Passengers']

## Centered Moving Average
fcast= y.rolling(3,center=True).mean()
plt.plot(y, label='Data')
plt.plot(fcast, label='Centered Moving Average')
plt.legend(loc='best')
plt.show()
#########################################################

# Trailing Moving Average
y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]

###########################################################

# Simple Exponential Smoothing for milk dataset
from statsmodels.tsa.api import SimpleExpSmoothing
y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]
alpha = 0.9

ses = SimpleExpSmoothing(y_train)
fit1 = ses.fit(smoothing_level=alpha)
fcast1 = fit1.forecast(len(y_test))
#plot
y_train.plot(color='blue',label='Train')
y_test.plot(color='pink',label='Test')
fcast1.plot(color='purple',label='Forecast')
plt.legend(loc='best')
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,fcast1)
sqrt = np.sqrt(mse)

print(f"RMSE for alpha = {alpha} is {sqrt}")
#RMSE for alpha = 0.1 is 57.02407775273818
#RMSE for alpha = 0.5 is 76.91666495915985
#RMSE for alpha = 0.9 is 75.97063314998215

###########################################################

# Simple Exponential Smoothing for WGEM-IND_CPTOTNSXN dataset
df = pd.read_csv(r"F:\PML\Datasets\WGEM-IND_CPTOTNSXN.csv")
df.head()
df.plot()
plt.show()

y = df['Value']

## Centered Moving Average

fcast= y.rolling(3,center=True).mean()
plt.plot(y, label='Data')
plt.plot(fcast, label='Centered Moving Average')
plt.legend(loc='best')
plt.show()


from statsmodels.tsa.api import SimpleExpSmoothing
y_train = df['Value'][:-4]
y_test = df['Value'][-4:]
alpha = 0.5

ses = SimpleExpSmoothing(y_train)
fit1 = ses.fit(smoothing_level=alpha)
fcast1 = fit1.forecast(len(y_test))
#plot
y_train.plot(color='blue',label='Train')
y_test.plot(color='pink',label='Test')
fcast1.plot(color='purple',label='Forecast')
plt.legend(loc='best')
plt.show()

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,fcast1)
sqrt = np.sqrt(mse)

print(f"RMSE for alpha = {alpha} is {sqrt}")

#RMSE for alpha = 0.9 is 21.443641537603213
#RMSE for alpha = 0.1 is 72.70753441659585
#RMSE for alpha = 0.5 is 30.457149712648373







