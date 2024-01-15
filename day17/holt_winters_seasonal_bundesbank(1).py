# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 11:10:12 2024

@author: dbda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Simple Exponential Smoothing for WGEM-IND_CPTOTNSXN dataset
df = pd.read_csv(r"F:\PML\Datasets\BUNDESBANK-BBK01_WT5511.csv")
df.head()
df.plot()
plt.show()


y = df['Value']
y_train = df['Value'][:-12]
y_test = df['Value'][-12:]

# Multiplicative
from statsmodels.tsa.api import ExponentialSmoothing
alpha = 0.3
beta = 0.02
gamma = 0.8
hw_mul = ExponentialSmoothing(y_train, seasonal_periods=12, trend='add', seasonal='mul')
fit1 = hw_mul.fit(smoothing_level=alpha, smoothing_trend=beta,smoothing_seasonal=gamma)
fcast1 = fit1.forecast(len(y_test))

#plot
y_train.plot(color='blue', label='Train')
y_test.plot(color='pink',label='Test')
fcast1.plot(color='purple',label='Forecast')
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,fcast1)
sqrt = np.sqrt(mse)
error = round(sqrt,2)
plt.text(5,110, 'RMSE = ' + str(error))
plt.title("HW Additive Trend Multiplicative")
plt.legend(loc='best')
plt.show()


# Additive
from statsmodels.tsa.api import ExponentialSmoothing
alpha = 0.3
beta = 0.02
gamma = 0.8 
hw_mul = ExponentialSmoothing(y_train, seasonal_periods=12, trend='add', seasonal='add')
fit1 = hw_mul.fit(smoothing_level=alpha, smoothing_trend=beta,smoothing_seasonal=gamma)
fcast1 = fit1.forecast(len(y_test))

#plot
y_train.plot(color='blue', label='Train')
y_test.plot(color='pink',label='Test')
fcast1.plot(color='purple',label='Forecast')
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,fcast1)
sqrt = np.sqrt(mse)
error = round(sqrt,2)
plt.text(5,110, 'RMSE = ' + str(error))
plt.title("HW Additive Trend Additive")
plt.legend(loc='best')
plt.show()

# Damping trend with additive 
from statsmodels.tsa.api import ExponentialSmoothing
alpha = 0.3
beta = 0.02
gamma = 0.8 
phi = 0.4
hw_mul = ExponentialSmoothing(y_train, seasonal_periods=12, trend='add', seasonal='add',damped_trend=True)
fit1 = hw_mul.fit(smoothing_level=alpha, smoothing_trend=beta,smoothing_seasonal=gamma,damping_trend=phi)
fcast1 = fit1.forecast(len(y_test))

#plot
y_train.plot(color='blue', label='Train')
y_test.plot(color='pink',label='Test')
fcast1.plot(color='purple',label='Forecast')
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,fcast1)
sqrt = np.sqrt(mse)
error = round(sqrt,2)
plt.text(5,110, 'RMSE = ' + str(error))
plt.title("HW Additive Trend Additive")
plt.legend(loc='best')
plt.show()

