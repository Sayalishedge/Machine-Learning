# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 08:32:13 2024

@author: dbda
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

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

###########################################################
# Holt Linear Trend Model
alpha = 0.5
beta = 0.3

from statsmodels.tsa.api import Holt
holt = Holt(y_train)
fit1 = holt.fit(smoothing_level=alpha,smoothing_trend=beta)
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
plt.title("Holt's Linear Trend")
plt.legend(loc='best')
plt.show()

#RMSE for beta=0.3 and alpha=0.5 is 3.25 

























