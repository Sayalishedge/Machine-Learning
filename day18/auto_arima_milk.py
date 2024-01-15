# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:13:05 2024

@author: dbda
"""

#refer sir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error as mse
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima

df = pd.read_csv("F:\PML\Datasets\monthly-milk-production-pounds-p.csv")
plot_acf(df['Milk'], lags=7)
plt.show()

y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]

model = auto_arima(y_train, trace=True,
                   error_action='ignore',
                   suppress_warnings=True)
#Best model:  ARIMA(1,1,4)(0,0,0)[0] intercept
#Total fit time: 3.634 seconds

forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast, index=y_test.index,columns=['Prediction'])

#plot
plt.plot(y_train, label='Train',color='blue')
plt.plot(y_test, label='Valid', color='pink')
plt.plot(forecast, label='Prediction', color='purple')
error = round(sqrt(mse(y_test, forecast)),2)
plt.text(100,600, "RMSE="+str(error))
plt.legend(loc='best')
plt.show()


#################################################################

wgem = pd.read_csv("F:\PML\Datasets\WGEM-IND_CPTOTNSXN.csv")
plot_acf(wgem['Value'], lags=7)
plt.show()

y_train = wgem['Value'][:-4]
y_test = wgem['Value'][-4:]

model = auto_arima(y_train, trace=True,
                   error_action='ignore',
                   suppress_warnings=True)
#Best model:  ARIMA(0,2,0)(0,0,0)[0] intercept
#Total fit time: 0.241 seconds


forecast = model.predict(n_periods=len(y_test))
forecast = pd.DataFrame(forecast, index=y_test.index,columns=['Prediction'])

#plot
plt.plot(y_train, label='Train',color='blue')
plt.plot(y_test, label='Valid', color='pink')
plt.plot(forecast, label='Prediction', color='purple')
error = round(sqrt(mse(y_test, forecast)),2)
plt.text(100,600, "RMSE="+str(error))
plt.legend(loc='best')
plt.show()










