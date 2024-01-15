# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 08:13:46 2024

@author: dbda
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error as mse
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error

df = pd.read_csv("F:\PML\Datasets\monthly-milk-production-pounds-p.csv")
plot_acf(df['Milk'], lags=7)
plt.show()

wgem = pd.read_csv("F:\PML\Datasets\WGEM-IND_CPTOTNSXN.csv")
plot_acf(wgem['Value'], lags=7)
plt.show()

y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]
###### AutoRegressive Models #############
from statsmodels.tsa.arima.model import ARIMA
# train MA
model = ARIMA(y_train,order=(1,0,0))
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
# make predictions
fcast1 = model_fit.predict(start=len(y_train), 
                           end=len(y_train)+len(y_test)-1, 
                           dynamic=False)
error = round(sqrt(mse(y_test, fcast1)),2)
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600, "RMSE="+str(error))
plt.legend(loc='best')
plt.show()


scores = []
ar = [1,2,3,4,5,6]
for a in ar:
    model = ARIMA(y_train,order=(a,0,0))
    model_fit = model.fit()
    fcast1 = model_fit.predict(start=len(y_train), 
                               end=len(y_train)+len(y_test)-1, 
                               dynamic=False)
    mse = mean_squared_error(y_test,fcast1)
    sqrt = np.sqrt(mse)
    scores.append(round(sqrt,2))

i_min = np.argmin(scores)
print("Best AR order = ",ar[i_min])
print("Best Scorer = ",scores[i_min])

############################## A-Dicky Fuller Test ################################

from statsmodels.tsa.stattools import adfuller
y = df['Milk']
ad_result = adfuller(y,autolag=None)

p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")


d_y = y - y.shift(1)    
ad_result = adfuller(d_y.iloc[1:],autolag=None)
p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")
    
#1st order is Stationary    

############################## A-Dicky Fuller Test ################################


from statsmodels.tsa.stattools import adfuller
y = wgem['Value']
ad_result = adfuller(y,autolag=None)

p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")

#1st order diff
d_y = y - y.shift(1)    
ad_result = adfuller(d_y.iloc[1:],autolag=None)
p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")
    

#2nd order diff
d_d_y = d_y - d_y.shift(1)    
ad_result = adfuller(d_d_y.iloc[2:],autolag=None)
p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")
#2nd order is stationary    

############################## A-Dicky Fuller Test ################################

gasoline = pd.read_csv(r"F:\PML\Datasets\Gasoline.csv")
from statsmodels.tsa.stattools import adfuller
y = gasoline['Sales']
ad_result = adfuller(y,autolag=None)

p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")


d_y = y - y.shift(1)    
ad_result = adfuller(d_y.iloc[1:],autolag=None)
p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")

#1st order is Stationary
 






