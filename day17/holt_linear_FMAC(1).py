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
df = pd.read_csv(r"F:\PML\Datasets\FMAC-HPI_24420.csv")
df.head()
df.plot()
plt.show()

y = df['NSA Value']

## Centered Moving Average

fcast= y.rolling(3,center=True).mean()
plt.plot(y, label='Data')
plt.plot(fcast, label='Centered Moving Average')
plt.legend(loc='best')
plt.show()


from statsmodels.tsa.api import SimpleExpSmoothing
y_train = df['NSA Value'][:-8]
y_test = df['NSA Value'][-8:]
alpha = 0.1

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

#RMSE for alpha = 0.9 is 3.644506577637956
#RMSE for alpha = 0.1 is 10.78337359064945
#RMSE for alpha = 0.5 is 2.570266724479979

###########################################################
# Holt Linear Trend Model
alpha = 0.02
beta = 0.22

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


#RMSE for beta=0.02 and alpha=0.22 is 1.77

scores = []
for alpha in np.linspace(0,1,100):
    for beta in np.linspace(0,1,100):
        holt = Holt(y_train)
        fit1 = holt.fit(smoothing_level=alpha,smoothing_trend=beta)
        fcast1 = fit1.forecast(len(y_test))
        mse = mean_squared_error(y_test,fcast1)
        sqrt = np.sqrt(mse)
        error = round(sqrt,2)
        sc = error
        scores.append([alpha,beta,sc])

pd_scores = pd.DataFrame(scores, columns=['alpha','beta','rmse'])
pd_scores.sort_values('rmse', ascending=True)
#######################################################################


# Exponential Linear Trend Model
alpha = 0.02
beta = 0.22

from statsmodels.tsa.api import Holt
holt = Holt(y_train,exponential=True)
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


#######################################################################

# Exponential Linear Trend Model
alpha = 0.02
beta = 0.22

from statsmodels.tsa.api import Holt
holt = Holt(y_train,exponential=True)
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























