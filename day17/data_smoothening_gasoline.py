# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 08:13:54 2024

@author: dbda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv(r"F:\PML\Datasets\Gasoline.csv")
df.head()
df.plot()
plt.show()


y = df['Sales']

## Centered Moving Average

fcast= y.rolling(3,center=True).mean()
plt.plot(y, label='Data')
plt.plot(fcast, label='Centered Moving Average')
plt.legend(loc='best')
plt.show()


from statsmodels.tsa.api import SimpleExpSmoothing
y_train = df['Sales'][:-4]
y_test = df['Sales'][-4:]
alpha = 0.4

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
    

#RMSE for alpha = 0.1 is 3.309926883141581
#RMSE for alpha = 0.2 is 3.1278440975425594
#RMSE for alpha = 0.3 is 3.098928864327261  #best
#RMSE for alpha = 0.4 is 3.1183909377378836
#RMSE for alpha = 0.5 is 3.148040064865757