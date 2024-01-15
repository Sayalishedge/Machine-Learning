# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:17:18 2024

@author: dbda
"""
import numpy as np
x = np.array([0.2,1.2,1,1.4,-1.5,0.5,-0.5])
y = np.array([5.6,8.6,8,9.2,0.5,6.5,3.5])
eta = 0.2
i_w = 0.5
i_b = -0.5

w,b = i_w,i_b
count = 0
for i in range(70):
    y_pred = w*x + b
    L = (1/2*len(x))*np.sum((y-y_pred)**2)
    print("Loss = ",L)
    
    if L < 0.0001:
        break;
        
    db = -(1/len(x))*np.sum(y-y_pred) 
    dw = -(1/len(x))*np.sum((y-y_pred)*x)
    
    new_w = w - eta*dw
    new_b = b - eta*db
    count+=1
    print(f"For iteration {count} : b = {new_b}, w={new_w}")
    w = new_w
    b = new_b
    
##########################################################
import pandas as pd
pizza = pd.read_csv(r"F:\PML\Datasets\pizza.csv")

x = pizza['Promote']
y = pizza['Sales']
eta = 0.2
i_w = 0.5
i_b = -0.5
w,b = i_w,i_b
count = 0
for i in range(500):
    y_pred = w*x + b
    L = (1/2*len(x))*np.sum((y-y_pred)**2)
    print("Loss = ",L)
    
    if L < 0.0001:
        break;
        
    db = -(1/len(x))*np.sum(y-y_pred) 
    dw = -(1/len(x))*np.sum((y-y_pred)*x)
    
    new_w = w - eta*dw
    new_b = b - eta*db
    count+=1
    print(f"For iteration {count} : b = {new_b}, w={new_w}")
    w = new_w
    b = new_b    

#######################################################################

#gradient descent works good with scaled data 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

pizza = pd.read_csv(r"F:\PML\Datasets\pizza.csv")
sc = MinMaxScaler().set_output(transform='pandas')

pizza_scaled = sc.fit_transform(pizza)
x = pizza_scaled['Promote']
y = pizza_scaled['Sales']
eta = 0.5
i_w = 0.5
i_b = -0.5

losses = []
epochs = []
w,b = i_w,i_b
count = 0
for i in range(100):
    y_pred = w*x + b
    L = (1/2*len(x))*np.sum((y-y_pred)**2)
    losses.append(L)
    epochs.append(i)
    print("Loss = ",L)
    
    if L < 0.0001:
        break;
        
    db = -(1/len(x))*np.sum(y-y_pred) 
    dw = -(1/len(x))*np.sum((y-y_pred)*x)
    
    new_w = w - eta*dw
    new_b = b - eta*db
    count+=1
    print(f"For iteration {count} : b = {new_b}, w={new_w}")
    w = new_w
    b = new_b   

losses

import matplotlib.pyplot as plt

plt.scatter(epochs,losses, c='red')
plt.plot(epochs,losses,c='black')
plt.title(f"Learning curve with eta = {eta}")
plt.show() 

############################################################

























