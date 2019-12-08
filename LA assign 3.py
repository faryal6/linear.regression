# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:39:18 2019

@author: HOME
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('aids.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'purple')
plt.plot(X_train, regressor.predict(X_train), color = 'orange')
plt.title('Deaths vs Years (Training set)')
plt.xlabel('Years')
plt.ylabel('Deaths')
plt.show()

plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'orange')
plt.title('Deaths vs Years (Test set)')
plt.xlabel('Years')
plt.ylabel('Deaths')
plt.show()

print(regressor.predict([[2000]]))