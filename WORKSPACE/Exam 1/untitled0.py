#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 16:17:51 2025

@author: aime
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Insurance_Data.csv')
X = dataset.iloc[:, :-1].to_numpy[]
y = dataset.iloc[:, -1].

######
###
#####




#######################

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X, y)


###Visualizing mulitple linear regeressor modle
plt.scatter(X[:, 5], y, color = 'red')
plt.scatter(X[:, 5], regressor1.predict(X), color='blue')
plt.title("Mulitiple")
plt.xlabel('Age')
plt.ylabel('Charges')
plot.show()


from sklearn.preprocessing import  PolynomialFeatures

poly_feature = PolynomialFeatures(degree = 2, include_bias = False)
age_poly = poly_feature.fit_transfrom(X[:, 5:6])
X =  np.apppend(X, age_poly[:, 1:2], axis = 1)



bmi30 = np.where(X[:, 6:7] >= 30, 1,0)
X = np.append(X, bmi30, axis=1)



bmi30smoker = bmi30 * X[:, 1:2]
X = np.append(X, bmi30smoker, axis = 1)


regressor2 = LinearRegression()
regressor2.fit(X,y)


plt.scatter(X[:, 5], y, color = 'red')
plt.scatter(X[:, 5], regressor1.predict(X), color='blue')
plt.title("Mulitiple")
plt.xlabel('Age')
plt.ylabel('Charges')
plot.show()



######step 4

from sklearn.preprocessing import  StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()


X[:, 5:9] = sc_X.fit_transform(X[:, 5:9])
y = sc_y.fit_transform(y.reshape(len(y),1)).flatten()



import statsmodels.api as sm

X = sm.add_constant(X).astype(np, float64)
X_opt = X[:, ]



###step5


X = X[:, 1:]


from sklearn.ssvm import SVR


regressor4 = SVR(kernel = 'rbf')

regressor4.fit
                              
    




