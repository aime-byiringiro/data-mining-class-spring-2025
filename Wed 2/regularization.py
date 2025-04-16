# Regularization

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],
                       remainder='passthrough')
X = ct.fit_transform(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Feature Scaling: scale your data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, 2:] = sc.fit_transform(X[:, 2:])
y = sc.fit_transform(y.reshape(len(y), 1)).flatten()

# Fitting Multiple Linear Regression with Regularization: use this after scaling 
# alpha is the regularization parameter
# we want 
from sklearn.linear_model import ElasticNet
regressor1 = ElasticNet(l1_ratio = 0.5, alpha = 0)
regressor1.fit(X, y)
print(regressor1.coef_)

regressor2 = ElasticNet(l1_ratio = 0.5, alpha = 1)
regressor2.fit(X, y)
print(regressor2.coef_)