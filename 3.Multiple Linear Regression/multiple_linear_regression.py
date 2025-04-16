# Multiple Linear Regression

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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 2:] = sc.fit_transform(X_train[:, 2:])
y_train = sc.fit_transform(y_train.reshape(len(y_train), 1)).flatten()

# Backward Elimination
import statsmodels.api as sm 
X_train = sm.add_constant(X_train).astype(np.float64)
X_opt = X.train[:, [0,1,2,3,4,5]]
regressor_opt = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_opt.summary()
X_opt = X_train[:, [0,3,4,5]]
regressor_opt = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_opt.summary()
X_opt = X_train [:,[0,3,5]]
regressor_opt = sm.OLS(endog = y_train, exog =  X_opt).fit()
regressor_opt.summary()
#1. Never remove intercep becuase if we remove it we force the model to go throught the orign 
#2. Under Categorical colum 
#3. 