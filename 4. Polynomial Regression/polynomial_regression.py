# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# we need this line so that we can keep track of our dataset and know dolums
X = dataset.iloc[:, 1:-1].to_numpy()

y = dataset.iloc[:, -1].to_numpy()

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_feature = PolynomialFeatures(degree = 2, include_bias = False)
# the 2 is indicating the polynomial degree
X_poly = poly_feature.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)

# Finding the optimal degree using p-value

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
sc = StandardScaler()
y = sc.fit_transform(y.reshape(len(y),1)).flatten()


poly_feature = PolynomialFeatures(degree = 2)
X_poly =  poly_feature.fit_transform(X)
X_poly[:, 1:] = sc.fit_transform(X_poly[:, 1:])
regressor = sm.OLS(endog = y , exog = X_poly).fit()
regressor.summary

poly_feature = PolynomialFeatures(degree = 3)
X_poly =  poly_feature.fit_transform(X)
X_poly[:, 1:] = sc.fit_transform(X_poly[:, 1:])
regressor = sm.OLS(endog = y , exog = X_poly).fit()
regressor.summary

poly_feature = PolynomialFeatures(degree = 4)
X_poly =  poly_feature.fit_transform(X)
X_poly[:, 1:] = sc.fit_transform(X_poly[:, 1:])
regressor = sm.OLS(endog = y , exog = X_poly).fit()
regressor.summary


'''
poly_feature = PolynomialFeatures(degree = 5)
X_poly =  poly_feature.fit_transform(X)
X_poly[:, 1:] = sc.fit_transform(X_pol[:, 1:])
regressor = sm.OLS(endog = y , exog = X_poly).fit()
regressor.summary
'''





#when you see a p value that is hight than .5


# Visualizing the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X_poly), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary (Scaled)')
plt.show()

# Visualizing the Polynomial Regression results (for higher resolution and
# smoother curve)
X_grid = np.arange(min(X), max(X) + 0.1, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
X_grid_poly = poly_feature.transform(X_grid)

X_grid_poly[:, 1:] = sc.transform(X_grid_poly[:, 1:])


plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid_poly),
         color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary (Scaled)')
plt.show()