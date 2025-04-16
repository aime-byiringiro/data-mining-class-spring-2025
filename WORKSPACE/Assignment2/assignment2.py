
# Simple Linear Regression homework
# Professor : Dr. Mei
# Student :  Aime Byiringiro

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Dealership_Data.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

#splitting the dataset into the Training set and Test set 25% of the data is for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the Test set results
y_test = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Predicting the Car Price (Training set)')
plt.xlabel('Price')
plt.ylabel('Sell Price')

plt.show()



# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title('Predicting the Car Price (Test set)')
plt.xlabel('Price')
plt.ylabel('Sell Price')

# Visualising the Test set results
plt.show()


# print out the slop and the intercept of the line
print(regressor.coef_)
print(regressor.intercept_)


#printing the predicted value of the car price for the price of 20
print(regressor.predict([[20]]))



















