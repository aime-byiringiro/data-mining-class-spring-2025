# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(len(y), 1)).flatten() # Reshaping the y array to a 2D array





# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_scaled, y_scaled)

# Making a single prediction
pred_scaled = regressor.predict(sc_X.transform([[6.6]]))
print(sc_y.inverse_transform(pred_scaled.reshape(1, 1)))

# Visualizing the SVR results
plt.scatter(X,y, color = 'red')
y_pred_scaled = regressor.predict(X_scaled)
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(len(y_pred_scaled), 1))
plt.plot(X, y_pred, color = 'blue')
plt.title('SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# Visualizing the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X) + 0.1, 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1)) # Reshaping the X_grid to a 2D array
X_grid_scaled = sc_X.transform(X_grid)
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
