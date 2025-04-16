


# Importing libraries
import numpy as np
import pandas as pd

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 22:00:56 2025

@author: aime
"""


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Importing dataset
dataset = pd.read_csv('Housing_Data.csv', skiprows=1)
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()
num_of_ind_vars = X.shape[1]





# =========================== SVR RUN 1 =========================================================

# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()
#y_train_scaled = sc.transform(y_test.reshape(len(y_test), 1)).flatn()
#y_test_scaled = sc.transform(y_train.reshape(len(y_train)).flatten())

# SVR with RBF Kernel
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)
y_pred_scaled_svr = svr.predict(X_test_scaled)
y_pred_svr = sc_y.inverse_transform(y_pred_scaled_svr.reshape(-1, 1)).flatten()

# Adjusted R-squared for SVR
r2_svr = r2_score(y_test, y_pred_svr)
r2_adjusted_svr_1 = 1 - (1 - r2_svr) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('SVR Adjusted R-squared 1 :', r2_adjusted_svr_1)



# =========================== SVR RUN 2 =========================================================

# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# SVR with RBF Kernel
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)
y_pred_scaled_svr = svr.predict(X_test_scaled)
y_pred_svr = sc_y.inverse_transform(y_pred_scaled_svr.reshape(-1, 1)).flatten()

# Adjusted R-squared for SVR
r2_svr = r2_score(y_test, y_pred_svr)
r2_adjusted_svr_2 = 1 - (1 - r2_svr) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('SVR Adjusted R-squared 2:', r2_adjusted_svr_2)

# =========================== SVR RUN 3 =========================================================

# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# SVR with RBF Kernel
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)
y_pred_scaled_svr = svr.predict(X_test_scaled)
y_pred_svr = sc_y.inverse_transform(y_pred_scaled_svr.reshape(-1, 1)).flatten()

# Adjusted R-squared for SVR
r2_svr = r2_score(y_test, y_pred_svr)
r2_adjusted_svr_3 = 1 - (1 - r2_svr) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('SVR Adjusted R-squared 3 :', r2_adjusted_svr_3)



# =========================== SVR RUN 4 =========================================================

# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# SVR with RBF Kernel
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)
y_pred_scaled_svr = svr.predict(X_test_scaled)
y_pred_svr = sc_y.inverse_transform(y_pred_scaled_svr.reshape(-1, 1)).flatten()

# Adjusted R-squared for SVR
r2_svr = r2_score(y_test, y_pred_svr)
r2_adjusted_svr_4 = 1 - (1 - r2_svr) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('SVR Adjusted R-squared 4 :', r2_adjusted_svr_4)


# =========================== SVR RUN 5 =========================================================

# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# SVR with RBF Kernel
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)
y_pred_scaled_svr = svr.predict(X_test_scaled)
y_pred_svr = sc_y.inverse_transform(y_pred_scaled_svr.reshape(-1, 1)).flatten()

# Adjusted R-squared for SVR
r2_svr = r2_score(y_test, y_pred_svr)
r2_adjusted_svr_5 = 1 - (1 - r2_svr) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('SVR Adjusted R-squared 5 :', r2_adjusted_svr_5)



# =========================== SVR RUN 6 =========================================================

# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# SVR with RBF Kernel
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)
y_pred_scaled_svr = svr.predict(X_test_scaled)
y_pred_svr = sc_y.inverse_transform(y_pred_scaled_svr.reshape(-1, 1)).flatten()

# Adjusted R-squared for SVR
r2_svr = r2_score(y_test, y_pred_svr)
r2_adjusted_svr_6 = 1 - (1 - r2_svr) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('SVR Adjusted R-squared 6 :', r2_adjusted_svr_6)

# =========================== SVR RUN 7 =========================================================

# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# SVR with RBF Kernel
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)
y_pred_scaled_svr = svr.predict(X_test_scaled)
y_pred_svr = sc_y.inverse_transform(y_pred_scaled_svr.reshape(-1, 1)).flatten()

# Adjusted R-squared for SVR
r2_svr = r2_score(y_test, y_pred_svr)
r2_adjusted_svr_7 = 1 - (1 - r2_svr) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('SVR Adjusted R-squared 7 :', r2_adjusted_svr_7)



# =========================== SVR RUN 8 =========================================================

# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# SVR with RBF Kernel
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)
y_pred_scaled_svr = svr.predict(X_test_scaled)
y_pred_svr = sc_y.inverse_transform(y_pred_scaled_svr.reshape(-1, 1)).flatten()

# Adjusted R-squared for SVR
r2_svr = r2_score(y_test, y_pred_svr)
r2_adjusted_svr_8 = 1 - (1 - r2_svr) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('SVR Adjusted R-squared 8 :', r2_adjusted_svr_8)


# =========================== SVR RUN 9 =========================================================

# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# SVR with RBF Kernel
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)
y_pred_scaled_svr = svr.predict(X_test_scaled)
y_pred_svr = sc_y.inverse_transform(y_pred_scaled_svr.reshape(-1, 1)).flatten()

# Adjusted R-squared for SVR
r2_svr = r2_score(y_test, y_pred_svr)
r2_adjusted_svr_9 = 1 - (1 - r2_svr) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('SVR Adjusted R-squared 9 :', r2_adjusted_svr_9)



# =========================== SVR RUN 10 =========================================================

# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()

# SVR with RBF Kernel
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train_scaled)
y_pred_scaled_svr = svr.predict(X_test_scaled)
y_pred_svr = sc_y.inverse_transform(y_pred_scaled_svr.reshape(-1, 1)).flatten()

# Adjusted R-squared for SVR
r2_svr = r2_score(y_test, y_pred_svr)
r2_adjusted_svr_10 = 1 - (1 - r2_svr) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('SVR Adjusted R-squared 10 :', r2_adjusted_svr_10)



#SVR Adjusted R-squared 1 : 0.7089142896145233
#SVR Adjusted R-squared 2: 0.5697465581480382
#SVR Adjusted R-squared 3 : 0.6349757523164654
#SVR Adjusted R-squared 4 : 0.7323732817385684
#SVR Adjusted R-squared 5 : 0.5597191016109845
#SVR Adjusted R-squared 6 : 0.7232449215253682
#SVR Adjusted R-squared 7 : 0.6101588894417513
#SVR Adjusted R-squared 8 : 0.5790797136004326
#SVR Adjusted R-squared 9 : 0.5833548927263611
#SVR Adjusted R-squared 10 : 0.5522191316309617


# average of all adjusted R_squared 

average_adjusted_svr = (
    r2_adjusted_svr_1 +
    r2_adjusted_svr_2 + 
    r2_adjusted_svr_3 +
    r2_adjusted_svr_4 +
    r2_adjusted_svr_5 +
    r2_adjusted_svr_6 +
    r2_adjusted_svr_7 +
    r2_adjusted_svr_8 +
    r2_adjusted_svr_9 +
    r2_adjusted_svr_10 
    
    )/10

print("Average of adjusted R2 for SVR over 10 runs is : ", average_adjusted_svr)






# =========================== Random Forest RUN 1 =========================================================


# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()


# Random Forest Regression with 500 Trees
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Adjusted R-squared for Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
r2_adjusted_rf_1 = 1 - (1 - r2_rf) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('Random Forest Adjusted R-squared_1:', r2_adjusted_rf_1)



# =========================== Random Forest RUN 2 =========================================================


# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()


# Random Forest Regression with 500 Trees
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Adjusted R-squared for Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
r2_adjusted_rf_2 = 1 - (1 - r2_rf) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('Random Forest Adjusted R-squared_2:', r2_adjusted_rf_2)


# =========================== Random Forest RUN 3 =========================================================


# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()


# Random Forest Regression with 500 Trees
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Adjusted R-squared for Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
r2_adjusted_rf_3 = 1 - (1 - r2_rf) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('Random Forest Adjusted R-squared_3:', r2_adjusted_rf_3)



# =========================== Random Forest RUN 4 =========================================================


# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()


# Random Forest Regression with 500 Trees
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Adjusted R-squared for Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
r2_adjusted_rf_4 = 1 - (1 - r2_rf) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('Random Forest Adjusted R-squared_4:', r2_adjusted_rf_4)

# =========================== Random Forest RUN 5 =========================================================


# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()


# Random Forest Regression with 500 Trees
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Adjusted R-squared for Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
r2_adjusted_rf_5 = 1 - (1 - r2_rf) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('Random Forest Adjusted R-squared_5:', r2_adjusted_rf_5)



# =========================== Random Forest RUN 6 =========================================================


# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()


# Random Forest Regression with 500 Trees
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Adjusted R-squared for Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
r2_adjusted_rf_6 = 1 - (1 - r2_rf) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('Random Forest Adjusted R-squared_6:', r2_adjusted_rf_6)


# =========================== Random Forest RUN 7 =========================================================


# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()


# Random Forest Regression with 500 Trees
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Adjusted R-squared for Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
r2_adjusted_rf_7 = 1 - (1 - r2_rf) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('Random Forest Adjusted R-squared_7:', r2_adjusted_rf_7)



# =========================== Random Forest RUN 8 =========================================================


# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()


# Random Forest Regression with 500 Trees
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Adjusted R-squared for Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
r2_adjusted_rf_8 = 1 - (1 - r2_rf) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('Random Forest Adjusted R-squared_8:', r2_adjusted_rf_8)


# =========================== Random Forest RUN 9 =========================================================


# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()


# Random Forest Regression with 500 Trees
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Adjusted R-squared for Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
r2_adjusted_rf_9 = 1 - (1 - r2_rf) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('Random Forest Adjusted R-squared_9:', r2_adjusted_rf_9)



# =========================== Random Forest RUN 10 =========================================================


# Splitting dataset into Training and Test set (25% Test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).flatten()


# Random Forest Regression with 500 Trees
rf = RandomForestRegressor(n_estimators=500)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# Adjusted R-squared for Random Forest
r2_rf = r2_score(y_test, y_pred_rf)
r2_adjusted_rf_10 = 1 - (1 - r2_rf) * (len(y_test) - 1) / (len(y_test) - num_of_ind_vars - 1)
print('Random Forest Adjusted R-squared_10:', r2_adjusted_rf_10)

average_adjusted_rf = (
    r2_adjusted_rf_1 +
    r2_adjusted_rf_2 + 
    r2_adjusted_rf_3 +
    r2_adjusted_rf_4 +
    r2_adjusted_rf_5 +
    r2_adjusted_rf_6 +
    r2_adjusted_rf_7 +
    r2_adjusted_rf_8 +
    r2_adjusted_rf_9 +
    r2_adjusted_rf_10 
    
    )/10

print("Average of adjusted R2 for Rf over 10 runs is : ", average_adjusted_rf)



# Model Comparison
if average_adjusted_rf > average_adjusted_svr:
    print("Random Forest model performs better than SVR.")
elif average_adjusted_rf < average_adjusted_svr:
    print("SVR model performs better than Random Forest.")
else:
    print("Both models have similar performance.")