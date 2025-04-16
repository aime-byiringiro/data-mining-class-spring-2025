# Data Preprocessing Tools

# 1. Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 2. Importing the dataset
dataset = pd.read_csv('Data.csv')

# iloc[rows, cols]    
# starting point inclusive : ending point exclusive, slicing
# colon by itself means all 
# independent variables:
# dependent variables:
# In this case, X includes all rows and every column aside from the last
X = dataset.iloc[:, :-1].to_numpy() # Capital for matrix, independent variables
# Here, Y includes every row, but only the last column
y = dataset.iloc[:, -1].to_numpy()  # lowercase for vector, dependent variables

# This line will include columns 1, 3, 5-8, and 9 and every row
# dataset.iloc[:, np.r_[1, 3, 5-8, 9]].to_numpy

# 3. Taking care of missing data
# there's a fill ways to deal with missing data
# if there is a lot of data, just remove it
# else use avg or median to fill missing data
# In this case, we use the avg (most common)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# fit: calculating the object, does not return. It involves calculating needed
# parameters and saving them within the object.
imputer.fit(X[:, 1:3])
# transform: Execute the object, and return processed data. It involves 
# calculating new data based on the existing data.
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Does the above fit + transform in one line
# X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# 4. Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])],
                       remainder = 'passthrough')


X = ct.fit_transform(X)

# Encoding the Dependent Variable  (inted for dependent variable only)

from sklearn.preprocessing import LabelEnoder
le = LabelEnoder()
y = le.fit_transform(y)


# 5. Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state =0)
# 6. Used to test performance of model? 


# 7. Sets are chosen randomly to prevent bias


# 8. Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler

# from sklearn.preprocessing import MinMaxScalar
# sc = MinMaxScalar

X_train[:,3:] = sc.fit_transform(X_train[:,3.5:])

# X_test[:, 3:] = sc.fit_transform(X_test[:, 3])


