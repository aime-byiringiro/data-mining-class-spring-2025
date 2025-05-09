#Author Aime Byiringiro

# We are  trying to predict if people in specific countries at certain age
# will or will not purchase 
# This is a classification problem since the dependent variable is categorical. 



#1. Importing libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler




#inputing dataset from Customer_Data.csv
dataset = pd.read_csv('Customer_Data.csv')


#Country, Age, and Salary will be independent(X)  and the last, Purchased, will
# be dependent y
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()


#Missing ages will be replacec by the mendian of the ages
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X[:,1:2])
X[:,1:2] = imputer.transform(X[:,1:2])


# Missing salaries will be replaced the mean of salaries 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,2:3])
X[:,2:3] = imputer.transform(X[:,2:3])


# Since country is texture, we need to enncoder it to numerical
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], 
                       remainder='passthrough')
X = ct.fit_transform(X) 


#Encoding the last column of Purchased
le = LabelEncoder()
y = le.fit_transform(y)


# Feature scaling using normarization 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
sc =  MinMaxScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_test[:,3:])















