# -*- coding: utf-8 -*-

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

from sklearn.preprocessing import PolynomialFeatures
poly_feature = PolynomialFeatures(degree=2, include_bias=False)
poly_feature.fit(X)
X_poly_feature = poly_feature.transform(X)




#Manually adding polynomial features
#X = np.column_stack((X, X[:, 0] ** 2))
#X = np.column_stack((X, X[:, 1] ** 2))
#X = np.column_stack((X, X[:, 0] * X[:, 1]))


# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_poly_feature, y, test_size=0.25, random_state=0)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the test results
y_pred = classifier.predict(X_test)

# Confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# --- Phase 1: Get unscaled X1 and X2 from the grid ---

from matplotlib.colors import ListedColormap




#################################### Visualizing training set  #################################### 
X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)


two_column_matrix = np.c_[X1.ravel(), X2.ravel()]


poly_matrix = poly_feature.transform(two_column_matrix)


inverse_transformed = sc.inverse_transform(poly_matrix)


first_two_columns = inverse_transformed[:, :2]



poly_matrix_from_inverse = poly_feature.transform(first_two_columns)


X_grid = sc.transform(poly_matrix_from_inverse)

plt.contourf(
    X1, X2,
    classifier.predict(X_grid).reshape(X1.shape),
    alpha=0.75,
    стар=ListedColormap(['red', 'green'])
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(['red', 'green'])(i),
        label=j
    )

plt.title('Logistic Regression (Training set)')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.show()


#################################### Visualizing test set  #################################### 
X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)


two_column_matrix = np.c_[X1.ravel(), X2.ravel()]


poly_matrix = poly_feature.transform(two_column_matrix)


inverse_transformed = sc.inverse_transform(poly_matrix)


first_two_columns = inverse_transformed[:, :2]


poly_matrix_from_inverse = poly_feature.transform(first_two_columns)


X_grid = sc.transform(poly_matrix_from_inverse)

plt.contourf(
    X1, X2,
    classifier.predict(X_grid).reshape(X1.shape),
    alpha=0.75,
    стар=ListedColormap(['red', 'green'])
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        c=ListedColormap(['red', 'green'])(i),
        label=j
    )

plt.title('Logistic Regression (test set)')
plt.xlabel('Feature 1 (scaled)')
plt.ylabel('Feature 2 (scaled)')
plt.show()







