# -*- coding: utf-8 -*-

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

# Manually adding polynomial features
X = np.column_stack((X, X[:, 0] ** 2))
X = np.column_stack((X, X[:, 1] ** 2))
X = np.column_stack((X, X[:, 0] * X[:, 1]))

# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

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

# Visualizing set
X_set, y_set = X_train, y_train

# Step 1: Generate meshgrid
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

# Step 2: Stack into two-column matrix
two_column_matrix = np.c_[X1.ravel(), X2.ravel()]

# Step 3: Add polynomial features (manually)
poly_matrix = np.column_stack((
    two_column_matrix,
    two_column_matrix[:, 0] ** 2,
    two_column_matrix[:, 1] ** 2,
    two_column_matrix[:, 0] * two_column_matrix[:, 1]
))






# Step 4: Inverse transform using existing StandardScaler
inverse_transformed = sc.inverse_transform(poly_matrix)

# Step 5: Extract first two columns (unscaled X1 and X2)
first_two_columns = inverse_transformed[:, :2]

# --- Phase 2: Transform back to scaled polynomial space ---

# Step 6: Recreate polynomial features from unscaled first two columns
poly_matrix_from_inverse = np.column_stack((
    first_two_columns,
    first_two_columns[:, 0] ** 2,
    first_two_columns[:, 1] ** 2,
    first_two_columns[:, 0] * first_two_columns[:, 1]
))






# Step 7: Apply scaler again to get scaled grid features
X_grid = sc.transform(poly_matrix_from_inverse)

plt.contourf(
    X1, X2,
    classifier.predict(X_grid).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(['red', 'green'])
)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# (Optional) You can also plot the actual training points
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
plt.legend()
plt.show()


