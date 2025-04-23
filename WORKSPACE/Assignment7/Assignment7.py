import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# Importing the dataset
dataset = pd.read_csv('Iris_Data.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = np.array([mapping[val] for val in y])

# Only use Sepal Length and Sepal Width
X_sepal = X[:, [0, 1]]

# Scale the two features
sc = StandardScaler()
X_sepal_scaled = sc.fit_transform(X_sepal)

# Train SVM using only Sepal features
classifier_sepal = SVC(kernel='rbf')
classifier_sepal.fit(X_sepal_scaled, y)
print("rbf accuracy:", classifier_sepal.score(X_sepal_scaled,y))




#confusion matrix 
y_pred = classifier_sepal.predict(X_sepal_scaled)
print(confusion_matrix(y, y_pred))


# Create mesh grid
x1_min, x1_max = X_sepal_scaled[:, 0].min() - 1, X_sepal_scaled[:, 0].max() + 1
x2_min, x2_max = X_sepal_scaled[:, 1].min() - 1, X_sepal_scaled[:, 1].max() + 1
X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                     np.arange(x2_min, x2_max, 0.01))

# Predict grid points
Z = classifier_sepal.predict(np.array([X1.ravel(), X2.ravel()]).T)
Z = Z.reshape(X1.shape)

# Plotting
plt.figure(figsize=(10, 6))
background_colors = ['#ffcccc', '#ccffcc', '#ccccff']  # light shades for background
dot_colors = ['#cc0000', '#009900', '#0000cc']         # darker shades for dots

# Background regions
plt.contourf(X1, X2, Z, alpha=0.3, cmap=ListedColormap(background_colors))

# Plot observation points
for i, color, label in zip(range(3), dot_colors, ['Setosa', 'Versicolor', 'Virginica']):
    plt.scatter(X_sepal_scaled[y == i, 0], X_sepal_scaled[y == i, 1],
                c=color, label=label, edgecolor='black', alpha=0.8)

plt.title('SVM Decision Regions (Sepal Features - Scaled)')
plt.xlabel('Scaled Sepal Length')
plt.ylabel('Scaled Sepal Width')
plt.legend()
plt.show()
