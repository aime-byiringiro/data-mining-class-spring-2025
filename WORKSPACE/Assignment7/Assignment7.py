import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Iris_Data.csv')
X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = np.array([mapping[val] for val in y])
#Iris-setosa; Iris-versicolor; Iris-virginica

#no need to split the dataset into a training set and a test set. 


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)





# Using Linear
#from sklearn.svm import SVC
#classifier_linear = SVC(kernel = 'linear')
#classifier_linear.fit(X_scaled, y)
#print("SVC accuracy:", classifier_linear.score(X_scaled, y))


#Using poly
#from sklearn.svm import SVC
#classifier_poly = SVC(kernel='poly', degree=3, C=1.0)
#lassifier_poly.fit(X_scaled, y)
#print("Poly accuracy:", classifier_poly.score(X_scaled,y))


#Using Sigmoid 
#from sklearn.linear_model import LogisticRegression
#classifier_sigmoid = LogisticRegression()
#classifier_sigmoid.fit(X_scaled, y)
#print("Sigmoid accuracy:", classifier_sigmoid.score(X_scaled, y))



#Using rbf
from sklearn.svm import SVC
classifier_rbf = SVC(kernel='rbf')
classifier_rbf.fit(X_scaled, y)
print("rbf accuracy:", classifier_rbf.score(X_scaled,y))



y_pred = classifier_rbf.predict(X_scaled)

#confusion matrix 
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_pred))


from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# Reduce dimensions from 4D to 2D for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Fit the model again using the reduced features
classifier_rbf_2d = SVC(kernel='rbf')
classifier_rbf_2d.fit(X_pca, y)

# Create mesh grid
X1, X2 = np.meshgrid(
    np.arange(start=X_pca[:, 0].min() - 1, stop=X_pca[:, 0].max() + 1, step=0.01),
    np.arange(start=X_pca[:, 1].min() - 1, stop=X_pca[:, 1].max() + 1, step=0.01)
)

# Predict each point on the grid
Z = classifier_rbf_2d.predict(np.array([X1.ravel(), X2.ravel()]).T)
Z = Z.reshape(X1.shape)

# Plot decision boundaries
plt.figure(figsize=(10, 6))
plt.contourf(X1, X2, Z, alpha=0.75, cmap=ListedColormap(['red', 'green', 'blue']))

# Plot actual data points
for i, color, label in zip(range(3), ['red', 'green', 'blue'], ['Setosa', 'Versicolor', 'Virginica']):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                color=color, label=label, edgecolor='black')

plt.title('SVM with RBF Kernel (Iris Dataset - PCA 2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()




















