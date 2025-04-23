# Regularization

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Feature Scaling
dataset[, c(1:3, 5)] = scale(dataset[, c(1:3, 5)])

# Preparing independent variable matrix
X = model.matrix(Profit ~ ., dataset)[, 2:6]

# Preparing dependent variable vector
y = dataset$Profit

# Fitting Multiple Linear Regression with Regularization
library(glmnet)
regressor1 = glmnet(X, y, alpha = 0.5, lambda = 0)
print(coef(regressor1))
regressor2 = glmnet(X, y, alpha = 0.5, lambda = 1)
print(coef(regressor2))