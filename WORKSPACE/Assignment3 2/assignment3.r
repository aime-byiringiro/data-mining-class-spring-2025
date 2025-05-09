# Multiple Linear Regression in R for COSC 40023 Assignment 3
# Texas Christian University - Spring 2025
# Due: February 19th
# Importing the dataset
dataset = read.csv('Crime_Data.csv')

# Viewing the first few rows of the dataset
head(dataset)

# Splitting the dataset into the Training set and Test set
# 10% Test set, with seed set to 123
library(caTools)
set.seed(123)
split = sample.split(dataset$Y, SplitRatio = 0.9)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Multiple Linear Regression to the Training set
regressor = lm(Y ~ ., data = training_set)
summary(regressor)

# Building the optimal model using Backward Elimination
# Significance level = 5%

# Step 1: Full Model
regressor_opt = lm(Y ~ X1 + X2 + X3 + X4 + X5 + X6, data = training_set)
# Step 2: Removing the predictor with the highest p-value > 0.05
regressor_opt = lm(Y ~ X1 + X2 + X4 + X5 + X6, data = training_set)
# Step 3: Repeat until all p-values < 0.05
regressor_opt = lm(Y ~ X1 + X2 + X4 + X5, data = training_set)

regressor_opt = lm(Y ~ X1 + X2 + X4, data = training_set)

# Predicting the Test set results using the optimal model
y_pred = predict(regressor_opt, newdata = test_set)
print("Predicted Y values for the Test Set:")
print(y_pred)

# Printing all the coefficients and the intercept
print("Coefficients of the optimal model:")
print(coefficients(regressor_opt))

# Predicting Y for given values: X1 = 500, X2 = 50, X3 = 40, X4 = 30, X5 = 20, X6 = 10
new_data = data.frame(X1=500, X2=50, X4=30)
y_new = predict(regressor_opt, newdata = new_data)
print("Predicted Y for given values:")
print(y_new)

