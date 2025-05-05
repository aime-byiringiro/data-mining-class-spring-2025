# k-NN

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')

# Encoding categorical data
dataset$Purchased = as.factor(dataset$Purchased)

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_scaled_cols = scale(training_set[, 1:2])
training_set[, 1:2] = training_scaled_cols
test_set[, 1:2] = scale(test_set[, 1:2],
                        center = attr(training_scaled_cols, 'scaled:center'),
                        scale = attr(training_scaled_cols, 'scaled:scale'))

# Fitting k-NN to the Training set and Predicting the Test set results
library(kknn)
classifier = kknn(formula = Purchased ~ ., train = training_set, test = test_set,
                  k = 17, distance = 2)
y_pred = classifier$fitted.values

# Showing the Confusion Matrix and Accuracy
library(caret)
cm = confusionMatrix(y_pred, test_set$Purchased)
print(cm$table)
print(cm$overall['Accuracy'])

# Visualizing the Training set results
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
classifier = kknn(formula = Purchased ~ ., train = training_set, test = grid_set,
                  k = 17, distance = 2)
y_grid = classifier$fitted.values
plot(NULL,
     main = 'k-NN (Training set)',
     xlab = 'Age (Scaled)', ylab = 'Estimated Salary (Scaled)',
     xlim = range(X1), ylim = range(X2))
points(grid_set, pch = 20, col = c('tomato', 'springgreen3')[y_grid])
points(set, pch = 21, bg = c('red3', 'green4')[set$Purchased])

# Visualizing the Test set results
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
classifier = kknn(formula = Purchased ~ ., train = training_set, test = grid_set,
                  k = 17, distance = 2)
y_grid = classifier$fitted.values
plot(NULL,
     main = 'k-NN (Test set)',
     xlab = 'Age (Scaled)', ylab = 'Estimated Salary (Scaled)',
     xlim = range(X1), ylim = range(X2))
points(grid_set, pch = 20, col = c('tomato', 'springgreen3')[y_grid])
points(set, pch = 21, bg = c('red3', 'green4')[set$Purchased])