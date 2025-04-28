

# Importing the dataset
dataset = read.csv('Iris_Data.csv')


# Encode categorical data 
dataset$Species = factor(dataset$Species,
                        levels = c('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        labels = c(0, 1, 2))
# datase$Species = as.factor()

# Feature Scaling
scaled_cols = scale(dataset[, 1:2])
dataset[, 1:2] = scaled_cols

library(e1071) # install.packages('e1071')
#linear kernel
#classifier = svm(formula = Species ~ .,
               #  data = dataset,
               #  type = 'C-classification', # epds_regression is the default for SVR
                # kernel = 'linear') # radial is the default for SVR
# Predicting the Test set results
#y_pred = predict(classifier, newdata = dataset)
# Showing the Confusion Matrix and Accuracy
#library(caret)
#cm = confusionMatrix(y_pred, dataset$Species)
#print(cm$table)
#print(cm$overall['Accuracy'])


#poly kernel
#classifier = svm(formula = Species ~ .,
                # data = dataset,
                # type = 'C-classification', # epds_regression is the default for SVR
                # kernel = 'poly',
                # degree = 3) # radial is the default for SVR
# Predicting the Test set results
#y_pred = predict(classifier, newdata = dataset)
# Showing the Confusion Matrix and Accuracy
#library(caret)
#cm = confusionMatrix(y_pred, dataset$Species)
#print(cm$table)
#print(cm$overall['Accuracy'])



#sigmoid 
#classifier = svm(formula = Species ~ .,
#  data = dataset,
# type = 'C-classification', # epds_regression is the default for SVR
#  kernel = 'sigmoid') # radial is the default for SVR
# Predicting the Test set results
#y_pred = predict(classifier, newdata = dataset)
# Showing the Confusion Matrix and Accuracy
#library(caret)
#cm = confusionMatrix(y_pred, dataset$Species)
#print(cm$table)
#print(cm$overall['Accuracy'])



#radial basis
classifier = svm(formula = Species ~ .,
                data = dataset,
                type = 'C-classification', # epds_regression is the default for SVR
                kernel = 'radial') # radial is the default for SVR
# Predicting the Test set results
y_pred = predict(classifier, newdata = dataset)
# Showing the Confusion Matrix and Accuracy
library(caret)
cm = confusionMatrix(y_pred, dataset$Species)
print(cm$table)
print(cm$overall['Accuracy'])



# Create a grid over feature space
x1_range <- seq(min(dataset[, 1]) - 1, max(dataset[, 1]) + 1, by = 0.01)
x2_range <- seq(min(dataset[, 2]) - 1, max(dataset[, 2]) + 1, by = 0.01)
grid <- expand.grid(Sepal.Length = x1_range, Sepal.Width = x2_range)

# Predict class for each grid point
grid$Species <- predict(classifier, newdata = grid)

# Convert prediction to factor for ggplot fill
grid$Species <- factor(grid$Species, levels = c(0, 1, 2),
                       labels = c('Setosa', 'Versicolor', 'Virginica'))

# Plot decision boundaries and data points
ggplot() +
  geom_tile(data = grid, aes(x = Sepal.Length, y = Sepal.Width, fill = Species), alpha = 0.8) +
  scale_fill_manual(values = c('#ffcccc', '#ccffcc', '#ccccff')) +
  geom_point(data = dataset, aes(x = Sepal.Length, y = Sepal.Width, color = Species),
             shape = 21, fill = NA, size = 2, stroke = 1) +
  scale_color_manual(values = c('#cc0000', '#009900', '#0000cc')) +
  labs(title = 'SVM Decision Regions (Sepal Features - Scaled)',
       x = 'Scaled Sepal Length', y = 'Scaled Sepal Width') +
  theme_minimal()









#


