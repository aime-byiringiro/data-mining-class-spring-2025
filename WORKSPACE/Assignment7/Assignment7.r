

# Importing the dataset
dataset = read.csv('Iris_Data.csv')


# Encode categorical data 
dataset$Species = factor(dataset$Species,
                        levels = c('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        labels = c(0, 1, 2))

# Feature Scaling
scaled_cols = scale(dataset[, 1:2])
dataset[, 1:2] = scaled_cols

#library(e1071) # install.packages('e1071')
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
















#


