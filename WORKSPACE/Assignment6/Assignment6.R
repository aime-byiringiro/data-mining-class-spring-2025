# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')



# adding a new column of age squared
dataset$AgeSquared = dataset$Age^2
dataset$EstimatedSalarySquared = dataset$EstimatedSalary^2
#new colum for age time estimated salary
dataset$AgeEstimatedSalary = dataset$Age * dataset$EstimatedSalary
# Encoding categorical data

#change this to numeric

dataset$Purchased = as.factor(dataset$Purchased) #

#make Purchased the last column in the dataset

dataset = dataset[, c(1, 2, 4, 5,3)]
#pring fields and they new order numeric
colnames(dataset)

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_scaled_cols = scale(training_set[, 1:4])
training_set [, 1:4] = training_scaled_cols
test_set [, 1:4] = scale(test_set [, 1:4],
                        center = attr(training_scaled_cols, 'scaled:center'),
                        scale = attr(training_scaled_cols, 'scaled:scale'))


# Fitting Logistic Regression to the Training set
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set)
y_pred = as.factor(ifelse(prob_pred > 0.5, 1, 0))


# Showing the Confusion Matrix and Accuracy
library(caret)
cm = confusionMatrix(y_pred, test_set$Purchased)
print(cm$table)
print(cm$overall['Accuracy'])







