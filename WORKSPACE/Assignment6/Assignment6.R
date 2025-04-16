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
training_scaled_cols = scale(training_set)
training_set = training_scaled_cols
test_set = scale(test_set,
                        center = attr(training_scaled_cols, 'scaled:center'),
                        scale = attr(training_scaled_cols, 'scaled:scale'))


