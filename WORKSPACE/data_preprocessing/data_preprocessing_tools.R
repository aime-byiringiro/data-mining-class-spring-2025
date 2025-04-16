# Data Preprocessing Tools

# Importing the dataset
dataset = read.csv('Data.csv')

# Taking care of missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     mean(dataset$Age, na.rm = TRUE),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        mean(dataset$Salary, na.rm = TRUE),
                        dataset$Salary)

# Encoding categorical data
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'),
                         labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# split = sample.split(dataset$Age, SplitRatio = 0.8)
# We always want to use a dependent variable because of the .....
#.8 is how many rows do I want 
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_scaled_cols = scale(training_set[, 2:3])
training_set[, 2:3] = training_scaled_cols
test_set[, 2:3] = scale(test_set[, 2:3], center=attr(training_scaled_cols, 'scaled:center'),
  
                      scale=attr(training_scaled_cols, 'scaled:scale'))
 # simple to hard and more details
 #view
 
 # print
 # str