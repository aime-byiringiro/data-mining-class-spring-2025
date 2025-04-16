#Author Aime Byiringiro

#We are  trying to predict if people in specific countries at certain age
#  will or will not purchase 
# This is a classification problem since the dependent variable is categorical. 


# Importing data from Customer_Data.csv in the working directory 
dataset = read.csv('Customer_Data.csv')

# replacing misssing Age with median of all ages
dataset$Age = ifelse(is.na(dataset$Age),
                     median(dataset$Age, na.rm = TRUE),
                     dataset$Age)
                     
#replacing missing salaires with mean of salaries 
dataset$Salary = ifelse(is.na(dataset$Salary),
                        mean(dataset$Salary, na.rm = TRUE),
                        dataset$Salary)
# 
# Encoding Country column data
#India = 1, Sri lanka = 2, China = 3
dataset$Country = factor(dataset$Country,
                         levels = c('India', 'Sri lanka', 'China'),
                         labels = c(1, 2, 3))

# Encoding Purchased column data 
#No = 0, Yes. =1
dataset$Purchased = factor(dataset$Purchased,
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))

# 1/3 of the data will be used for test and ther rest for training
library(caTools)
set.seed(123) # for reproducibility
# we use the dependent variable to split becuase it is a classification problem
# if it was a regression problem we would use the independent variables

split = sample.split(dataset$Purchased, SplitRatio = 2/3)


training_set = subset(dataset, split == TRUE) # training set which is 2/3
test_set = subset(dataset, split == FALSE) # test set which is 1/3

# Feature Scaling
# Define min-max normalization function
min_max_norm <- function(x, min_val, max_val) {

  return((x - min_val) / (max_val - min_val))
}

# Compute min and max values for each feature in the training set

# explain why 2, 3, and 3 are used
# [, 2:3], 2, min means select columns 2 and 3 and apply min function to them 
# we use 2 and 3 because we want to normalize age and salary
# we dont want to normalize country and purchased because they are categorical

min_vals <- apply(training_set[, 2:3], 2, min)
max_vals <- apply(training_set[, 2:3], 2, max)

# Normalize training set using its own min-max values
training_set[, 2:3] <- as.data.frame(mapply(min_max_norm, 
                                            training_set[, 2:3], 
                                            min_vals, 
                                            max_vals))

# Normalize test set using training set's min-max values
test_set[, 2:3] <- as.data.frame(mapply(min_max_norm, 
                                        test_set[, 2:3], 
                                        min_vals, 
                                        max_vals))


# str
