dataset = read.csv('Dealership_Data.csv')

library(caTools)
set.seed(123)

split = sample.split(dataset$Sell.Price, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Multiple Linear Regression to the Training set
# we do not need to include the independent variable in the formula because it is already in the dataset
# but we need to include the dependent variable because it is not in the dataset
regressor = lm(formula = Sell.Price ~ List.Price, data = training_set)

# Predicting the Test set results
# newdata is the test set
# this new data is not in the training set
y_pred = predict(regressor, newdata = test_set)

# Visualising the Training set results
library(ggplot2)
ggplot() +
    geom_point(aes(x = training_set$List.Price, y = training_set$Sell.Price),
                 colour = 'red')+
    geom_line(aes(x = training_set$List.Price, y = predict(regressor, newdata = training_set)),
                colour = 'blue') +
    ggtitle('Sell Price vs List Price (Training set)') +
    xlab('List Price') +
    ylab('Sell Price')

# Visualising the Test set results
library(ggplot2)
ggplot() +
    geom_point(aes(x = test_set$List.Price, y = test_set$Sell.Price),
                 colour = 'red') +
    geom_line(aes(x = training_set$List.Price, y = predict(regressor, newdata = training_set)),
                colour = 'black') +
    ggtitle('Sell Price vs List Price (Test set)') +
    xlab('List Price') +
    ylab('Sell Price')

    # regressor is a linear model
    # new data is the test set


print(predict(regressor, newdata = data.frame('List Price' = 20)))
#

# slope and intercept
print(regressor$coefficients)
