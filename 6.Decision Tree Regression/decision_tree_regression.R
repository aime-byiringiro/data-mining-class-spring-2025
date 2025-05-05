# Decision Tree Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# Fitting Decision Tree Regression to the dataset
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset)

# Making a single prediction
print(predict(regressor, newdata = data.frame('Level' = c(6.6))))

# Visualizing the Decision Tree Regression results (for higher resolution and smoother curve)
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame('Level' = x_grid))),
            colour = 'blue') +
  ggtitle('Decision Tree Regression') +
  xlab('Level') +
  ylab('Salary')