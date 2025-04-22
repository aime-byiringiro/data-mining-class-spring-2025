# Random Forest Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# Fitting Random Forest Regression to the whole dataset
library(randomForest)
regressor = randomForest(formula = Salary ~ .,
                         data = dataset,
                         ntree = 100)

# Making a single prediction
print(predict(regressor, newdata = data.frame('Level' = c(6.6))))

# Visualizing the Random Forest Regression results (for higher resolution and smoother curve)
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Random Forest Regression') +
  xlab('Level') +
  ylab('Salary')