# Polynomial Regression

# Importing the dataset from a CSV file named 'Position_Salaries.csv'
dataset = read.csv('Position_Salaries.csv')

# Selecting only columns 2 and 3 from the dataset
dataset = dataset[, 2:3]

# Creating a new feature by squaring the Level column to introduce polynomial terms
dataset$Level2 = dataset$Level^2

# Fitting a polynomial regression model using both the original Level and Level^2
# The formula Salary ~ . means predict Salary using all other variables in the dataset
regressor = lm(formula = Salary ~ .,
               data = dataset)
summary(regressor)

dataset$Level3 = dataset$Level^3

# Fitting a new polynomial regression model with Level, Level^2, and Level^3
regressor = lm(formula = Salary ~ .,
               data = dataset)

# Checking if the third-degree polynomial improves the model
summary(regressor)

# Adding a fourth polynomial term: Level^4
dataset$Level4 = dataset$Level^4

# Fitting another polynomial model with degrees 1 through 4
regressor = lm(formula = Salary ~ .,
               data = dataset)

# Checking if the fourth-degree polynomial further improves the model
summary(regressor)

# Adding a fifth polynomial term: Level^5
# Note: There appears to be a commented-out section in the original code
# dataset$Level5 = dataset$Level^5

# Fitting a fifth-degree polynomial model
# regressor = lm(formula = Salary ~ .,
#                data = dataset)

# Visualizing the Polynomial Regression results
# Loading the ggplot2 visualization package
library(ggplot2)

# Creating a scatter plot with the polynomial regression curve
ggplot() +
  # Adding red scatter points representing actual data
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  # Adding blue curve showing the polynomial model's predictions
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  # Adding title and axis labels
  ggtitle('Polynomial Regression') +
  xlab('Level') +
  ylab('Salary')

# Creating a smoother, higher-resolution visualization of the model
library(ggplot2)

# Creating a sequence of Level values with smaller intervals for smoother plotting
# The sequence goes from minimum to maximum Level value with steps of 0.1
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

# Creating a more detailed plot with the finer grid
ggplot() +
  # Adding the original data points in red
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  # Adding a smoother blue curve using predictions from the model on the finer grid
  # Note: This only includes Level^1 through Level^4 in the prediction data frame
  geom_line(aes(x = x_grid, y = predict(regressor,
                                        newdata = data.frame('Level' = x_grid,
                                                             'Level2' = x_grid^2,
                                                             'Level3' = x_grid^3,
                                                             'Level4' = x_grid^4))),
            colour = 'blue') +
  # Adding title and axis labels
  ggtitle('Polynomial Regression') +
  xlab('Level') +
  ylab('Salary')