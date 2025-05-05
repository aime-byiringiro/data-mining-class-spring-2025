# SVR (Support Vector Regression)

# Importing the dataset from a CSV file named 'Position_Salaries.csv'
dataset = read.csv('Position_Salaries.csv')

# Selecting only columns 2 and 3 from the dataset
# This likely keeps only the Level and Salary columns, removing any ID column
dataset = dataset[, 2:3]

# Loading the e1071 package which contains the SVM implementation in R
library(e1071) # install.packages('e1071')

# Fitting a Support Vector Regression model to the dataset
# The formula Salary ~ . means predict Salary using all other variables (in this case, just Level)
# type = 'eps-regression' specifies epsilon-SVR algorithm (standard for regression problems)
# kernel = 'radial' uses a Radial Basis Function (RBF) kernel, which is effective for nonlinear relationships
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression', # epds_regression is the default for SVR
                kernel = 'radial') # radial is the default for SVR

# Making a single prediction for a new data point (Level = 6.6)
# This predicts the salary for someone at position level 6.6
print(predict(regressor, newdata = data.frame('Level' = c(6.6))))

# Visualizing the SVR results
# Loading the ggplot2 visualization package
library(ggplot2)

# Creating a basic plot showing the SVR model and data points
ggplot() +
  # Adding red scatter points representing actual data
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  # Adding blue line showing the SVR model's predictions at each Level in the dataset
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  # Adding title and axis labels
  ggtitle('SVR') +
  xlab('Level') +
  ylab('Salary')

# Creating a smoother, higher-resolution visualization of the SVR model
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
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame('Level' = x_grid))),
            colour = 'blue') +
  # Adding title and axis labels
  ggtitle('SVR') +
  xlab('Level') +
  ylab('Salary')