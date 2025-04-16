

# Importing the dataset
dataset = read.csv('Disease_Data.csv')

# Fitting Polynomial Regression to the whole dataset
dataset$Day2 = dataset$Day^2 # we used Day as an independent variable
regressor = lm(formula = Cumulative.Cases ~ .,
               data = dataset) # in regerssor we use the cumulative cases as the dependent variable

# Finding the optimal degree using p-value
summary(regressor)
dataset$Day3 = dataset$Day^3
regressor = lm(formula = Cumulative.Cases ~ .,
               data = dataset)
summary(regressor)

dataset$Day4 = dataset$Day^4
regressor = lm(formula = Cumulative.Cases ~ .,
               data = dataset)
summary(regressor)

dataset$Day5 = dataset$Day^5
regressor = lm(formula = Cumulative.Cases ~ .,
               data = dataset)
summary(regressor)

# dataset$Day6 = dataset$Day^6
# regressor = lm(formula = Cumulative.Cases ~ .,
#                data = dataset)
# summary(regressor)

# Making a single prediction
predictedCases <- predict(regressor, newdata = data.frame(Day = 365, Day2 = 365^2, Day3 = 365^3, Day4 = 365^4, Day5 = 365^5))
print(paste("Predicted cumulative cases:", predictedCases))

# Visualizing the Polynomial Regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Day, y = dataset$Cumulative.Cases),
             colour = 'red') +
  geom_point(aes(x = dataset$Day, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Cumulative Cases vs Day') +
  xlab('Day') +
  ylab('Cumulative Cases')