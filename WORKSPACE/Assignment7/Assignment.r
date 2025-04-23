

# Importing the dataset
dataset = read.csv('Iris_Data.csv')


# Encode categorical data 
dataset$Species = factor(dataset$Species,
                        levels = c('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        labels = c(0, 1, 2))