
# STEP 1

dataset = read.csv('Insurance_Data.csv')
dataset$sex = factor(dataset$sex,
                           levels = c('male', 'female'),
                           labels = c(0, 1))

dataset$smoker = factor(dataset$smoker,
                           levels = c('no', 'yes'),
                           labels = c(0, 1))

dataset$region = factor(dataset$region,
                           levels = c('northeast', 'southeast', 'southwest', 'northwest'),
                           labels = c(1, 2, 3, 4))

# manually avoid dummy variable trap if needed





##############################
#STEP 2
############################
# use all variable independent variables to build an linear regression model

regressor = lm(formula = charges ~ ., data = dataset)
summary(regressor)

# using library(ggplot2), plot Age as X and charges as Y and add the regression line, red dots for observed data ans blue dots for predicted data
 #add labels X is age and Y is charges

 library(ggplot2)

ggplot() +
  geom_point(aes(x = dataset$age, y = dataset$charges),
             colour = 'red') +  
  geom_point(aes(x = dataset$age, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Age vs Charges') +
  xlab('Age') +
  ylab('Charges')


  ##################################
  # step 3
  ################################


  dataset$age_squared = dataset$age^2 # add a new column for bmi squared
  # make sure there no repetive clumsy for age degree1
  dataset = dataset[, -which(names(dataset) == "age")] # remove age column to avoid repetition

  dataset$bmi_high = ifelse(dataset$bmi >= 30, 1, 0) # add new column for BMI. IBM > = 30 is high BMI and < 30 is low BMI

  dataset$smoker_bmi = ifelse(dataset$bmi >= 30 & dataset$smoker == 1, 1, 0) # add new column for smomker and BMI. if bmi is hight and smoker do 1, otherwise 0



  new_regressor = lm(formula = charges ~ ., data = dataset)

   library(ggplot2)

ggplot() +
  geom_point(aes(x = dataset$age, y = dataset$charges),
             colour = 'red') +  
  geom_line(aes(x = dataset$age, y = predict(new_regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Age vs Charges') +
  xlab('Age') +
  ylab('Charges')


  ################################
  # step 4: Optimal Linear Regression
  ###############################

  regressor_opt = lm(charges ~ sex + bmi + children + smoker + region + age_squared + bmi_high + smoker_bmi, data = dataset)
  summary(regressor_opt)
  
  regressor_opt = lm(charges ~ bmi + children + smoker + region + age_squared + bmi_high + smoker_bmi, data = dataset)
  summary(regressor_opt)
  
  regressor_opt = lm(charges ~ bmi + smoker + region + age_squared + bmi_high + smoker_bmi, data = dataset)
  summary(regressor_opt)
  
  regressor_opt = lm(charges ~ bmi + smoker + age_squared + bmi_high+ smoker_bmi, data = dataset)
  summary(regressor_opt)
  
  regressor_opt = lm(charges ~ smoker + age_squared+ smoker_bmi, data = dataset)
  summary(regressor_opt)
  
  # step 4
  
  library(e1071) # install.packages('e1071')
  
  
  regressor = svm(formula = Charges ~ .,
                  data = dataset,
                  type = 'eps-regression', # epds_regression is the default for SVR
                  kernel = 'radial') # radial is the default for SVR
  

  
  
  
  
  




 











   

   
  # 3. add colum for BMI. IBM > = 30 is high BMI and < 30 is low BMI 1 adn 0
  # add new column for smomker and BMI. if bmi is hight and smoker do 1, otherwise 0
   
   # codes for step3

   
    




















  ########################################
  # STEP 3
  #######################################







