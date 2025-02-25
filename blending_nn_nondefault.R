## Richter's Predictor Modeling Earthquake Damage
## Creating Neural Network Classifier to be used in blended ensemble model. 
# Kylie Foster

# Useful websites:

## Loading required packages -----
#library(klaR)
library(plyr) #  to rename factors using revalue
library(tidyverse)
library(caret) #  for all models
library(MLmetrics) #  additional metrics
library(tictoc) #  to check timing of models
library(e1071)
library(nnet) #  for neural network
library(corrplot) # to plot correlations
library(RColorBrewer) # nice colours for plots

## Function to calculate F1 micro score -----
f1_micro <- function(data, lev = NULL, model = NULL) {
  Tot <- dim(data)[1] #  total number of observations
  conf <- confusionMatrix(data$pred, data$obs)[[2]] #  confusion matrix table
  TP <- conf[1, 1] + conf[2, 2] + conf[3, 3] #  true positives
  f1 <- TP/Tot #  micro Precision, micro Recall and micro F1 are all equal
  c(F1 = f1)
}

## Loading training data -----
# Loading labels
train_labels <- read_csv("./Richters_Predictor/data/train_labels.csv")

# Loading predictors
train_values <- read_csv("./Richters_Predictor/data/train_values.csv") %>%
  mutate_if(is.character, as.factor) #  convert characters to factors

# Combining data sets
train <- full_join(train_labels, train_values, by = "building_id") 

# Removing useless variables
train <- select(train, -geo_level_2_id, -geo_level_3_id, -building_id)

# Keeping only variables with high importance from random forest results (keeping top 11 predictors)
train <- select(train, damage_grade, geo_level_1_id, area_percentage, age, height_percentage, 
                foundation_type, ground_floor_type, count_families, roof_type, 
                other_floor_type, land_surface_condition, position)
# geo_level_1_id                         100.000
# area_percentage                         49.127
# age                                     45.710
# height_percentage                       29.764
# foundation_type                         18.002
# ground_floor_type                       15.714
# count_families                          12.521
# roof_type                               11.786
# other_floor_type                        11.421
# land_surface_condition                  11.300
# position                                11.039

# Converting all factors to dummy variables
dmy <- dummyVars(" ~ .", data = train) #  dummyVars from caret package
train <- data.frame(predict(dmy, newdata = train)) %>%
  mutate(damage_grade = as.factor(damage_grade))

# Changing levels of Y to prevent problems with level names later.
train$damage_grade <- revalue(train$damage_grade, 
                              c("1" = "low", "2"="med", "3"="high"))

# Checking correlations
# corrplot(cor(select_if(train, is.numeric)), type="upper", order="hclust",
#          col=brewer.pal(n=8, name="RdYlBu"))
# There are some highly correlated predictors.

# splitting into two data sets (75/25%)
set.seed(200)
train_index <- createDataPartition(train$damage_grade, p = 0.75, 
                                   list = FALSE, 
                                   times = 1)
train_layer1 <- train[train_index, ] #  used to fit models in first layer of ensemble
#train_layer2 <- train[-train_index, ] #  used to fit final model combining layer 1 model predictions

## Setting up code that is common for all first level models -----
my_control <- trainControl(method = "cv", # for “cross-validation”
                           number = 5, # number of k-folds
                           summaryFunction = f1_micro,
                           allowParallel = TRUE,
                           classProbs = TRUE,
                           verboseIter = TRUE)
seed <- 123
metric <- "F1"

## Neural Network -----
set.seed(seed)
nnet_grid <- expand.grid(.size = c(10, 20, 25, 30, 35), # number of units in the single hidden layer
                         .decay = c(0.5, 0.1, 1e-2, 1e-3)) # regularisation parameter to avoid over-fitting
tic()
nn_model <- train(x = train_layer1[, -1],
                  y = train_layer1$damage_grade,
                  method = "nnet", 
                  metric = metric,
                  trControl = my_control,
                  tuneGrid = nnet_grid,
                  preProc = c("center", "scale", "nzv","corr"))  # nzv checks for near zero variance predictors
toc() # 19566.42 sec

# Printing results
print(nn_model) # 0.6441581
# Plotting results
jpeg("nn_nondefault.jpg")
plot(nn_model)
dev.off()

# save the model to disk
saveRDS(nn_model, "./nn_model_nondefault.rds")

