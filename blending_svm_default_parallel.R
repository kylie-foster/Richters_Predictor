## Richter's Predictor Modeling Earthquake Damage
## Creating SVM Classifier to be used in blended ensemble model. 
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
library(kernlab) #  for SVM

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

# Converting all factors to dummy variables
dmy <- dummyVars(" ~ .", data = train) #  dummyVars from caret package
train <- data.frame(predict(dmy, newdata = train)) %>%
  mutate(damage_grade = as.factor(damage_grade))

# Changing levels of Y to prevent problems with level names later.
train$damage_grade <- revalue(train$damage_grade, 
                              c("1" = "low", "2"="med", "3"="high"))

# splitting into two data sets (80/20%)
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

## SVM -----
# Setting up parallel workers
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

set.seed(seed)
tic()
svm_model <- train(damage_grade ~. ,
                   data = train_layer1,
                   method = "svmRadial", 
                   metric = metric,
                   trControl = my_control,
                   preProc = c("center", "scale"))
toc()
# Stopping parallel workers
stopCluster(cl)

# Printing results
print(svm_model) 
# Plotting results
jpeg("svm_default.jpg")
plot(svm_model)
dev.off()
# compare predicted outcome and true outcome
#confusionMatrix(predict(knn_model), train_layer1$damage_grade)

# save the model to disk
saveRDS(svm_model, "./svm_model_default.rds")

# predict the outcome on a train_layer2 set
#knn_pred <- predict(knn_model, train_layer2)
