## Richter's Predictor Modeling Earthquake Damage
## Creating blended ensemble model. 
# Kylie Foster

# Useful websites:

## Loading required packages -----
library(klaR) #  for naive bayes
library(plyr) #  to rename factors using revalue
library(tidyverse)
library(caret) #  for all models
library(MLmetrics) #  additional metrics
library(tictoc) #  to check timing of models
library(e1071)
library(nnet) #  for neural network
library(mda) #  for flexible discriminant analysis
library(earth) #  for flexible discriminant analysis

## Function to calculate F1 micro score -----
f1_micro <- function(data, lev = NULL, model = NULL) {
  Tot <- dim(data)[1] #  total number of observations
  conf <- confusionMatrix(data$pred, data$obs)[[2]] #  confusion matrix table
  TP <- conf[1, 1] + conf[2, 2] + conf[3, 3] #  true positives
  f1 <- TP/Tot #  micro Precision, micro Recall and micro F1 are all equal
  c(F1 = f1)
}

## Loading training data for random forest and Naive Bayes -----
# Loading labels
train_labels <- read_csv("./Richters_Predictor/data/train_labels.csv")

# Loading predictors
train_values <- read_csv("./Richters_Predictor/data/train_values.csv") %>%
  mutate_if(is.character, as.factor) #  convert characters to factors

# Combining data sets
train <- full_join(train_labels, train_values, by = "building_id") %>%
  mutate(damage_grade = as.factor(damage_grade))

# Removing useless variables
train <- select(train, -geo_level_2_id, -geo_level_3_id, -building_id)

# Will need to keep has_secondary_use_other as a separate predictor, but can combine all other has_secondary_use predictors
# Adding new predictor combining has_secondary_use predictors
train <- mutate(train, has_secondary_use = as.factor(case_when(has_secondary_use_hotel == 1 ~ "hotel",
                                                               has_secondary_use_agriculture == 1 ~ "ag",
                                                               has_secondary_use_rental == 1 ~ "rent",
                                                               has_secondary_use_institution == 1 ~ "inst",
                                                               has_secondary_use_school == 1 ~ "school",
                                                               has_secondary_use_industry == 1 ~ "ind",
                                                               has_secondary_use_health_post == 1 ~ "health",
                                                               has_secondary_use_gov_office == 1 ~ "gov",
                                                               has_secondary_use_use_police == 1 ~ "police",
                                                               TRUE ~ "none"))) %>%
  select(-c(has_secondary_use_hotel, #  removing unneeded predictors
            has_secondary_use_agriculture,
            has_secondary_use_rental,
            has_secondary_use_institution,
            has_secondary_use_school,
            has_secondary_use_industry,
            has_secondary_use_health_post,
            has_secondary_use_gov_office,
            has_secondary_use_use_police))

# Converting remaining binary variables to factors:
train <- mutate(train, has_superstructure_adobe_mud = as.factor(has_superstructure_adobe_mud),
                has_superstructure_mud_mortar_stone = as.factor(has_superstructure_mud_mortar_stone),
                has_superstructure_stone_flag = as.factor(has_superstructure_stone_flag),
                has_superstructure_cement_mortar_stone = as.factor(has_superstructure_cement_mortar_stone),
                has_superstructure_mud_mortar_brick = as.factor(has_superstructure_mud_mortar_brick),
                has_superstructure_cement_mortar_brick = as.factor(has_superstructure_cement_mortar_brick),
                has_superstructure_timber = as.factor(has_superstructure_timber),
                has_superstructure_bamboo = as.factor(has_superstructure_bamboo),
                has_superstructure_rc_non_engineered = as.factor(has_superstructure_rc_non_engineered),
                has_superstructure_rc_engineered = as.factor(has_superstructure_rc_engineered),
                has_superstructure_other = as.factor(has_superstructure_other),
                has_secondary_use_other = as.factor(has_secondary_use_other))

# Changing levels of Y to prevent problems with level names later.
train$damage_grade <- revalue(train$damage_grade, 
                              c("1" = "low", "2"="med", "3"="high"))

# splitting into two data sets (75/25%)
set.seed(200)
train_index <- createDataPartition(train$damage_grade, p = 0.75, 
                                   list = FALSE, 
                                   times = 1)
#train_layer1 <- train[train_index, ] #  used to fit models in first layer of ensemble
train_layer2 <- train[-train_index, ] #  used to fit final model combining layer 1 model predictions

## load rf model -----
rf_model <- readRDS("./rf_model_6mtry.rds")

# make predictions using rf model
rf_pred <- predict(rf_model, train_layer2)

## load nb model -----
nb_model <- readRDS("./nb_model_default.rds")

# make predictions using nb model
nb_pred <- predict(nb_model, train_layer2)

## Reloading training data and converting to dummy variables for remaining models -----
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

# splitting into two data sets (75/15%)
set.seed(200)
train_index <- createDataPartition(train$damage_grade, p = 0.75, 
                                   list = FALSE, 
                                   times = 1)
#train_layer1 <- train[train_index, ] #  used to fit models in first layer of ensemble
train_layer2 <- train[-train_index, ] #  used to fit final model combining layer 1 model predictions

# Making predictions using base classifies
# load fda model
fda_model <- readRDS("./fda_model_nondefault.rds")
# make predictions using the fda model
fda_pred <- predict(fda_model, train_layer2)

# load knn model
knn_model <- readRDS("./knn_model_nondefault.rds")
# make predictions using the knn model
knn_pred <- predict(knn_model, train_layer2)

# load nn model
nn_model <- readRDS("./nn_model_nondefault.rds")
# make predictions using the knn model
nn_pred <- predict(nn_model, train_layer2)

## Checking model correlations -----
data_predictions <- tibble(rf_pred = as.factor(rf_pred), 
                           knn_pred = as.factor(knn_pred),
                           nn_pred = as.factor(nn_pred),
                           nb_pred = as.factor(nb_pred),
                           fda_pred = as.factor(fda_pred))

data_cor <- transmute(data_predictions,
                      rf_pred = case_when(rf_pred == "low" ~ 1,
                                          rf_pred == "med" ~ 2,
                                          rf_pred == "high" ~ 3),
                      knn_pred = case_when(knn_pred == "low" ~ 1,
                                           knn_pred == "med" ~ 2,
                                           knn_pred == "high" ~ 3),
                      nn_pred = case_when(nn_pred == "low" ~ 1,
                                          nn_pred == "med" ~ 2,
                                          nn_pred == "high" ~ 3),
                      nb_pred = case_when(nb_pred == "low" ~ 1,
                                          nb_pred == "med" ~ 2,
                                          nb_pred == "high" ~ 3),
                      fda_pred = case_when(fda_pred == "low" ~ 1,
                                          fda_pred == "med" ~ 2,
                                          fda_pred == "high" ~ 3))
cor(data_cor) #  correlations should be less than 0.75

## Setting up code for blended model -----
my_control_blend <- trainControl(method = "cv", # for “cross-validation”
                                 number = 10, # number of k-folds
                                 summaryFunction = f1_micro,
                                 allowParallel = TRUE,
                                 classProbs = TRUE,
                                 verboseIter = TRUE)
seed <- 123
metric <- "F1"

## Blended model with only new predictions (as dummy variables) -----
features_blend <- tibble(rf_pred = as.factor(rf_pred),
                                             knn_pred = as.factor(knn_pred),
                                             nn_pred = as.factor(nn_pred),
                                             nb_pred = as.factor(nb_pred),
                         fda_pred = as.factor(fda_pred))

dmy <- dummyVars(" ~ .", data = features_blend) #  dummyVars from caret package
features_blend <- data.frame(predict(dmy, newdata = features_blend))

features_blend <- cbind(train_layer2[1], features_blend)

set.seed(seed)
tic()
blended_model <- train(damage_grade ~. ,
                       data = features_blend,
                       method = "multinom", metric = metric,
                       tuneLength = 7,
                       trControl = my_control_blend) 
toc()
# Printing results
print(blended_model) 
# size  decay  F1       
# 1     0e+00  0.6912301
# 1     1e-04  0.6913383
# 1     1e-01  0.6998876
# 3     0e+00  0.7005783
# 3     1e-04  0.7007319
# 3     1e-01  0.7006858
# 5     0e+00  0.7008086
# 5     1e-04  0.7007472
# 5     1e-01  0.7009161
# Plotting results
plot(blended_model)
# compare predicted outcome and true outcome
#confusionMatrix(predict(ensemble_model), train_layer1$damage_grade)


# save the model to disk
saveRDS(blended_model, "./blended_model_logit.rds")


# tunegrid_blend_test <- expand.grid(.mtry = c(4,5)) #  trying 6 values
# 
# set.seed(seed)
# tic()
# blended_model_test <- train(x = features_blend[, -1],
#                             y = features_blend$damage_grade,
#                             method = "parRF", metric = metric,
#                             tuneGrid = tunegrid_blend_test, trControl = my_control_blend) 
# toc()
# print(blended_model_test) 
# # mtry  F1       
# # 4     0.6998108
# # 5     0.7007932
# # varImp(blended_model_test)
# # rf_pred                                100.000
# # knn_pred                                42.675
# # geo_level_1_id                          41.586
# # nb_pred                                 35.979
# # area_percentage                         24.333
# # age                                     24.076
# # nn_pred                                 21.922
# # height_percentage                       17.835
# # foundation_type.r                       10.886
# # has_superstructure_mud_mortar_stone      9.472
# # count_floors_pre_eq                      9.114
# # count_families                           8.633
# # ground_floor_type.v                      7.513
# # has_superstructure_cement_mortar_brick   6.293
# # ground_floor_type.f                      6.018
# # has_superstructure_timber                5.708
# # roof_type.x                              5.281
# # foundation_type.i                        5.255
# # other_floor_type.q                       15.205
# # position.s                               4.356

## Loading test data and performing same transformations as were carried out on training data -----
# Loading predictors
test_values <- read_csv("./Richters_Predictor/data/test_values.csv") %>%
  mutate_if(is.character, as.factor) #  convert characters to factors

# Checking for missing values
any(is.na(test_values))

# Removing useless variables
test_values <- select(test_values, -geo_level_2_id, -geo_level_3_id, -building_id)

# Will need to keep has_secondary_use_other as a separate predictor, but can combine all other has_secondary_use predictors
# Adding new predictor combining has_secondary_use predictors
test_values <- mutate(test_values, has_secondary_use = as.factor(case_when(has_secondary_use_hotel == 1 ~ "hotel",
                                                                           has_secondary_use_agriculture == 1 ~ "ag",
                                                                           has_secondary_use_rental == 1 ~ "rent",
                                                                           has_secondary_use_institution == 1 ~ "inst",
                                                                           has_secondary_use_school == 1 ~ "school",
                                                                           has_secondary_use_industry == 1 ~ "ind",
                                                                           has_secondary_use_health_post == 1 ~ "health",
                                                                           has_secondary_use_gov_office == 1 ~ "gov",
                                                                           has_secondary_use_use_police == 1 ~ "police",
                                                                           TRUE ~ "none"))) %>%
  select(-c(has_secondary_use_hotel, #  removing unneeded predictors
            has_secondary_use_agriculture,
            has_secondary_use_rental,
            has_secondary_use_institution,
            has_secondary_use_school,
            has_secondary_use_industry,
            has_secondary_use_health_post,
            has_secondary_use_gov_office,
            has_secondary_use_use_police))

# Converting remaining binary variables to factors:
test_values <- mutate(test_values, has_superstructure_adobe_mud = as.factor(has_superstructure_adobe_mud),
                      has_superstructure_mud_mortar_stone = as.factor(has_superstructure_mud_mortar_stone),
                      has_superstructure_stone_flag = as.factor(has_superstructure_stone_flag),
                      has_superstructure_cement_mortar_stone = as.factor(has_superstructure_cement_mortar_stone),
                      has_superstructure_mud_mortar_brick = as.factor(has_superstructure_mud_mortar_brick),
                      has_superstructure_cement_mortar_brick = as.factor(has_superstructure_cement_mortar_brick),
                      has_superstructure_timber = as.factor(has_superstructure_timber),
                      has_superstructure_bamboo = as.factor(has_superstructure_bamboo),
                      has_superstructure_rc_non_engineered = as.factor(has_superstructure_rc_non_engineered),
                      has_superstructure_rc_engineered = as.factor(has_superstructure_rc_engineered),
                      has_superstructure_other = as.factor(has_superstructure_other),
                      has_secondary_use_other = as.factor(has_secondary_use_other))

## load rf model -----
#rf_model <- readRDS("./rf_model_6mtry.rds")

# make predictions using rf model
rf_pred_test <- predict(rf_model, test_values)

## load nb model -----
#nb_model <- readRDS("./nb_model_default.rds")

# make predictions using nb model
nb_pred_test <- predict(nb_model, test_values)

## Reloading training data and converting to dummy variables for remaining models -----
# Loading predictors
test <- read_csv("./Richters_Predictor/data/test_values.csv") %>%
  mutate_if(is.character, as.factor) #  convert characters to factors

# Checking for missing values
any(is.na(test))

# Removing useless variables
test_values <- select(test, -geo_level_2_id, -geo_level_3_id, -building_id)

# Converting all factors to dummy variables
dmy <- dummyVars(" ~ .", data = test_values) #  dummyVars from caret package
test_values <- data.frame(predict(dmy, newdata = test_values))

# Making predictions using base classifies
# make predictions using the fda model
fda_pred_test <- predict(fda_model, test_values)

# load knn model
#knn_model <- readRDS("./knn_model_default.rds")
# make predictions using the knn model
knn_pred_test <- predict(knn_model, test_values)

# load nn model
#nn_model <- readRDS("./nn_model_default.rds")
# make predictions using the knn model
nn_pred_test <- predict(nn_model, test_values)

## Predictions on test data -----
features_blend_test <- tibble(rf_pred = as.factor(rf_pred_test),
                         knn_pred = as.factor(knn_pred_test),
                         nn_pred = as.factor(nn_pred_test),
                         nb_pred = as.factor(nb_pred_test),
                         fda_pred = as.factor(fda_pred_test))

dmy <- dummyVars(" ~ .", data = features_blend_test) #  dummyVars from caret package
features_blend_test <- data.frame(predict(dmy, newdata = features_blend_test))

pred <- predict(object = blended_model, newdata = features_blend_test)

# Changing levels of Y from names to numbers
pred <- revalue(pred,
                c("low" = "1", "med" = "2", "high" = "3"))

predictions <- as.data.frame(cbind(building_id = test$building_id, damage_grade = pred))

## Saving predictions as csv -----
write_csv(predictions, "./Richters_Predictor/data/test_predictions_blended_4.csv", na = "NA", append = FALSE, col_names = TRUE,
          quote_escape = "double")
