## Richter's Predictor Modeling Earthquake Damage
## Creating Naive Bayes model to be used in blended ensemble model. 
# Kylie Foster
# Changing how has_secondary_use is encoded

# Useful websites:

## Loading required packages -----
#
library(klaR) #  for naive bayes
library(plyr) #  to rename factors using revalue
library(tidyverse)
library(caret) #  for all models
library(MLmetrics) #  additional metrics
library(tictoc) #  to check timing of models
library(e1071)

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

## Naive Bayes -----
set.seed(seed)
tic()
nb_model <- train(x = train_layer1[, -1],
                  y = train_layer1$damage_grade,
                  method = "nb", 
                  metric = metric,
                  trControl = my_control,
                  preProc = c("center", "scale"))
toc() # 694.33 sec elapsed

# Printing results
print(nb_model) 
# usekernel  F1       
# FALSE      0.5131132
# TRUE      0.6305333
# Plotting results
jpeg("nb_default.jpg")
plot(nb_model)
dev.off()
# compare predicted outcome and true outcome
#confusionMatrix(predict(rf_model), train_layer1$damage_grade)

# save the model to disk
saveRDS(nb_model, "./nb_model_default.rds")


