library(randomForest)

train_labels <- read.csv("data/train_labels.csv")
train_values <- read.csv("data/train_values.csv")
test_values <- read.csv("data/test_values.csv")

train_data <- merge( train_values, train_labels, by = "building_id")

# standardize numerical predictors 
scl <- function(x){ (x - min(x))/(max(x) - min(x)) }
train_data[, 2:8] <- data.frame(lapply(train_data[, 2:8], scl))

# Create Random Forest Model
m.rf = randomForest(as.factor(damage_grade) ~ . -building_id -has_secondary_use - has_secondary_use_other - has_secondary_use_agriculture - has_secondary_use_hotel -has_secondary_use_rental -has_secondary_use_institution -has_secondary_use_school -has_secondary_use_industry -has_secondary_use_health_post -has_secondary_use_gov_office -has_secondary_use_use_police -plan_configuration -has_superstructure_other, data = train_data, importance = TRUE, ntree=1000, mtry=23)



cl.rf = predict(model, test_values, type="class")

# RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((as.numeric(true_ratings) - as.numeric(predicted_ratings))^2))
}
# Calculate RMSE
rf.rmse <- RMSE(cl.rf, train_labels$damage_grade)

# Calculate accuracy
rf.acc <- confusionMatrix(as.factor(cl.rf), as.factor(train_labels$damage_grade))$overall["Accuracy"]

# Plot variable importance
varImpPlot(model)

print(model)

# tune RF-model using CARET
library(caret)
set.seed(1337)
# split data for reduced tuning time
splitIndex <- createDataPartition(as.factor(train_data$damage_grade), p = .8, 
                                  list = FALSE, 
                                  times = 1)

control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(train_data))
tunegrid <- expand.grid(.mtry=c(10:25))
rf_default <- train(as.factor(damage_grade) ~ ., data=train_data[-splitIndex,], method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)

print(rf_default)
plot(rf_default)

# Create submission file
submit <- data.frame(building_id = test_values$building_id, damage_grade = cl.rf)
write.csv(submit, file = "output/forest_mtry23_standardized.csv", row.names = FALSE)


# Support Vector Machine
# install.packages( 'e1071' )
library( 'e1071' )

model_svm <- svm( as.factor(damage_grade)~.-building_id -has_secondary_use - has_secondary_use_other - has_secondary_use_agriculture - has_secondary_use_hotel -has_secondary_use_rental -has_secondary_use_institution -has_secondary_use_school -has_secondary_use_industry -has_secondary_use_health_post -has_secondary_use_gov_office -has_secondary_use_use_police -plan_configuration -has_superstructure_other, train_data )
res <- predict( model_svm, newdata=test_values )

# Calculate RMSE
svm.rmse <- RMSE(res, train_labels$damage_grade)

# Calculate accuracy
svm.acc <- confusionMatrix(as.factor(res), as.factor(train_labels$damage_grade))$overall["Accuracy"]

# Create submission file
submit_svm <- data.frame(building_id = test_values$building_id, damage_grade = res)
write.csv(submit_svm, file = "output/svm.csv", row.names = FALSE)
