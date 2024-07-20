# Load packages
if(!require(caret)){install.packages('caret');require(caret)}
if(!require(ggplot2)){install.packages('ggplot2');require(ggplot2)}
if(!require(lattice)){install.packages('lattice');require(lattice)}
if(!require(xgboost)){install.packages('xgboost');require(xgboost)}
if(!require(plyr)){install.packages('plyr');require(plyr)}

# Exploratory data analysis (EDA)

# Load class_data RData file
load("class_data.RData")

# Train

# Format data and set up trControl argument for later models for param tuning
Xtrain            <- x
Ytrain            <- factor(y, labels = c('No', 'Yes'))
YtrainRelevel     <- relevel(Ytrain, ref = 'Yes')
AllData           <- cbind(Xtrain, YtrainRelevel)
n                 <- nrow(Xtrain)

# Use k-fold CV for later models with tuning parameters
K                 <- 25
cvIndex           <- createFolds(YtrainRelevel, k = K, returnTrain = TRUE)
trControl         <- trainControl(method = "cv", index = cvIndex)

# Boosting

# Use k-fold CV for tuning parameters via trControl in formatData chunk
set.seed(7)
tuneGrid <- data.frame('nrounds'= c(100,200,300,400,500,600,700,800,900,1000),
                       'max_depth' = 6,
                       'eta' = 0.01,
                       'gamma' = 0,
                       'colsample_bytree' = 1,
                       'min_child_weight' = 0,
                       'subsample' = 1)
outBoost <- train(x = Xtrain, y = YtrainRelevel,
                  method = "xgbTree", 
                  verbose = 0,
                  tuneGrid = tuneGrid,
                  trControl = trControl)

# test error estimate using cvIndexTest

# test error estimate
set.seed(1)
KTest       <- 25
cvIndexTest <- createFolds(YtrainRelevel, k = KTest, returnTrain = FALSE)
fold.error  <- rep(0, KTest) ### place holder to save error for each fold
grid  <- data.frame(nrounds = outBoost$bestTune$nrounds,
                    max_depth = outBoost$bestTune$max_depth,
                    eta = outBoost$bestTune$eta,
                    gamma = outBoost$bestTune$gamma,
                    colsample_bytree = outBoost$bestTune$colsample_bytree,
                    min_child_weight = outBoost$bestTune$min_child_weight,
                    subsample = outBoost$bestTune$subsample)

for (j in 1:KTest){
  # fit boosting on K-1 fold of data
  boostFit <- train(YtrainRelevel~.,
                    data = AllData,
                    subset = unlist(cvIndexTest[-j]),
                    method = 'xgbTree',
                    tuneGrid = grid,
                    trControl = trainControl(method = "none"))
  # predict on 1 fold of data
  boostFit.pred <- predict(boostFit,
                           newdata = Xtrain[cvIndexTest[[j]], ],
                           nrounds = outBoost$bestTune$nrounds,
                           max_depth = outBoost$bestTune$max_depth,
                           eta = outBoost$bestTune$eta,
                           gamma = outBoost$bestTune$gamma,
                           colsample_bytree = 
                             outBoost$bestTune$colsample_bytree,
                           min_child_weight = 
                             outBoost$bestTune$min_child_weight,
                           subsample = outBoost$bestTune$subsample
  )
  # Misclassfication rate
  fold.error[j] <- sum(boostFit.pred != YtrainRelevel[cvIndexTest[[j]]])
}

### CV error rate is total misclassification rate
CV.error <- sum(fold.error)/n
testErrorBoost <- CV.error

# Final Output for Part 1: Supervised Learning
test_error <- testErrorBoost
ynew <- predict(boostFit,
                newdata          = xnew,
                nrounds          = outBoost$bestTune$nrounds,
                max_depth        = outBoost$bestTune$max_depth,
                eta              = outBoost$bestTune$eta,
                gamma            = outBoost$bestTune$gamma,
                colsample_bytree = outBoost$bestTune$colsample_bytree,
                min_child_weight = outBoost$bestTune$min_child_weight,
                subsample        = outBoost$bestTune$subsample
                )
# Convert back to 0-1 response 
ynew <- ifelse(ynew == "Yes", 1, 0)
save(ynew, test_error, file="26.RData")