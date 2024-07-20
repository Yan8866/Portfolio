# Load packages
rm(list=ls())

if(!require(tidyverse)){install.packages('tidyverse');require(tidyverse)}
if(!require(caret)){install.packages('caret');require(caret)}
if(!require(glmnet)){install.packages('glmnet');require(glmnet)}
if(!require(klaR)){install.packages('klaR');require(klaR)}
if(!require(ggplot2)){install.packages('ggplot2');require(ggplot2)}
if(!require(xgboost)){install.packages('xgboost');require(xgboost)}
if(!require(rpart)){install.packages('rpart');require(rpart)}
if(!require(randomForest)){install.packages('randomForest');require(randomForest)}

# Exploratory data analysis (EDA)

# Load class_data RData file
load("C://TAMU//639-data mining//final project//class_data.RData")

# Training data covariates
class(x)
dim(x)

# Training data response
class(y)
length(y)
table(y) # To identify class imbalance

# Testing data covariates
class(xnew)
dim(xnew)

# Check if there is any missing data in case we need to impute or
# drop rows/columns
anyNA(x)
anyNA(y)
anyNA(xnew)

# Check data structures of columns
str(x)
str(y)
str(xnew)

# Identify unique values for each column
uniqueX <- sapply(x, function(x){ length(unique(x)) })
uniqueXnew <- sapply(xnew, function(x){ length(unique(xnew)) })
uniqueY <- sapply(y, function(x){ length(unique(y)) })

# Train

# Format data and set up trControl argument for later models for parameter tuning
Xtrain            <- x
Ytrain            <- factor(y, labels = c('No', 'Yes'))
YtrainRelevel     <- relevel(Ytrain, ref = 'Yes')

XtrainMat         <- as.matrix(Xtrain)
n                 <- nrow(XtrainMat) ### sample size
AllData           <- cbind(Xtrain, YtrainRelevel)

XtrainCenterScale <- scale(Xtrain)

# Use k-fold CV for later models with tuning parameters
K                 <- 25
cvIndex           <- createFolds(YtrainRelevel, k = K, returnTrain = TRUE)
trControl         <- trainControl(method = "cv", index = cvIndex)

# Logistic regression

# No tuning parameters in logistic regression
set.seed(7)
outLogistic   <- train(x = Xtrain, y = YtrainRelevel, method = 'glm'
                       , trControl = trainControl(method = "none")
                       , family = 'binomial')

# test error estimate
set.seed(1)
KTest       <- 25
cvIndexTest <- createFolds(YtrainRelevel, k = KTest, returnTrain = FALSE)
fold.error  <- rep(0, KTest) ### place holder to save error for each fold

for (j in 1:KTest){
  # fit logistic on K-1 fold of data
  logisticFit <- glm(YtrainRelevel~.,
                     data = AllData,
                     subset = unlist(cvIndexTest[-j]),
                     family=binomial)
  # predict on 1 fold of data
  logistic.probs <- predict(logisticFit,
                            newdata=Xtrain[cvIndexTest[[j]], ],
                            type="response")
  logistic.pred <- ifelse(logistic.probs > 0.5, "Yes", "No")
  # Misclassfication rate
  fold.error[j] <- sum(logistic.pred != YtrainRelevel[cvIndexTest[[j]]])
}

### CV error rate is total misclassification rate
CV.error <- sum(fold.error)/n
testErrorLogistic <- CV.error

# LDA

# No tuning parameters in LDA
set.seed(7)
outLDA       <- train(x = Xtrain, y = YtrainRelevel, method = 'lda'
                      , trControl = trainControl(method = "none"))

# test error estimate using cvIndexTest
set.seed(1)
fold.error  <- rep(0, KTest) ### place holder to save error for each fold

for (j in 1:KTest){
  # fit lda on K-1 fold of data
  ldaFit <- train(YtrainRelevel~.,
                  data = AllData,
                  subset = unlist(cvIndexTest[-j]),
                  method = 'lda')
  # predict on 1 fold of data
  ldaFit.pred <- predict(ldaFit,
                         newdata=Xtrain[cvIndexTest[[j]], ])
  # Misclassfication rate
  fold.error[j] <- sum(ldaFit.pred != YtrainRelevel[cvIndexTest[[j]]])
}

### CV error rate is total misclassification rate
CV.error <- sum(fold.error)/n
testErrorLDA <- CV.error

# QDA

message("Due to the high number of predictors, we did not employ QDA.")

# Naive Bayes

# Use k-fold CV for tuning parameters via trControl in formatData chunk
set.seed(7)
grid        <- expand.grid(fL=0:1,
                           usekernel=c(TRUE, FALSE),
                           adjust=0:1)
outNB       <- train(x = Xtrain, y = YtrainRelevel, method = 'nb'
                     , trControl = trControl
                     , tuneGrid = grid)

# test error estimate using cvIndexTest
set.seed(1)
fold.error  <- rep(0, KTest) ### place holder to save error for each fold
grid  <- data.frame(fL = outNB$bestTune$fL,
                    usekernel = outNB$bestTune$usekernel,
                    adjust = outNB$bestTune$adjust)

for (j in 1:KTest){
  # fit nb on K-1 fold of data
  nbFit <- train(YtrainRelevel~.,
                 data = AllData,
                 subset = unlist(cvIndexTest[-j]),
                 method = 'nb',
                 tuneGrid = grid,
                 trControl = trainControl(method = "none"))
  # predict on 1 fold of data
  nbFit.pred <- predict(nbFit,
                        newdata=Xtrain[cvIndexTest[[j]], ],
                        fL = outNB$bestTune$fL,
                        usekernel = outNB$bestTune$usekernel,
                        adjust = outNB$bestTune$adjust
  )
  # Misclassfication rate
  fold.error[j] <- sum(nbFit.pred != YtrainRelevel[cvIndexTest[[j]]])
}

### CV error rate is total misclassification rate
CV.error <- sum(fold.error)/n
testErrorNB <- CV.error

# k-nearest neighbors (k-NN)

# Use k-fold CV for tuning parameters via trControl in formatData chunk
set.seed(7)
grid   <- expand.grid(k=c(seq(5, 100, 5)))
outKNN <- train(x = Xtrain, y = YtrainRelevel
                , method = 'knn'
                , trControl = trControl
                , preProcess = c('center', 'scale')
                , tuneGrid = grid)

# test error estimate using cvIndexTest
set.seed(1)
fold.error <- rep(0, KTest) ### place holder to save error for each fold
grid       <- data.frame(k=outKNN$bestTune$k)

for (j in 1:KTest){
  # fit k-NN on K-1 fold of data
  kNNFit <- train(YtrainRelevel~.,
                  data = AllData,
                  subset = unlist(cvIndexTest[-j]),
                  method = 'knn',
                  tuneGrid = grid,
                  preProcess = c('center', 'scale'),
                  trControl = trainControl(method = "none"))
  # predict on 1 fold of data
  kNNFit.pred <- predict(kNNFit,
                         newdata=XtrainCenterScale[cvIndexTest[[j]], ],
                         k = outKNN$bestTune$k
  )
  # Misclassfication rate
  fold.error[j] <- sum(kNNFit.pred != YtrainRelevel[cvIndexTest[[j]]])
}

### CV error rate is total misclassification rate
CV.error <- sum(fold.error)/n
testErrorKNN <- CV.error

# SVM (radial and linear)

# Use k-fold CV for tuning parameters via trControl in formatData chunk
set.seed(7)
pairWiseDist <- dist(scale(Xtrain), method = 'euclidean')**2

sigmaRange   <- quantile(pairWiseDist, c(0.9,0.5,0.1))
hist(pairWiseDist)
abline(v = sigmaRange[1])
abline(v = sigmaRange[2])
abline(v = sigmaRange[3])

tuneGrid     <- expand.grid(C = c(0.001, 0.01, 0.1, 0.5, 0.75, 1, 1.25, 1.5)
                            , sigma  = 1/sigmaRange)
outSVMRadial <- train(x = Xtrain, y = YtrainRelevel, 
                      method = "svmRadial", 
                      tuneGrid = tuneGrid,
                      preProc = c("center", "scale"),
                      trControl = trControl)

# test error estimate using cvIndexTest
set.seed(1)
fold.error  <- rep(0, KTest) ### place holder to save error for each fold
grid  <- data.frame(sigma = outSVMRadial$bestTune$sigma,
                    C = outSVMRadial$bestTune$C)

for (j in 1:KTest){
  # fit svm radial on K-1 fold of data
  svmRadialFit <- train(YtrainRelevel~.,
                        data = AllData,
                        subset = unlist(cvIndexTest[-j]),
                        method = 'svmRadial',
                        tuneGrid = grid,
                        preProc = c("center", "scale"),
                        trControl = trainControl(method = "none"))
  # predict on 1 fold of data
  svmRadialFit.pred <- predict(svmRadialFit,
                               newdata=XtrainCenterScale[cvIndexTest[[j]], ],
                               sigma = outSVMRadial$bestTune$sigma,
                               C = outSVMRadial$bestTune$C
  )
  # Misclassfication rate
  fold.error[j] <- sum(svmRadialFit.pred != YtrainRelevel[cvIndexTest[[j]]])
}

### CV error rate is total misclassification rate
CV.error <- sum(fold.error)/n
testErrorSVMRadial <- CV.error
rm(pairWiseDist)

# Use k-fold CV for tuning parameters via trControl in formatData chunk
set.seed(7)
tuneGrid     <- expand.grid( C = c(0.001, 0.01, 0.1, 5, 10, 25, 50))
outSVMLinear <- train(x = Xtrain, y = YtrainRelevel, 
                      method = "svmLinear", 
                      tuneGrid = tuneGrid,
                      preProc = c("center", "scale"),
                      trControl = trControl)

# test error estimate using cvIndexTest
set.seed(1)
fold.error  <- rep(0, KTest) ### place holder to save error for each fold
grid  <- data.frame(C = outSVMLinear$bestTune$C)

for (j in 1:KTest){
  # fit svm linear on K-1 fold of data
  svmLinearFit <- train(YtrainRelevel~.,
                        data = AllData,
                        subset = unlist(cvIndexTest[-j]),
                        method = 'svmLinear',
                        tuneGrid = grid,
                        preProc = c("center", "scale"),
                        trControl = trainControl(method = "none"))
  # predict on 1 fold of data
  svmLinearFit.pred <- predict(svmLinearFit,
                               newdata=XtrainCenterScale[cvIndexTest[[j]], ],
                               C = svmLinearFit$bestTune$C
  )
  # Misclassfication rate
  fold.error[j] <- sum(svmLinearFit.pred != YtrainRelevel[cvIndexTest[[j]]])
}

### CV error rate is total misclassification rate
CV.error <- sum(fold.error)/n
testErrorSVMLinear <- CV.error

# Logistic Elastic Net

# Use k-fold CV for tuning parameters via trControl in formatData chunk
set.seed(7)
tuneGrid   <- expand.grid('alpha'=c(0, 0.25, 0.5, 0.75, 1)
                          ,'lambda' = seq(1e-6, 0.25, length = 50))
outElastic <- train(x = Xtrain, y = YtrainRelevel,
                    method = "glmnet",
                    preProc = c("center", "scale"),
                    trControl = trControl,
                    tuneGrid = tuneGrid)

# test error estimate using cvIndexTest
set.seed(1)
fold.error  <- rep(0, KTest) ### place holder to save error for each fold
grid  <- data.frame(alpha = outElastic$bestTune$alpha,
                    lambda = outElastic$bestTune$lambda)

for (j in 1:KTest){
  # fit elastic net on K-1 fold of data
  elasticFit <- train(YtrainRelevel~.,
                      data = AllData,
                      subset = unlist(cvIndexTest[-j]),
                      method = 'glmnet',
                      tuneGrid = grid,
                      preProc = c("center", "scale"),
                      trControl = trainControl(method = "none"))
  # predict on 1 fold of data
  elasticFit.pred <- predict(elasticFit,
                             newdata=XtrainCenterScale[cvIndexTest[[j]], ],
                             alpha = outElastic$bestTune$alpha,
                             lambda = outElastic$bestTune$lambda
  )
  # Misclassfication rate
  fold.error[j] <- sum(elasticFit.pred != YtrainRelevel[cvIndexTest[[j]]])
}

### CV error rate is total misclassification rate
CV.error <- sum(fold.error)/n
testErrorElastic <- CV.error

# Pruned Tree

# Use k-fold CV for tuning parameters via trControl in formatData chunk
set.seed(7)
tuneGrid <- expand.grid(cp = c(0.001, 0.01, 0.1))
outRPart <- train(x = Xtrain, y = YtrainRelevel,
                  method = "rpart",
                  tuneGrid = tuneGrid,
                  trControl = trControl)
plot(outRPart$finalModel,margin= rep(.1,4))
text(outRPart$finalModel, cex = 0.4, digits = 1)

# test error estimate using cvIndexTest
set.seed(1)
fold.error  <- rep(0, KTest) ### place holder to save error for each fold
grid  <- data.frame(cp = outRPart$bestTune$cp)

for (j in 1:KTest){
  # fit pruned tree on K-1 fold of data
  rPartFit <- train(YtrainRelevel~.,
                    data = AllData,
                    subset = unlist(cvIndexTest[-j]),
                    method = 'rpart',
                    tuneGrid = grid,
                    trControl = trainControl(method = "none"))
  # predict on 1 fold of data
  rPartFit.pred <- predict(rPartFit,
                           newdata=Xtrain[cvIndexTest[[j]], ],
                           cp = outRPart$bestTune$cp
  )
  # Misclassfication rate
  fold.error[j] <- sum(rPartFit.pred != YtrainRelevel[cvIndexTest[[j]]])
}

### CV error rate is total misclassification rate
CV.error <- sum(fold.error)/n
testErrorRPart<- CV.error

# Bagging

# No tuning parameters in bagging
set.seed(7)
tuneGridBag <- data.frame(mtry = ncol(Xtrain)) # number of random features is p
outBag      <- train(x = Xtrain, y = YtrainRelevel,
                     method = "rf",
                     tuneGrid = tuneGridBag,
                     trControl = trainControl(method = "none"))

# test error estimate using cvIndexTest
set.seed(1)
fold.error  <- rep(0, KTest) ### place holder to save error for each fold

for (j in 1:KTest){
  # fit bagging on K-1 fold of data
  bagFit <- train(YtrainRelevel~.,
                  data = AllData,
                  subset = unlist(cvIndexTest[-j]),
                  method = 'rf',
                  tuneGrid = tuneGridBag,
                  trControl = trainControl(method = "none"))
  # predict on 1 fold of data
  bagFit.pred <- predict(bagFit,
                         newdata=Xtrain[cvIndexTest[[j]], ],
                         mtry = ncol(Xtrain)
  )
  # Misclassfication rate
  fold.error[j] <- sum(bagFit.pred != YtrainRelevel[cvIndexTest[[j]]])
}

### CV error rate is total misclassification rate
CV.error <- sum(fold.error)/n
testErrorBag <- CV.error

# Random Forest

# Use k-fold CV for tuning parameters via trControl in formatData chunk
set.seed(7)
tuneGridRf <- data.frame(mtry = c(round(sqrt(ncol(Xtrain))), 10, 50))
outRF      <- train(x = Xtrain, y = YtrainRelevel,
                    method = "rf",
                    tuneGrid = tuneGridRf,
                    trControl = trControl)

# test error estimate using cvIndexTest
set.seed(1)
fold.error  <- rep(0, KTest) ### place holder to save error for each fold
grid  <- data.frame(mtry = outRF$bestTune$mtry)

for (j in 1:KTest){
  # fit random forest on K-1 fold of data
  rFFit <- train(YtrainRelevel~.,
                 data = AllData,
                 subset = unlist(cvIndexTest[-j]),
                 method = 'rf',
                 tuneGrid = grid,
                 trControl = trainControl(method = "none"))
  # predict on 1 fold of data
  rFFit.pred <- predict(rFFit,
                        newdata=Xtrain[cvIndexTest[[j]], ],
                        mtry = outRF$bestTune$mtry
  )
  # Misclassfication rate
  fold.error[j] <- sum(rFFit.pred != YtrainRelevel[cvIndexTest[[j]]])
}

### CV error rate is total misclassification rate
CV.error <- sum(fold.error)/n
testErrorRF <- CV.error

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
set.seed(1)
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
TestErrorsDF <- data.frame(modelName = c('Logistic',
                                         'LDA',
                                         'NB',
                                         'K-NN',
                                         'SVMRadial',
                                         'SVMLinear',
                                         'Elastic Net',
                                         'PrunedTree',
                                         'BagTrees',
                                         'RandomForest',
                                         'BoostTrees'
                                         ),
                           testError = c(testErrorLogistic,
                                         testErrorLDA,
                                         testErrorNB,
                                         testErrorKNN,
                                         testErrorSVMRadial,
                                         testErrorSVMLinear,
                                         testErrorElastic,
                                         testErrorRPart,
                                         testErrorBag,
                                         testErrorRF,
                                         testErrorBoost
                                         )
                           )
minTestErrorDF <- TestErrorsDF %>% 
  filter(testError == min(testError))
rm(y)


# Part 2: Unsupervised Learning
rm(list=ls())


# Load custer_data RData file
load("C://TAMU//639-data mining//final project//cluster_data.RData")

# Clustering data
class(y)
n <- nrow(y)
p <- ncol(y)

# Check if there is any missing data in case we need to impute or
# drop rows/columns
anyNA(y)

# k-means clustering
set.seed(7)
y <- scale(y)

# Create a helper function to calculate optimum cluster number, K, via CH Index
ch.index <- function(x, kmax, iter.max=100, nstart=10, algorithm="Lloyd"){
  ch <- numeric(length=kmax-1)
  n  <- nrow(x)
  for (k in 2:kmax) {
    a       <- kmeans(x,
                      k,
                      iter.max=iter.max,
                      nstart=nstart,
                      algorithm=algorithm
                      )
    w       <- a$tot.withinss
    b       <- a$betweenss
    ch[k-1] <- (b/(k - 1)) / (w/(n - k))
  }
  return(list(k=2:kmax, ch=ch))
}

kMax       <- 50
outCHIndex <- ch.index(y, kMax)
kBest      <- outCHIndex$k[which.max(outCHIndex$ch)]
plot(unlist(outCHIndex$k)
     , unlist(outCHIndex$ch)
     , main="CH Plot"
     , xlab="K"
     , ylab="CH")

# Iterate with different starting values and pick lowest total within cluster variation
nstart <- 1:50
# place holder to save total within cluster variation for each starting value
nstart.withinClusterVar <- rep(0, length(nstart))
kmeansCluster           <- vector(mode = "list", length=length(nstart))
for (j in 1:length(nstart.withinClusterVar)){
  # Run K-means with specific starting points
  kmeans.out <- kmeans(y, centers=kBest, nstart=j)
  # Store total within cluster variation
  nstart.withinClusterVar[j] <- kmeans.out$tot.withinss
  kmeansCluster[[j]] <- kmeans.out$cluster
}

nstartBest  <- nstart[which.min(nstart.withinClusterVar)]
# Final Model Cluster
kmeans.bestCluster <- kmeansCluster[[which.min(nstart.withinClusterVar)]]

# Plot in 2 dimensions
plot(y, col=(kmeans.bestCluster + 1), xlab="", ylab="", pch=20, cex=2)
message(sprintf("The ideal number of clusters identified: %s\n", kBest))

### Gaussian mixture model 
# first we apply PCA to our data 
pr.out=prcomp(y, scale=TRUE)

# plot Cumulative Proportion of Variance Explained
pr.var=pr.out$sdev^2
pve=pr.var/sum(pr.var)
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')

# select the first 80 PC and fit model1
selectedPC <- pr.out$x[,1:80]
pr.var1=(pr.out$sdev[1:80])^2
pve1=pr.var1/sum(pr.var1)
plot(pve1, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve1), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')

# do Gaussian mixture model based clustering 
library(mclust)
mod1 <- Mclust(selectedPC, G = 3, modelNames = c("EII","VII","EVI","VEI"))
summary(mod1)

# select the first 40 PCs and fit model2
selectedPC2 <- pr.out$x[,1:40]
pr.var2=(pr.out$sdev[1:40])^2
pve2=pr.var2/sum(pr.var2)
plot(pve2, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve2), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')

# do Gaussian mixture model based clustering 
library(mclust)
mod2 <- Mclust(selectedPC2, G = 3, modelNames = c("EII","VII","EVI","VEI"))
summary(mod2)




