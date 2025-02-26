### lung cancer predictive models
# clear the working space
rm(list=ls())

# load data and libraries 
lung_cc <- read.csv("C://TAMU//Statistical consulting//project//lung_cancer.csv")
library(ISLR)
library(MASS)

# preprocessing data 
lung_cc$LUNG_CANCER <-ifelse(lung_cc$LUNG_CANCER == "YES",1,0)
lung_cc$GENDER <- ifelse(lung_cc$GENDER=="M",1,0)
plot(lung_cc$AGE,lung_cc$LUNG_CANCER,xlab="Age", ylab="Cancer", main= "Lung Cancer VS Age")

lung_cc$SMOKING <-ifelse(lung_cc$SMOKING == "2",1,0)
lung_cc$YELLOW_FINGERS <-ifelse(lung_cc$YELLOW_FINGERS == "2",1,0)
lung_cc$ANXIETY <-ifelse(lung_cc$ANXIETY == "2",1,0)
lung_cc$PEER_PRESSURE <-ifelse(lung_cc$PEER_PRESSURE == "2",1,0)
lung_cc$CHRONIC.DISEASE <-ifelse(lung_cc$CHRONIC.DISEASE == "2",1,0)
lung_cc$FATIGUE <-ifelse(lung_cc$FATIGUE == "2",1,0)
lung_cc$ALLERGY <-ifelse(lung_cc$ALLERGY == "2",1,0)
lung_cc$WHEEZING <-ifelse(lung_cc$WHEEZING == "2",1,0)
lung_cc$ALCOHOL.CONSUMING <-ifelse(lung_cc$ALCOHOL.CONSUMING == "2",1,0)
lung_cc$COUGHING <-ifelse(lung_cc$COUGHING == "2",1,0)
lung_cc$SHORTNESS.OF.BREATH <-ifelse(lung_cc$SHORTNESS.OF.BREATH == "2",1,0)
lung_cc$SWALLOWING.DIFFICULTY <-ifelse(lung_cc$SWALLOWING.DIFFICULTY == "2",1,0)
lung_cc$CHEST.PAIN <-ifelse(lung_cc$CHEST.PAIN == "2",1,0)


# splitting the data into training set and testing set 
set.seed(1)
train=sample(1:nrow(lung_cc),250) # by default sample is without replacement, : means sequence
cancer.test=lung_cc$LUNG_CANCER[-train]

slices <- c(250, 309-250)
lbls <- c("Train", "Test")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels
pie(slices,labels = lbls, col=rainbow(length(lbls)),
   main="Pie Chart of Data Splitting")


### logistic regression 
set.seed(1)
train=sample(1:nrow(lung_cc),250) # by default sample is without replacement, : means sequence
glm.fit=glm(LUNG_CANCER ~.-LUNG_CANCER, data=lung_cc,family=binomial, subset=train)
glm.probs=predict(glm.fit,newdata=lung_cc[-train,1:15],type="response") 
glm.pred=ifelse(glm.probs >0.5,"1","0")
table(glm.pred,cancer.test)
mean(glm.pred==cancer.test)
LG_err = 1 - mean(glm.pred==cancer.test)
summary(glm.fit)
anova(glm.fit)

### tree based models
library(tree)
library(randomForest)
library(gbm)

# let's grow the tree on the training set, and evaluate its performance on the test set.
set.seed(1)
lung_cc$LUNG_CANCER <- as.factor(lung_cc$LUNG_CANCER)
train=sample(1:nrow(lung_cc),250) # by default sample is without replacement
tree.lcc=tree(LUNG_CANCER ~.-LUNG_CANCER,data = lung_cc, subset=train)
plot(tree.lcc)
text(tree.lcc,pretty=0)
tree.pred=predict(tree.lcc,newdata = lung_cc[-train,1:15],type="class")   # type="class" returns the labels of the class
table(tree.pred, lung_cc[-train,]$LUNG_CANCER)
tree_err= 1- mean(tree.pred==lung_cc$LUNG_CANCER[-train])

# This tree was grown to full depth, and might be too variable. 
# We now use CV to prune it.
set.seed(1)
#?cv.tree
cv.lcc=cv.tree(tree.lcc,FUN=prune.misclass)  # use misclassification error rate to prune a tree
plot(cv.lcc)
prune.lcc=prune.misclass(tree.lcc,best=6)
#prune.carseats=prune.misclass(tree.carseats,k=0.5) #this gives the same result; k is the tuning parameter alpha in the slides
plot(prune.lcc);text(prune.lcc,pretty=0)

# Now lets evaluate this pruned tree on the test data.
tree.pred=predict(prune.lcc,lung_cc[-train,1:15],type="class")
table(tree.pred, lung_cc[-train,]$LUNG_CANCER)
prunedtree_err = 1 - mean(tree.pred==lung_cc$LUNG_CANCER[-train])

### random forest 
set.seed(2)
lung_cc$LUNG_CANCER <- as.factor(lung_cc$LUNG_CANCER)
rf.lcc=randomForest(LUNG_CANCER ~.,data=lung_cc,subset=train, mtry=3, ntree=10000,importance = TRUE)
print(rf.lcc)
plot(rf.lcc, type="l", main="Random Forest")

# check importance of all predictors 
importance(rf.lcc)
varImpPlot(rf.lcc)

# make predictions 
rf.pred <- predict(rf.lcc,newdata=lung_cc[-train,1:15], type="response")
table(rf.pred, lung_cc[-train,]$LUNG_CANCER)
rf_err = 1 - mean(rf.pred==lung_cc$LUNG_CANCER[-train])

# check the margins 
plot(margin(rf.lcc,lung_cc[-train,]$LUNG_CANCER),main="The Margin of Predictions")

# tune mtry 
tune.rf <- tuneRF(lung_cc[,-16],lung_cc[,16], stepFactor=0.5)
print(tune.rf)

## random forest with mtry=6
rf.lcc_m6=randomForest(LUNG_CANCER ~.,data=lung_cc,subset=train, mtry=6, ntree=10000,importance = TRUE)
print(rf.lcc_m6)

plot(rf.lcc_m6, type="l", main="Random Forest")
importance(rf.lcc_m6)
varImpPlot(rf.lcc_m6)
rf.pred1 <- predict(rf.lcc_m6,newdata=lung_cc[-train,1:15], type="response")
table(rf.pred1, lung_cc[-train,]$LUNG_CANCER)
rf_err1 = 1 - mean(rf.pred1==lung_cc$LUNG_CANCER[-train])


# summary 
TestErrorsDF <- data.frame(modelName = c('Logistic',
                                         'tree',
                                         'PrunedTree',
                                         'RandomForest'),
                           testError = c(LG_err,
                                         tree_err,
                                         prunedtree_err,
                                         rf_err)
                           )


# calculate AIC and BIC of the Logistic Regression model
library(broom)
broom::glance(glm.fit) 

# sampling bias
sum(lung_cc$AGE>40); 3/309
sum(lung_cc$AGE>50); 294/309
sum(lung_cc$AGE>60); 186/309
dim(lung_cc)
sum(lung_cc$LUNG_CANCER == 1); 270/309
