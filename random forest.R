
rm(list=ls())

library(randomForest)
library(datasets)
install.packages("caret")
library(caret)
install.packages("rlang")
remove.packages("rlang")

data<-iris
str(data)

data$Species <- as.factor(data$Species)
table(data$Species)

set.seed(222)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
train <- data[ind==1,]
test <- data[ind==2,]

rf <- randomForest(Species~., data=train, proximity=TRUE) 
print(rf)

p1 <- predict(rf, train)
confusionMatrix(p1, train$ Species)

p2 <- predict(rf, test)
confusionMatrix(p2, test$ Species)

plot(rf)

t <- tuneRF(train[,-5], train[,5],
       stepFactor = 0.5,
       plot = TRUE,
       ntreeTry = 150,
       trace = TRUE,
       improve = 0.05)

hist(treesize(rf),
     main = "No. of Nodes for the Trees",
     col = "green")
Variable Importance
varImpPlot(rf,
           sort = T,
           n.var = 10,
           main = "Top 10 - Variable Importance")
importance(rf)
MeanDecreaseGini

partialPlot(rf, train, Petal.Width, "setosa")

MDSplot(rf, train$Species)

