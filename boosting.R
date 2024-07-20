rm(list=ls())

library(gbm)
data <- iris               # reads the dataset

head(data)           # head() returns the top 6 rows of the dataframe

summary(data)       # returns the statistical summary of the data columns

dim(data)

parts = sample(1:nrow(data),120)
train = data[parts, ]
test = data[-parts, ]

# train a model using our training data
model_gbm = gbm(Species ~.,
              data = train,
              distribution = "multinomial",
              cv.folds = 10,
              shrinkage = .01,
              n.minobsinnode = 10,
              n.trees = 500)       # 500 tress to be built

summary(model_gbm)