rm(list=ls())
outcome <- read.csv("C://TAMU//Statistical consulting//data explore//Graduation Outcome.csv")
dim(outcome)

stores <- read.csv("C://TAMU//Statistical consulting//data explore//Stores.csv")
dim(stores)
head(stores)
hist(stores$Store_Area)
hist(stores$Items_Available)
hist(stores$Daily_Customer_Count)
hist(stores$Store_Sales)
boxplot(stores$Store_Area)
boxplot(stores$Items_Available)
boxplot(stores$Daily_Customer_Count)
boxplot(stores$Store_Sales)

anyNA(stores)

cor(stores[,c(2,3,4)])
plot(stores$Daily_Customer_Count, stores$Store_Sales, main="Customer Count and Sales",
   xlab="Daily Customer Count ", ylab="Store Sales", pch=19)

plot(stores$Items_Available, stores$Store_Sales, main="Customer Count and Sales",
   xlab="Daily Customer Count ", ylab="Store Sales", pch=19)

comndeath <- read.csv("C://TAMU//Statistical consulting//Complications and Deaths.csv")
dim(comndeath)
names(comndeath)
head(comndeath)

library("readxl")
my_data <- read_excel("C://TAMU//Statistical consulting//data explore//tabn303.70.xls")
anyNA(my_data)
dim(my_data)
sum(is.na(my_data))#number of missing values
Hitters=na.omit(Hitters)#remove observations with missing values
dim(Hitters)
sum(is.na(Hitters))

