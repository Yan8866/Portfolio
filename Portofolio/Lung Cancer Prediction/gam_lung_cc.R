#### kaggle lung cancer data 

# Clear the workspace and load the libraries
rm(list = ls())
library(mgcv)

# load the data
lung_cc <- read.csv("C://TAMU//Statistical consulting//project//lung_cancer.csv")

# EDA
dim(lung_cc)
head(lung_cc)
hist(lung_cc$AGE, main = "Distribution of Age", xlab="Age")

lung_cc$LUNG_CANCER <-ifelse(lung_cc$LUNG_CANCER == "YES",1,0)
unique(lung_cc$LUNG_CANCER)
lung_cc$LUNG_CANCER[1:10]
plot(lung_cc$AGE,lung_cc$LUNG_CANCER,xlab="Age", ylab="Cancer")

# preprocessing data 

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

# check missing values 
anyNA(lung_cc)     # no missing values 
table(lung_cc$LUNG_CANCER)   # To identify class imbalance

# check outliers in age 
par(mfrow=c(1,1))
boxplot(lung_cc$AGE, main="Age Distribution")

# check differences between males and females 
male_lcc <- lung_cc[lung_cc$GENDER=="M",]
fem_lcc <- lung_cc[lung_cc$GENDER=="F",]
table(male_lcc$LUNG_CANCER)
table(fem_lcc$LUNG_CANCER)
lung_cc[lung_cc$AGE<40,]

x1 <- c(145/(145+17),17/(145+17))
lbls <- c("lung cancer","no cancer")
x2 <- c(125/(125+22),22/(125+22))

par(mfrow=c(2,1))
pie(x1,labels=lbls, col=rainbow(length(lbls)),main="Male")
pie(x2,labels=lbls, col=rainbow(length(lbls)),main="Female")

#check differences between smokers and nonsmokers 
smokers_lcc <- lung_cc[lung_cc$SMOKING == 1,]
nonsmokers_lcc <- lung_cc[lung_cc$SMOKING == 0,]
table(smokers_lcc$LUNG_CANCER)
table(nonsmokers_lcc$LUNG_CANCER)

x3 <- c(155/(155+20),20/(155+20))
x4 <- c(115/(115+20),20/(115+20))
pie(x3,labels=lbls, col=rainbow(length(lbls)),main="Smokers")
pie(x4,labels=lbls, col=rainbow(length(lbls)),main="Nonsmokers")

# check if correlation between predictors 
cor(lung_cc[,2:16])
length(male_lcc$LUNG_CANCER)
length(fem_lcc$LUNG_CANCER)

# a logistic regression with age as the predictor 
LG_age <- glm(LUNG_CANCER ~ AGE,family = binomial(link='logit'),data = lung_cc)
summary(LG_age)

# the fitted line and the confidence band for the fit

par(mfrow=c(1,1))
AGE <- lung_cc$AGE
newAge = seq(min(AGE),max(AGE),2)
fHat_CI = predict(LG_age,list(AGE=newAge),
                          type="response", se.fit=TRUE)
fHat = fHat_CI$fit
upCIfHat = fHat + 1.96*fHat_CI$se.fit
lowCIfHat = fHat - 1.96*fHat_CI$se.fit

plot(lung_cc$AGE,lung_cc$LUNG_CANCER,xlab="Age", ylab = "Prob of Lung Cancer", main="Lung Cancer",
     pch=16,col="grey",lwd=2,xlim=c(21,87), ylim=c(0,1))
polygon(x=c(newAge, rev(newAge)),
        y=c(upCIfHat, rev(lowCIfHat)),
        col="lightblue",border=NA)
lines(newAge,fHat,
      col = "blue",type="l",lwd=3)


# test whether the fit is linear or quadratic versus the need to do a semiparametric fit
age2 <- AGE^2
qfit <- glm(LUNG_CANCER ~ AGE + age2,family = binomial(link='logit'),data = lung_cc)
semifit <- gam(LUNG_CANCER ~ s(AGE, k=27),family = binomial(link='logit'),data = lung_cc)
anova(LG_age,qfit, semifit, test="Chisq")


# Fit a logistic gam with all the predictors but only age modeled as a spline.

loggam <-gam(LUNG_CANCER ~ GENDER + AGE+ SMOKING + YELLOW_FINGERS + ANXIETY + PEER_PRESSURE
             + CHRONIC.DISEASE + FATIGUE   +ALLERGY   + WHEEZING  + ALCOHOL.CONSUMING + COUGHING
             + SHORTNESS.OF.BREATH + SWALLOWING.DIFFICULTY + CHEST.PAIN, 
             family = binomial(link='logit'),data = lung_cc)

loggamsp <-gam(LUNG_CANCER ~ GENDER + s(AGE,k=27) + SMOKING + YELLOW_FINGERS + ANXIETY + PEER_PRESSURE
             + CHRONIC.DISEASE + FATIGUE   +ALLERGY   + WHEEZING  + ALCOHOL.CONSUMING + COUGHING
             + SHORTNESS.OF.BREATH + SWALLOWING.DIFFICULTY + CHEST.PAIN, 
             family = binomial(link='logit'),data = lung_cc)
summary(loggam)
summary(loggamsp)

# check if the spline is needed
anova(loggam,loggamsp,test="Chisq")

# fit a logistic regression with selected variables 
loggam_select <-gam(LUNG_CANCER ~ s(AGE,k=27) + SMOKING + PEER_PRESSURE
             + CHRONIC.DISEASE + FATIGUE + ALLERGY + COUGHING + SWALLOWING.DIFFICULTY, 
             family = binomial(link='logit'),data = lung_cc)
summary(loggam_select)

### stepwise regression
detach("package:mgcv",unload=TRUE)
library(gam)
fitInit = gam::gam(LUNG_CANCER ~ as.factor(GENDER)+ AGE + YELLOW_FINGERS + ANXIETY + PEER_PRESSURE
             + CHRONIC.DISEASE + FATIGUE + ALLERGY + WHEEZING  + ALCOHOL.CONSUMING + COUGHING
             + SHORTNESS.OF.BREATH + SWALLOWING.DIFFICULTY + CHEST.PAIN + as.factor(SMOKING),
              family = binomial,data = lung_cc)
summary(fitInit)

# Now code and run the stepwise regression
stepFit = gam::step.Gam(fitInit,scope =
                      list("GENDER"     = ~1 + GENDER,
                           "YELLOW_FINGERS" = ~1 + YELLOW_FINGERS,
                           "AGE"      = ~1 + AGE + s(AGE,5),
                           "ANXIETY"   = ~1 + ANXIETY,
                           "PEER_PRESSURE"   = ~1 + PEER_PRESSURE,
                           "CHRONIC.DISEASE"   = ~1 + CHRONIC.DISEASE,
                           "FATIGUE"   = ~1 + FATIGUE,
                           "ALLERGY"   = ~1 + ALLERGY,
                           "WHEEZING"   = ~1 + WHEEZING,
                           "ALCOHOL.CONSUMING"   = ~1 + ALCOHOL.CONSUMING,
                           "COUGHING"   = ~1 + COUGHING,
                           "SHORTNESS.OF.BREATH"   = ~1 + SHORTNESS.OF.BREATH,
                           "SWALLOWING.DIFFICULTY"   = ~1 + SWALLOWING.DIFFICULTY,
                           "CHEST.PAIN"   = ~1 + CHEST.PAIN,
                           "SMOKING"   = ~1 + SMOKING), 
                           family = binomial,data = lung_cc)

# the final model
print(names(stepFit$"model")[-1])

detach("package:gam",unload=TRUE)
library(mgcv)
gamfinal <- gam(LUNG_CANCER ~  YELLOW_FINGERS + PEER_PRESSURE
             + CHRONIC.DISEASE + FATIGUE + ALLERGY + ALCOHOL.CONSUMING + COUGHING
             + SWALLOWING.DIFFICULTY  + as.factor(SMOKING),
              family = binomial,data = lung_cc)
summary(gamfinal)


# calculate AIC and BIC of the fitted models
install.packages("vctrs")
library(vctrs)
library(broom)
broom::glance(loggam) 
broom::glance(loggamsp) 
broom::glance(gamfinal) 


cor(lung_cc$YELLOW_FINGERS,lung_cc$SMOKING)
