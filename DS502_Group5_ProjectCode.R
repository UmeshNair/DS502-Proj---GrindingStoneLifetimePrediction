# DS502 Group 5
# Valerie Tuzel, ML Tlachac, Umesh Nair, Mo Cheng

library(dplyr)
library(ggplot2)
library(leaps) # For Best Subset Selection
library(rpart) #for rpart
library(glmnet) #for lasso.cv and ridge
library(pls) # For PCR
set.seed(2)

stones <- read.csv("stones2.csv")
str(stones)
stones <- stones[,-c(3,23,24)]
stones$Car <- as.factor(stones$Car)
levels(stones$Carriage) <- c("Crg1","Crg2","Crg3","Crg4","Crg5","Crg6",
                             "Crg7","Crg8","Crg9","Crg10","Crg11","Crg12")
str(stones)
# plot the CalcTime vs CalcMiles
ggplot(stones, aes(x = CalcMiles, y= CalcTime, color = Car)) + 
  geom_point() +
  labs(title = "Lifetime vs. Distance") +
  theme(plot.title = element_text(hjust = 0.5)) +
  xlab("Distance") +
  ylab("Lifetime")


# plot the CalcTime vs AverageAmpSetpoint
ggplot(stones, aes(x = AverageAmpSetpoint, y= CalcTime, color = Car)) + 
  geom_point() +
  labs(title = "Lifetime vs. Average Programmed Amps") +
  theme(plot.title = element_text(hjust = 0.5))+
  xlab("Average Programmed Amps") +
  ylab("Lifetime")

# plot the CalcTime vs OverSetpoint
ggplot(stones, aes(x = OverSetpoint, y= CalcTime, color = Car)) + 
  geom_point() +
  labs(title = "Lifetime vs. Actual Amps / Programmed Amps") +
  theme(plot.title = element_text(hjust = 0.5))+
  xlab("Actual Amps / Programmed Amps") +
  ylab("Lifetime")

# plot the CalcTime vs Car
ggplot(stones, aes(x = Car, y= CalcTime)) + 
  geom_point() +
  labs(title = "CalcTime vs. Car") +
  theme(plot.title = element_text(hjust = 0.5))


ggplot(stones, aes(x = Year, y= CalcTime)) + 
  geom_point() +
  labs(title = "CalcTime vs. Year") +
  theme(plot.title = element_text(hjust = 0.5)) 

ggplot(stones, aes(x = Month, y= CalcTime)) + 
  geom_point() +
  labs(title = "CalcTime vs. Month") +
  theme(plot.title = element_text(hjust = 0.5)) 


stones %>% 
  summarize(sum(Range7Percent == 0))
stones %>% 
  summarize(sum(Range5SP == 0))

set.seed(2)
sample_rows <- sample(nrow(stones), nrow(stones) * 0.65)
# Make those train set
s_train <- stones[sample_rows, ]
# Rest is the test set
s_test <- stones[-sample_rows, ]
dim(s_train)
dim(s_test)


#############################################################


##########################
#linear regression model 0
##########################

fmlad<- CalcTime ~ CalcMiles

modeldis <- lm(fmlad , data= s_train)
summary(modeldis)
# Predictions on test data
preddis<- predict(modeldis, s_test) 
# RMSE Calculation
resdis <- s_test$CalcTime - preddis
(rmsedis <- sqrt(mean(resdis^2)))


# plot the results
ggplot(s_test, aes(x = preddis, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "blue") +
  labs(title = "Linear Regression Model with only Distance - Actual vs.Prediction ") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")+
  ylab("Actual Lifetime")
res_dis<- resdis
res_dis <-data.frame(res_dis)

# Draw a density plot of residuals
ggplot(res_dis, aes(res_dis)) +
  # Add a density curve
  geom_density(color = "blue") +
  # Add a vertical line through zero
  geom_vline(xintercept = 0) +
  xlab("residuals")

##########################
#linear regression model 1
##########################

model_lm <- lm(CalcTime ~ . , data= s_train)
summary(model_lm)
#plot(model_lm)
#pred_tlm <- predict.lm(model_lm)

length(model_lm$coefficients) > model_lm$rank
#library(car)
#vif(model_lm, digits = 3)

# Predictions on test data
predlm <- predict(model_lm, s_test) 
summary(predlm)

# RMSE Calculation
res <- s_test$CalcTime - predlm
(rmse <- sqrt(mean(res^2)))


# plot the results
ggplot(s_test, aes(x = predlm, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "blue") +
  labs(title = "Linear Regression Model All predictors - Actual vs.Prediction ") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")+
  ylab("Actual Lifetime")



resd<- res
resd <-data.frame(resd)
# Draw a density plot of residuals
ggplot(resd, aes(resd)) +
  # Add a density curve
  geom_density(color = "blue") +
  # Add a vertical line through zero
  geom_vline(xintercept = 0) +
  xlab("residuals")


##########################
#linear regression model 2
##########################

#######################
# Best Subset Selection
#######################
# User defined predict function for regsubsets
predict.regsubsets <- function(object, newdata, id, ...) {
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  xvars <- names(coefi)
  mat[, xvars] %*% coefi
}

# Best Subset Selection Using 10-fold Cross-Validation
# Total 22 variables so choose nvmax=21
set.seed(2)
vmax = 18
k = 10
folds <- sample(1:k, nrow(s_train), replace = TRUE)
cv.errors <- matrix(NA, k, vmax, dimnames = list(NULL, paste(1:vmax))) 
for (j in 1:k) {
  best.fit <- regsubsets(CalcTime ~ .-Range5SP-Carriage-Car, data = s_train[folds != j, ], nvmax = vmax) 
  for (i in 1:vmax) {
    pred <- predict(best.fit, s_train[folds == j, ], id = i)
    cv.errors[j, i] <- mean((s_train$CalcTime[folds == j] - pred)^2)
  }
}

(mean.cv.errors <- apply(cv.errors, 2, mean))
plot(mean.cv.errors, type = "b", xlab = "Number of Variables", 
     ylab = "Mean Cross-Validation Error")

bestsubset = which.min(mean.cv.errors)
points(bestsubset,mean.cv.errors[bestsubset],pch=20,col="red",cex=2)

reg_best <- regsubsets(CalcTime ~ .-Range5SP-Carriage-Car, data = s_train, nvmax = vmax) 
coef(reg_best, 16)


#bestsubset but we added Range6SP and Carriage back so subset has  18 variables rather than 16

(fmla2 <- CalcTime ~ .  -Range6SP -Car -Range5SP  -OverSetpoint )
model_lm2 <- lm(fmla2 , data= s_train)
summary(model_lm2)
#plot(model_lm2)
#pred_tlm2 <- predict(model_lm2)

# Predictions on test data
predlm2 <- predict(model_lm2, s_test) 
summary(predlm2)
# RMSE Calculation
res2 <- s_test$CalcTime - predlm2
(rmse2 <- sqrt(mean(res2^2)))


# plot the results
ggplot(s_test, aes(x = predlm2, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "blue") +
  labs(title = "Linear Regression Model Best Subset(18) - Actual vs.Prediction ") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")+
  ylab("Actual Lifetime")



resd2<- res2
resd2 <-data.frame(resd2)
# Draw a density plot of residuals
ggplot(resd2, aes(resd2)) +
  # Add a density curve
  geom_density(color = "blue") +
  # Add a vertical line through zero
  geom_vline(xintercept = 0) +
  xlab("residuals")

##########################
#linear regression model 3 interactions
##########################

(fmla3 <- CalcTime ~ .+ CalcMiles:Module -Car -Range5SP -OverSetpoint )


model_lm3 <- lm(fmla3 , data= s_train)
summary(model_lm3)
#plot(model_lm3)
#pred_tlm3 <- predict(model_lm3)

# Predictions on test data
predlm3<- predict(model_lm3, s_test) 
# RMSE Calculation
res3 <- s_test$CalcTime - predlm3
(rmse3 <- sqrt(mean(res3^2)))

# plot the results
ggplot(s_test, aes(x = predlm3, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "blue") +
  labs(title = "Linear Regression Model Best Subset w Interactions - Actual vs.Prediction ") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")+
  ylab("Actual Lifetime")



resd3<- res3
resd3 <-data.frame(resd3)
# Draw a density plot of residuals
ggplot(resd3, aes(resd3)) +
  # Add a density curve
  geom_density(color = "blue") +
  # Add a vertical line through zero
  geom_vline(xintercept = 0) +
  xlab("residuals")


##########################
#linear regression model 4 interactions
##########################

(fmla4 <- CalcTime ~ .+ Range4SP:Range6SP -Car -Range5SP -OverSetpoint )

model_lm4 <- lm(fmla4 , data= s_train)
summary(model_lm4)
#plot(model_lm4)
#pred_tlm4 <- predict(model_lm4)

# Predictions on test data
predlm4<- predict(model_lm4, s_test) 
# RMSE Calculation
res4 <- s_test$CalcTime - predlm4
(rmse4 <- sqrt(mean(res4^2)))
#34.8855

ggplot(s_test, aes(x = predlm4, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "green") +
  labs(title = "Linear Regression Model 4") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")

##########################
#linear regression model 5 interactions
##########################

(fmla5 <- CalcTime ~ .+Month:AverageAmpSetpoint -Car -Range5SP -OverSetpoint )

model_lm5 <- lm(fmla5 , data= s_train)
summary(model_lm5)
#plot(model_lm4)
#pred_tlm4 <- predict(model_lm4)

# Predictions on test data
predlm5<- predict(model_lm5, s_test) 
# RMSE Calculation
res5 <- s_test$CalcTime - predlm5
(rmse5 <- sqrt(mean(res5^2)))


# plot the results
ggplot(s_test, aes(x = predlm5, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "blue") +
  labs(title = "Linear Regression Model Best Subset w/ Ints - Actual vs.Prediction ") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")+
  ylab("Actual Lifetime")



resd5<- res5
resd5 <-data.frame(resd5)
# Draw a density plot of residuals
ggplot(resd5, aes(resd5)) +
  # Add a density curve
  geom_density(color = "blue") +
  # Add a vertical line through zero
  geom_vline(xintercept = 0) +
  xlab("residuals")




#################################################################
# Elastic Net vs Ridge and Lasso
################################################################
set.seed(2)
x <- model.matrix(CalcTime ~ ., s_train)[, -1]
y <- s_train$CalcTime

stestm <- model.matrix(CalcTime ~ ., s_test)[, -1]
# Fit models:
fit.lasso <- glmnet(x, y, family = "gaussian", alpha = 1)
fit.ridge <- glmnet(x, y, family = "gaussian", alpha = 0)
fit.elnet <- glmnet(x, y, family = "gaussian", alpha = 0.5)


# 10-fold Cross validation for each alpha = 0, 0.1, ... , 0.9, 1.0
fit.lasso.cv <- cv.glmnet(x,y, type.measure = "mse", alpha = 1, 
                          family = "gaussian")
fit.ridge.cv <- cv.glmnet(x, y, type.measure = "mse", alpha = 0,
                          family = "gaussian")
fit.elnet.cv <- cv.glmnet(x, y, type.measure = "mse", alpha = 0.5,
                          family = "gaussian")

for (i in 0:10) {
  assign(paste("fit", i, sep=""), cv.glmnet(x, y, type.measure = "mse", 
                                            alpha=i/10,family = "gaussian"))
}

# Plot solution paths:
par(mfrow=c(3,2))
# For plotting options, type '?plot.glmnet' in R console
plot(fit.lasso, xvar="lambda")
plot(fit10, main="LASSO")

plot(fit.ridge, xvar="lambda")
plot(fit0, main="Ridge")

plot(fit.elnet, xvar="lambda")
plot(fit5, main="Elastic Net")

# RMSE Calculation

#yhat0 <- predict(fit0, s=fit0$lambda.1se, newx=stestm)
pred0 <- predict(fit0, s=fit0$lambda.min, newx=stestm)
pred1 <- predict(fit1, s=fit1$lambda.min, newx=stestm)
pred2 <- predict(fit2, s=fit2$lambda.min, newx=stestm)
pred3 <- predict(fit3, s=fit3$lambda.min, newx=stestm)
pred4 <- predict(fit4, s=fit4$lambda.min, newx=stestm)
pred5 <- predict(fit5, s=fit5$lambda.min, newx=stestm)
pred6 <- predict(fit6, s=fit6$lambda.min, newx=stestm)
pred7 <- predict(fit7, s=fit7$lambda.min, newx=stestm)
pred8 <- predict(fit8, s=fit8$lambda.min, newx=stestm)
pred9 <- predict(fit9, s=fit9$lambda.min, newx=stestm)
pred10 <- predict(fit10, s=fit10$lambda.min, newx=stestm)

(rmse0 <- sqrt(mean((s_test$CalcTime - pred0)^2)))
(rmse1 <- sqrt(mean((s_test$CalcTime - pred1)^2)))
(rmse2 <- sqrt(mean((s_test$CalcTime - pred2)^2)))
(rmse3 <- sqrt(mean((s_test$CalcTime - pred3)^2)))
(rmse4 <- sqrt(mean((s_test$CalcTime - pred4)^2)))
(rmse5 <- sqrt(mean((s_test$CalcTime - pred5)^2)))
(rmse6 <- sqrt(mean((s_test$CalcTime - pred6)^2)))
(rmse7 <- sqrt(mean((s_test$CalcTime - pred7)^2)))
(rmse8 <- sqrt(mean((s_test$CalcTime - pred8)^2)))
(rmse9 <- sqrt(mean((s_test$CalcTime - pred9)^2)))
(rmse10 <- sqrt(mean((s_test$CalcTime - pred10)^2)))
res10 <- (s_test$CalcTime - pred10)
# alpha=1 lasso wins with rmse 34.94326

# plot the results
ggplot(s_test, aes(x = pred10, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "blue") +
  labs(title = "Elastic Net- alpha= 1 - Actual vs.Prediction ") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")+
  ylab("Actual Lifetime")




#################################################################
# Elastic Net vs Ridge and Lasso with custom grid
################################################################
set.seed(2)
grid <- 10 ^ seq(10, -2, length = 100)
# Fit models:
fitc.lasso <- glmnet(x, y, family = "gaussian", alpha = 1, lambda = grid, thresh = 1e-12)
fitc.ridge <- glmnet(x, y, family = "gaussian", alpha = 0, lambda = grid, thresh = 1e-12)
fitc.elnet <- glmnet(x, y, family = "gaussian", alpha = 0.5, lambda = grid, thresh = 1e-12)


# 10-fold Cross validation for each alpha = 0, 0.1, ... , 0.9, 1.0
fitc.lasso.cv <- cv.glmnet(x,y, type.measure = "mse", alpha = 1, 
                          family = "gaussian", lambda = grid, thresh = 1e-12)
fitc.ridge.cv <- cv.glmnet(x, y, type.measure = "mse", alpha = 0,
                          family = "gaussian",lambda = grid, thresh = 1e-12)
fitc.elnet.cv <- cv.glmnet(x, y, type.measure = "mse", alpha = 0.5,
                          family = "gaussian", lambda = grid, thresh = 1e-12)

for (i in 0:10) {
  assign(paste("fitc", i, sep=""), cv.glmnet(x, y, type.measure = "mse", 
                                            alpha=i/10,family = "gaussian",lambda = grid, thresh = 1e-12))
}

# Plot solution paths:
par(mfrow=c(3,2))
# For plotting options, type '?plot.glmnet' in R console
plot(fitc.lasso, xvar="lambda")
plot(fitc10, main="LASSO")

plot(fitc.ridge, xvar="lambda")
plot(fitc0, main="Ridge")

plot(fitc.elnet, xvar="lambda")
plot(fitc5, main="Elastic Net")

# RMSE Calculation

#yhat0 <- predict(fit0, s=fit0$lambda.1se, newx=stestm)
pred0_2 <- predict(fitc0, s=fitc0$lambda.min, newx=stestm)
pred1_2 <- predict(fitc1, s=fitc1$lambda.min, newx=stestm)
pred2_2 <- predict(fitc2, s=fitc2$lambda.min, newx=stestm)
pred3_2 <- predict(fitc3, s=fitc3$lambda.min, newx=stestm)
pred4_2 <- predict(fitc4, s=fitc4$lambda.min, newx=stestm)
pred5_2 <- predict(fitc5, s=fitc5$lambda.min, newx=stestm)
pred6_2 <- predict(fitc6, s=fitc6$lambda.min, newx=stestm)
pred7_2 <- predict(fitc7, s=fitc7$lambda.min, newx=stestm)
pred8_2 <- predict(fitc8, s=fitc8$lambda.min, newx=stestm)
pred9_2 <- predict(fitc9, s=fitc9$lambda.min, newx=stestm)
pred10_2 <- predict(fitc10, s=fitc10$lambda.min, newx=stestm)

(rmse0 <- sqrt(mean((s_test$CalcTime - pred0_2)^2)))
(rmse1 <- sqrt(mean((s_test$CalcTime - pred1_2)^2)))
(rmse2 <- sqrt(mean((s_test$CalcTime - pred2_2)^2)))
(rmse3 <- sqrt(mean((s_test$CalcTime - pred3_2)^2)))
(rmse4 <- sqrt(mean((s_test$CalcTime - pred4_2)^2)))
(rmse5 <- sqrt(mean((s_test$CalcTime - pred5_2)^2)))
(rmse6 <- sqrt(mean((s_test$CalcTime - pred6_2)^2)))
(rmse7 <- sqrt(mean((s_test$CalcTime - pred7_2)^2)))
(rmse8 <- sqrt(mean((s_test$CalcTime - pred8_2)^2)))
(rmse9 <- sqrt(mean((s_test$CalcTime - pred9_2)^2)))
(rmse10 <- sqrt(mean((s_test$CalcTime - pred10_2)^2)))

# alpha=0 ridge wins with min rmse which corresponds to ridge 2, 34.92901
ggplot(s_test, aes(x = pred0_2, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "blue") +
  labs(title = "Elastic Net with custom grid on beta- alpha= 0 - Actual vs.Prediction ") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")+
  ylab("Actual Lifetime")



#############################################
# The Principal Component Regression (PCR)
#############################################

set.seed(2)
model_pcr2 <- pcr(CalcTime ~ . -Car -Carriage -Range5SP -OverSetpoint , 
                  data = s_train, scale = TRUE, validation = "CV")
summary(model_pcr2)
validationplot(model_pcr2,val.type = "RMSEP")

pred_pcr2 <- predict(model_pcr2, s_test, ncomp = 10)
rmse_pcr2 <- sqrt(mean((pred_pcr2 - s_test$CalcTime)^2))
paste("Test rmse for PCR model is", round(rmse_pcr2))


ggplot(s_test, aes(x = pred_pcr2, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "green") +
  labs(title = "Principal Component Regression - Actual vs.Prediction") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")


#############################################
# The Partial Least Squares (PLS) 
#############################################
set.seed(2)
pls_model <- plsr(CalcTime ~ .-Car -Carriage, data = s_train, scale = TRUE, validation = "CV")
summary(pls_model)
validationplot(pls_model,val.type = "RMSEP")

pred_pls <- predict(pls_model, s_test, ncomp = 4)
rmse_pls <- sqrt(mean((pred_pls - s_test$CalcTime)^2))
paste("Test rmse for PLS model w/ 4 comp  is", round(rmse_pls))
#Test rmse for PLS model w/ 15 comp  is 36"

ggplot(s_test, aes(x = pred_pls, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "green") +
  labs(title = "Partial Least Squares - Actual vs.Prediction ") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")+
  ylab("Actual Lifetime")

#############################################
# Decision Tree using tree
#############################################


#tree
library(tree) #For tree()
#Tree fit using tree()
treefit <- tree(CalcTime ~ ., data = s_train) 

summary(treefit)

#Plot of the tree
plot(treefit)
text(treefit, pretty = 0)

# RMSE Calculation
treepred <- predict(treefit, newdata = s_test)
res_tree <- s_test$CalcTime - treepred
(rmse <- sqrt(mean(res_tree^2)))
# Plot
ggplot(s_test, aes(x = treepred, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "red") +
  labs(title = "Tree Model - Actual vs.Prediction") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")+
  ylab("Actual Lifetime")


#############################################
# Decision Tree -Rpart
#############################################

model_rpart <- rpart(CalcTime ~ ., data= s_train, cp = 0.0004)
#summary(model_rpart
plot(model_rpart)

# Predictions on test data
predrpart <- predict(model_rpart, s_test) 

# RMSE Calculation
res_rpart <- s_test$CalcTime - predrpart
(rmse_rpat <- sqrt(mean(res_rpart^2)))

 
ggplot(s_test, aes(x = predrpart, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "red") +
  labs(title = "Recursive Partitioning Model - Actual vs.Prediction ") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")+
  ylab("Actual Lifetime")



#################################
#Random Forest (Bagging)
#################################

library(randomForest)
# Use the Bagging Approach: mtry = p
set.seed(2)
bagged <- randomForest(CalcTime ~ ., data = s_train, mtry = 22, ntree = 500, 
                       importance = TRUE)
# RMSE Calculation
pred_b <- predict(bagged, newdata = s_test)
res_bagg <- s_test$CalcTime - pred_b
(rmse_b <- sqrt(mean(res_bagg^2)))
# Best Rmse so far : 32.59519

# Determine which variables are most important
importance(bagged)


#################################
#Random Forest 
#################################

# default mtry = p/3
set.seed(2)

forest <- randomForest(CalcTime ~ ., data = s_train, mtry=10, ntree = 500, 
                       importance = TRUE)

# RMSE Calculation
pred_f <- predict(forest, newdata = s_test)
res_for <- s_test$CalcTime - pred_f
(rmse_forest <- sqrt(mean(res_for^2)))


importance(forest)

# Plot
ggplot(s_test, aes(x = pred_f, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "red") +
  labs(title = "Forest Model") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")

#################################
#Random Forest with ranger
#################################
library(ranger)
set.seed(2)

 (fit_rf <- ranger(CalcTime ~ ., # formula
                          s_train, # data
                          num.trees = 1000,
                          mtry=11,
                          respect.unordered.factors = "order"))

 pred_rf <- predict(fit_rf, s_test)$predictions
 # RMSE Calculation
 res_rf <- s_test$CalcTime - pred_rf
 (rmse_rf <- sqrt(mean(res_rf^2)))




res_drf<- res_rf
res_drf <-data.frame(res_drf)
# Plot
ggplot(s_test, aes(x = pred_rf, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "red") +
  labs(title = "Forest Model - Actual vs.Prediction") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")+
  ylab("Actual Lifetime")
# Draw a density plot of residuals
ggplot(res_drf, aes(res_drf)) +
  # Add a density curve
  geom_density(color = "blue") +
  # Add a vertical line through zero
  geom_vline(xintercept = 0) +
  xlab("residuals")


#############################################
# The Principal ComponentAnalysis (PCA)
#############################################

s_train_pc <- s_train[,c(-4,-5)]
pr<-prcomp(s_train_pc, scale = TRUE)

plot(pr, type = "l")
biplot(pr , scale =0)
newdat<-pr$x[,1:10]

s_trainnw<-cbind(s_train,newdat)

#####Linear regression on PCA’d data ######
lmmodd<-lm(CalcTime~PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9+PC10+Carriage,data=s_trainnw)

# apply same transformations on test data 
s_test_pc <- s_test[,c(-4,-5)]
pcatest<- predict(pr, newdata= s_test_pc)
pca_test<-as.data.frame(pcatest)
Pca_test <- cbind(s_test,pca_test)

#Prediction on test data
predlmpca <- predict(object = lmmodd, newdata=Pca_test)

# RMSE Calculation
respc <- s_test$CalcTime - predlmpca
(rmsepc <- sqrt(mean(respc^2)))
# plot the results
ggplot(s_test, aes(x = predlmpca, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "blue") +
  labs(title = "Linear Regression After PCA - Actual vs.Prediction ") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions") +
  ylab("Actual Lifetime")

#################################
#Random Forest with ranger After PCA
#################################
library(ranger)
set.seed(2)

(fit_rfpca <- ranger(CalcTime ~ PC1 + PC2+ PC3+ PC4+ PC5+ PC6+ PC7+ PC8 + PC9 + PC10  + Carriage,
                  data=s_trainnw ,
                  num.trees = 1000,
                  mtry=6,
                  respect.unordered.factors = "order"))

pred_rfpca <- predict(fit_rfpca, Pca_test)$predictions
# RMSE Calculation
res_rfpca <- s_test$CalcTime - pred_rfpca
(rmse_rfpca <- sqrt(mean(res_rfpca^2)))

# plot the results
ggplot(s_test, aes(x = pred_rfpca, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "red") +
  labs(title = "Random Forest After PCA - Actual vs.Prediction ") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions") +
  ylab("Actual Lifetime")


# Plot
ggplot(s_test, aes(x = pred_rf, y= CalcTime)) + 
  geom_point() +
  geom_abline(color = "red") +
  labs(title = "Forest Model - Actual vs.Prediction") +
  theme(plot.title = element_text(hjust = 0.5)) + 
  xlab("Predictions")+
  ylab("Actual Lifetime")
