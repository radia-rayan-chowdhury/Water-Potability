# library
library(dplyr)
library(caret)
library(corrplot)
library(tidyr)
library(corrgram)
library(ggplot2)
library(ggthemes)
library(cluster)
library(ggplot2)
library(e1071)    # Calculate Skewness
library(ggpubr)   # Log Transformation
library(moments)  # Log Transformation

#-------------------------------------------------------------------------#
#load the dataset
water_dataset <- read.csv(file='.....') # Add the file path

#-------------------------------------------------------------------------#
# Analysing data set 
dim(water_dataset)            # dimension of the dataset
sapply(water_dataset, class)  # list types for each attribute
summary(water_dataset)        # summary of the dataset

#Checking the dependent variable - how many 1 and 0
one <- filter(water_dataset,Potability==1)
o <- nrow(one)  #1278

zero <- filter(water_dataset,Potability==0)
z <- nrow(zero) #1998

# histograms of water potable or not
par(mfrow=c(1,1))
count <- table(water_dataset$Potability)
barplot(count, main="Potability Distribution", xlab="Number of Potable Water and non Potable Water Observation")

# Correlation between all of the numeric attributes
cor(water_dataset[,1:10])

#Checking Missing Data
#Feature Selection
pctmiss <- colSums(is.na(water_dataset))/nrow(water_dataset)
round(pctmiss, 2)


#-------------------------------------------------------------------------#
# Data Cleaning
#------------------------------------------------------------------------

#Listwise Deletion - Row Delete
water_dataset <- na.omit(water_dataset)


one_newdata <- filter(water_dataset,Potability==1)
#one_newdata
nrow(one_newdata)  #811

zero_newdata <- filter(water_dataset,Potability==0)
#zero_newdata
nrow(zero_newdata) #1200

cor(water_dataset[,1:10])   # Correlation check After Row deletion


# Normalization
normalize <- function(x) {
  + return ((x - min(x)) / (max(x) - min(x))) }

water_norm <- water_dataset
#apply Min-Max normalization to all columns in iris dataset
water_norm[,1:9] <- as.data.frame(lapply(water_norm[,1:9], normalize))


#--------------------------------------------------------------------------#
## Unimodal Data Visualizations - On normalized data

# histograms each attribute
par(mfrow=c(3,4))
for(i in 1:10) {
  hist(water_norm[,i], main=names(water_norm)[i])
}

# density plot for each attribute
par(mfrow=c(3,4))
for(i in 1:10) {
  plot(density(water_norm[,i]), main=names(water_norm)[i])
}

# boxplots for each attribute
par(mfrow=c(2,5))
for(i in 1:10) {
  boxplot(water_norm[,i], main=names(water_norm)[i])
}

#------------------------------------
## Multi-modal Data Visualisations

# scatterplot matrix
pairs(water_norm[,1:10])

par(mfrow=c(1,1))
# correlation plot
correlations <- cor(water_norm[,1:10])
corrplot(correlations, method="circle")

#----------------------------------------------------------------------------

#---------------------------------------------------------------------------#
# Calculate the skewness - for each column
skewness(water_norm[,-10])
dim(water_norm)

# Data Cleaning----------------------------------------
#Log transformation of the skewed data: (Solids column)
water_norm$Solids <- log10(water_norm$Solids)

# Distribution of Solids variable - after log transformation
ggdensity(water_norm, x = "Solids", fill = "lightgray", title = "Solids") +
  stat_overlay_normal_density(color = "red", linetype = "dashed")

#Compute skewness on the transformed data:
skewness(water_norm$Solids, na.rm = TRUE)

dim(water_norm)


#------------------------------------------------------------------------------
# Remove Outliers
# Define Outlier Function

outliers <- function(x) {
  
  Q1 <- quantile(x, probs=.25)
  Q3 <- quantile(x, probs=.75)
  iqr = Q3-Q1
  
  upper_limit = Q3 + (iqr*1.5)
  lower_limit = Q1 - (iqr*1.5)
  
  x > upper_limit | x < lower_limit
}

remove_outliers <- function(df, cols = names(df)) {
  for (col in cols) {
    df <- df[!outliers(df[[col]]),]
  }
  df
}

# Apply outlier function to data frame
df <- water_norm
dim(df)
df <- remove_outliers(df, c('ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                            'Conductivity','Organic_carbon','Trihalomethanes',
                            'Turbidity'))
#df

dim(df)
water_norm <- df
#--------------------------------------------------------------------#
#------------------Data Partitioning for Validation------------------#
#--------------------------------------------------------------------#
set.seed(7)
#dat.d <- sample(1:nrow(water_norm),size=nrow(water_norm)*0.7,replace = FALSE) #random selection of 70% data.
data_partition <- createDataPartition(water_norm$Potability, p=0.70,list=FALSE)

training <- water_dataset[data_partition,] # 70% training data
testing <- water_dataset[-data_partition,] # remaining 30% test data

#Now creating seperate dataframe for 'Creditability' feature which is our target.
train_labels <- water_dataset[data_partition,10]
test_labels  <- water_dataset[-data_partition,10]   

#--------------------------------------------------------------------#
#---------------------------------KNN--------------------------------#
#--------------------------------------------------------------------#

library(class)          # to call class package

NROW(train_labels)   # to find the number of observation

knn.35 <-  knn(train=training, test=testing, cl=train_labels, k=35)
knn.36 <-  knn(train=training, test=testing, cl=train_labels, k=36)

## Let's calculate the proportion of correct classification for k = 35, 36

ACC.35 <- 100 * sum(test_labels == knn.35)/NROW(test_labels)  
ACC.36 <- 100 * sum(test_labels == knn.36)/NROW(test_labels) 
ACC.35
ACC.36

confusionMatrix(table(knn.35 ,test_labels))
confusionMatrix(table(knn.36 ,test_labels))


#Step 7: Optimization
i=1
k.optm=1
for (i in 1:100){ knn.mod <- knn(train=training, test=testing, cl=training$Potability, k=i)
k.optm[i] <- 100 * sum(testing$Potability == knn.mod)/NROW(testing$Potability)
k=i
cat(k,'=',k.optm[i],' + ')}

par(mfrow=c(1,1))
plot(k.optm, type="b", xlab="K- Value",ylab="Accuracy level")


knn.49 = knn(train=training, test=testing, cl=train_labels, k=49)
ACC.49 <- 100 * sum(test_labels == knn.49)/NROW(test_labels)  # For knn = 27
ACC.49   #Accuracy is 67.67%
confusionMatrix(table(knn.49 ,test_labels))



#--------------------------------------------------------------------#
#--------------------Evaluation Algorithms : baseline----------------#
#--------------------------------------------------------------------#
# Prepare the test harness for evaluating algorithms.
# Run algorithms using 10-fold cross validation
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"

#------------------------------------------------#
# Estimate accuracy of machine learning algorithms
# LM
set.seed(7)
fit.lm <- train(Potability~., data=df, method="lm", metric=metric, preProc=c("center", "scale"), trControl=trainControl)
# GLM
set.seed(7)
fit.glm <- train(Potability~., data=df, method="glm", metric=metric, preProc=c("center", "scale"), trControl=trainControl)

# GLMNET
set.seed(7)
fit.glmnet <- train(Potability~., data=df, method="glmnet", metric=metric, preProc=c("center", "scale"), trControl=trainControl)

# SVM
set.seed(7)
fit.svm <- train(Potability~., data=df, method="svmRadial", metric=metric, preProc=c("center", "scale"), trControl=trainControl)

# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(Potability~., data=df, method="rpart", metric=metric, tuneGrid=grid, preProc=c("center", "scale"), trControl=trainControl)

# KNN
set.seed(7)
fit.knn <- train(Potability~., data=df, method="knn", metric=metric, preProc=c("center", "scale"), trControl=trainControl)


# Collect resample statistics from models and summarize results. Compare algorithms
results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet,
                          SVM=fit.svm, CART=fit.cart, KNN=fit.knn))
summary(results)
dotplot(results)



#--------------------------------------------------------------------#
#--------------------Evaluation Algorithms : boxcox------------------#
#--------------------------------------------------------------------#

# Estimate accuracy of algorithms on transformed dataset.
# Run algorithms using 10-fold cross validation
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"

# lm
set.seed(7)
fit.lm <- train(Potability~., data=df, method="lm", metric=metric, preProc=c("center","scale", "BoxCox"), trControl=trainControl)

# GLM
set.seed(7)
fit.glm <- train(Potability~., data=df, method="glm", metric=metric, preProc=c("center","scale", "BoxCox"), trControl=trainControl)

# GLMNET
set.seed(7)
fit.glmnet <- train(Potability~., data=df, method="glmnet", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=trainControl)

# SVM
set.seed(7)
fit.svm <- train(Potability~., data=df, method="svmRadial", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=trainControl)

# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(Potability~., data=df, method="rpart", metric=metric, tuneGrid=grid, preProc=c("center", "scale", "BoxCox"), trControl=trainControl)

# KNN
set.seed(7)
fit.knn <- train(Potability~., data=df, method="knn", metric=metric, preProc=c("center", "scale", "BoxCox"), trControl=trainControl)

# Compare algorithms
transformResults <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm, CART=fit.cart, KNN=fit.knn))

summary(transformResults)
dotplot(transformResults)


#--------------------------------------------------------------------#
#-------------Evaluation Algorithms : ensemble methods---------------#
#--------------------------------------------------------------------#

# Random Forest
set.seed(7)
fit.rf <- train(Potability~., data=df, method="rf", metric=metric, preProc=c("BoxCox"), trControl=trainControl)

# Stochastic Gradient Boosting
set.seed(7)
fit.gbm <- train(Potability~., data=df, method="gbm", metric=metric, preProc=c("BoxCox"), trControl=trainControl, verbose=FALSE)

# Cubist
set.seed(7)
fit.cubist <- train(Potability~., data=df, method="cubist", metric=metric, preProc=c("BoxCox"), trControl=trainControl)

# Compare algorithms
ensembleResults <- resamples(list(RF=fit.rf, GBM=fit.gbm, CUBIST=fit.cubist))
summary(ensembleResults)
dotplot(ensembleResults)


