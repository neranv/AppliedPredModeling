library(AppliedPredictiveModeling)
library(caret)
library(dplyr)
data(twoClassData)
str(predictors)
str(classes)


#split the dataset
set.seed(1)
trainingRows <- createDataPartition(classes, 
                                    p = 0.80, 
                                    list = FALSE)

head(trainingRows)

trainPredictors <- predictors[trainingRows,]
trainClasses <- classes[trainingRows]

testPredictors <- predictors[-trainingRows,]
testClasses <- classes[-trainingRows]

str(trainPredictors)
str(testPredictors)

#Resampling

# Repeated splitting
set.seed(1)
repeatedSplits <- createDataPartition(trainClasses, p=0.8, times=3)

str(repeatedSplits)

#createFolds - for k fold cross validation

set.seed(1)
cvSplits <- createFolds(trainClasses, k=10, returnTrain = TRUE)
str(cvSplits)
fold1 <- cvSplits[[1]]
# getting the first 10% of the data
cvPredictors1 <- trainPredictors[fold1,]
cvClasses1 <- trainClasses[fold1]

nrow(trainPredictors)

nrow(cvPredictors1)

# Basic Model Building in R

trainPredictors <- as.matrix(trainPredictors)

knnFit <- knn3(x=trainPredictors, y=trainClasses, k=5)


testPredictions <- predict(knnFit, newdata = testPredictors, type = "class")
head(testPredictions)
str(testPredictions)

## Determination of Tuning Parameters

data(GermanCredit)

#REmove near zero variance
GermanCredit_new <- GermanCredit[,-nearZeroVar(GermanCredit)]
GermanCredit_new$Class <- GermanCredit$Class

#determine training rows
creditTrainRows <- createDataPartition(GermanCredit_new$Class, p=0.8, list=FALSE)

GermanCreditTrain <- GermanCredit_new[creditTrainRows,]
GermanCreditTest <- GermanCredit_new[-creditTrainRows,]


set.seed(1056)

#by default bootstrap
svmFit <- train(Class ~ . ,
                data = GermanCreditTrain,
                method = "svmRadial",
                preProc = c("center", "scale"),
                tuneLength = 10 )

#To use Repeated 10-fold cross-validation
svmFit <- train(Class ~ . ,
                data = GermanCreditTrain,
                method = "svmRadial",
                preProc = c("center", "scale"),
                tuneLength = 10,
                trControl = trainControl(method = "repeatedcv",
                                         repeats = 5,
                                         classProbs = TRUE))

plot(svmFit, scales = list(x=list(log=2)))

predictedClasses <- predict(svmFit, GermanCreditTest)

#the type argument can be used to predict class probabilities

predictedProbs <- predict(svmFit, GermanCreditTest, type = "prob")
head(predictedProbs)

# prediction performance
table(GermanCreditTest$Class,predictedClasses)

## Between-Model Comparisons

# Building the glm model doesnt need any tuning parametets


set.seed(1056)
logisticReg <- train(Class ~ .,
                     data = GermanCreditTrain,
                     method = "glm",
                     trControl = trainControl(method="repeatedcv",
                                              repeats = 5))

logisticReg

# using resmaples function to compare 2 models


resamp <- resamples(list(SVM=svmFit, Logistic= logisticReg))
summary(resamp)

modelDifferences <- diff(resamp)

summary(modelDifferences)

#p-values are high for both accuracy and kappa. Hence there is no significant differrence 
#between the models

