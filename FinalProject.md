---
title: "Practical Machine Learning - Week 4 Course Project"
date: "4/10/2017"
output: html_document
---

## Synopsis:
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are use to find patterns in their human movement. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  
The goal of this project is to build the best performing model using the training data set provided and predict on a test dataset. This is the "classe" variable in the training set. This report describes how the various models were built, their error rates & accuracies and provides a rationale for the approach.  

#### Exploratory Data Analysis  
  
Read in the training and the test datasets  

```{r, echo=FALSE, message=FALSE}
library(caret)
```
```{r, message=FALSE, cache=TRUE}
pml <- read.csv("./data/pml-training.csv")
pml_test <- read.csv("./data/pml-testing.csv")
```

Review the structure of the dataset. Eyeball for missing values columns, near zero variance features and degenerate columns
```{r, eval=FALSE}
dim(pml)
colnames(pml)
str(pml)
colSums(is.na(pml))
```

#### Data Preprocessing

Remove majority NA columns, degenerate features and predictors that will not make a meaningful contribute  to the response (dependent variable 'classe')  
```{r, message=FALSE}
pml <- pml[-nearZeroVar(pml)]
pml <- pml[colSums(is.na(pml)) == 0]
pml$X <- NULL
pml$user_name <- NULL
pml$raw_timestamp_part_1 <- NULL
pml$raw_timestamp_part_2 <- NULL
pml$cvtd_timestamp <- NULL
pml$num_window <- NULL
```

After preprocessing, we are left with the following 52 predictors to predict the repsonse 'classe'.  
```{r, comment="", echo=FALSE}
colnames(pml[-53])
```
  
#### Data Partitioning  
  
Partition the pml dataset into two - training (80%) and validation (20%). Perform the split on the 'classe' variable to ensure even distribution of response classes in the training and the validation subsets  

```{r, message=FALSE, cache=TRUE}
set.seed(123)
training <- createDataPartition(pml$classe, p = 0.8, list = FALSE)
pml_train <- pml[training,]
pml_valid <- pml[-training,]
```

#### Modeling, Prediction and Performance  

We'll try various models to achieve the best prediction accuracy. For each of the models, we'll cross validate with 3 folds.
```{r, message=FALSE}
fitCtrl <- trainControl(method = "cv", number = 3)
```

##### Decision Tree  

Fit a Decision Tree model to the training data and review the model  
```{r}
treeFit <- train(classe ~ ., method = "rpart", data = pml_train, trControl = fitCtrl)
treeFit
```

Predict the fitted model against the validation dataset, review the confusion matrix and compute the out of sample error. Finally, compute the Accuracy of the model.
```{r}
treePred <- predict(treeFit, newdata = pml_valid)
print(paste("Error rate = ", (treeErr <- round(mean(treePred != pml_valid$classe), digits = 4))))
print(paste("Accuracy = ", 1-treeErr))
table(treePred, pml_valid$classe)
```
  
The confusion matrix is shown. With a single Decision Tree, we see that the out of sample error rate is 50.27% and prediction accuracy on the validation dataset is 49.73%. Therefore, this model isn't the best for this exercise. We'll proceed with other models for an improved fit.  
  
##### Random Forest  
  
Fit a Random Forest model to the training data and review the model  
```{r}
rfFit <- train(classe ~ ., method = "rf", data = pml_train, ntree = 100, trControl = fitCtrl)
rfFit
```
  
Predict the fitted model against the validation dataset, review the confusion matrix and compute the out of sample error. Finally, compute the Accuracy of the model.  
  
```{r}
rfPred <- predict(rfFit, newdata = pml_valid)
print(paste("Error rate = ", (rfErr <- round(mean(rfPred != pml_valid$classe), digits = 4))))
print(paste("Accuracy = ", 1-rfErr))
table(rfPred, pml_valid$classe)
```
  
The confusion matrix is shown. With a Random Forest, we see that the out of sample error rate is 0.48% and prediction accuracy on the validation dataset is 99.52%. True to its reputation, the Random Forest model best predictive model for this exercise. We'll proceed now to make a prediction on the final test set.  
  
#### Test Dataset - Data Preprocessing  
  
Repeat the same preprocessing steps on the final test dataset as was done on the training dataset.  
  
```{r}
pml_test <- pml_test[-nearZeroVar(pml_test)]
pml_test <- pml_test[colSums(is.na(pml_test)) == 0]
pml_test$X <- NULL
pml_test$user_name <- NULL
pml_test$raw_timestamp_part_1 <- NULL
pml_test$raw_timestamp_part_2 <- NULL
pml_test$cvtd_timestamp <- NULL
pml_test$num_window <- NULL
```
  
#### Test Dataset - Prediction  
  
Use the Random Forest fitted model to predict the response classes of the test data set. Print the test set predictions.  
  
```{r}
testPred <- predict(rfFit, newdata = pml_test)
cbind(problem_id = pml_test$problem_id, as.data.frame(testPred))
```
  
Using these predictions to answer the Quiz associated with the execise, the model correctly predicted all of the responses. Therefore, the Random Forest model is the best predictive model.
