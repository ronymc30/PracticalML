Synopsis:
---------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are use to find patterns
in their human movement. One thing that people regularly do is quantify
how much of a particular activity they do, but they rarely quantify how
well they do it.  
The goal of this project is to build the best performing model using the
training data set provided and predict on a test dataset. This is the
"classe" variable in the training set. This report describes how the
various models were built, their error rates & accuracies and provides a
rationale for the approach.

#### Exploratory Data Analysis

Read in the training and the test datasets

    pml <- read.csv("./data/pml-training.csv")
    pml_test <- read.csv("./data/pml-testing.csv")

Review the structure of the dataset. Eyeball for missing values columns,
near zero variance features and degenerate columns

    dim(pml)
    colnames(pml)
    str(pml)
    colSums(is.na(pml))

#### Data Preprocessing

Remove majority NA columns, degenerate features and predictors that will
not make a meaningful contribute to the response (dependent variable
'classe')

    pml <- pml[-nearZeroVar(pml)]
    pml <- pml[colSums(is.na(pml)) == 0]
    pml$X <- NULL
    pml$user_name <- NULL
    pml$raw_timestamp_part_1 <- NULL
    pml$raw_timestamp_part_2 <- NULL
    pml$cvtd_timestamp <- NULL
    pml$num_window <- NULL

After preprocessing, we are left with the following 52 predictors to
predict the repsonse 'classe'.

     [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
     [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
     [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
    [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
    [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
    [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
    [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
    [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
    [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
    [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
    [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
    [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
    [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
    [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
    [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
    [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
    [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
    [52] "magnet_forearm_z"    

#### Data Partitioning

Partition the pml dataset into two - training (80%) and validation
(20%). Perform the split on the 'classe' variable to ensure even
distribution of response classes in the training and the validation
subsets

    set.seed(123)
    training <- createDataPartition(pml$classe, p = 0.8, list = FALSE)
    pml_train <- pml[training,]
    pml_valid <- pml[-training,]

#### Modeling, Prediction and Performance

We'll try various models to achieve the best prediction accuracy. For
each of the models, we'll cross validate with 3 folds.

    fitCtrl <- trainControl(method = "cv", number = 3)

##### Decision Tree

Fit a Decision Tree model to the training data and review the model

    treeFit <- train(classe ~ ., method = "rpart", data = pml_train, trControl = fitCtrl)

    ## Loading required package: rpart

    treeFit

    ## CART 
    ## 
    ## 15699 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 10465, 10467, 10466 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          Accuracy   Kappa     
    ##   0.03676012  0.4942347  0.33890510
    ##   0.05957573  0.4074859  0.19425001
    ##   0.11544281  0.3384292  0.08235031
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was cp = 0.03676012.

Predict the fitted model against the validation dataset, review the
confusion matrix and compute the out of sample error. Finally, compute
the Accuracy of the model.

    treePred <- predict(treeFit, newdata = pml_valid)
    print(paste("Error rate = ", (treeErr <- round(mean(treePred != pml_valid$classe), digits = 4))))

    ## [1] "Error rate =  0.5027"

    print(paste("Accuracy = ", 1-treeErr))

    ## [1] "Accuracy =  0.4973"

    table(treePred, pml_valid$classe)

    ##         
    ## treePred    A    B    C    D    E
    ##        A 1011  308  304  283  107
    ##        B   16  262   26  132   87
    ##        C   85  189  354  228  203
    ##        D    0    0    0    0    0
    ##        E    4    0    0    0  324

The confusion matrix is shown. With a single Decision Tree, we see that
the out of sample error rate is 50.27% and prediction accuracy on the
validation dataset is 49.73%. Therefore, this model isn't the best for
this exercise. We'll proceed with other models for an improved fit.

##### Random Forest

Fit a Random Forest model to the training data and review the model

    rfFit <- train(classe ~ ., method = "rf", data = pml_train, ntree = 100, trControl = fitCtrl)

    ## Loading required package: randomForest

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    rfFit

    ## Random Forest 
    ## 
    ## 15699 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (3 fold) 
    ## Summary of sample sizes: 10467, 10465, 10466 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9891712  0.9863001
    ##   27    0.9917191  0.9895242
    ##   52    0.9838206  0.9795314
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 27.

Predict the fitted model against the validation dataset, review the
confusion matrix and compute the out of sample error. Finally, compute
the Accuracy of the model.

    rfPred <- predict(rfFit, newdata = pml_valid)
    print(paste("Error rate = ", (rfErr <- round(mean(rfPred != pml_valid$classe), digits = 4))))

    ## [1] "Error rate =  0.0056"

    print(paste("Accuracy = ", 1-rfErr))

    ## [1] "Accuracy =  0.9944"

    table(rfPred, pml_valid$classe)

    ##       
    ## rfPred    A    B    C    D    E
    ##      A 1115    7    0    0    0
    ##      B    1  750    2    1    0
    ##      C    0    2  679    5    0
    ##      D    0    0    3  637    1
    ##      E    0    0    0    0  720

The confusion matrix is shown. With a Random Forest, we see that the out
of sample error rate is 0.48% and prediction accuracy on the validation
dataset is 99.52%. True to its reputation, the Random Forest model best
predictive model for this exercise. We'll proceed now to make a
prediction on the final test set.

#### Test Dataset - Data Preprocessing

Repeat the same preprocessing steps on the final test dataset as was
done on the training dataset.

    pml_test <- pml_test[-nearZeroVar(pml_test)]
    pml_test <- pml_test[colSums(is.na(pml_test)) == 0]
    pml_test$X <- NULL
    pml_test$user_name <- NULL
    pml_test$raw_timestamp_part_1 <- NULL
    pml_test$raw_timestamp_part_2 <- NULL
    pml_test$cvtd_timestamp <- NULL
    pml_test$num_window <- NULL

#### Test Dataset - Prediction

Use the Random Forest fitted model to predict the response classes of
the test data set. Print the test set predictions.

    testPred <- predict(rfFit, newdata = pml_test)
    cbind(problem_id = pml_test$problem_id, as.data.frame(testPred))

    ##    problem_id testPred
    ## 1           1        B
    ## 2           2        A
    ## 3           3        B
    ## 4           4        A
    ## 5           5        A
    ## 6           6        E
    ## 7           7        D
    ## 8           8        B
    ## 9           9        A
    ## 10         10        A
    ## 11         11        B
    ## 12         12        C
    ## 13         13        B
    ## 14         14        A
    ## 15         15        E
    ## 16         16        E
    ## 17         17        A
    ## 18         18        B
    ## 19         19        B
    ## 20         20        B

Using these predictions to answer the Quiz associated with the execise,
the model correctly predicted all of the responses. Therefore, the
Random Forest model is the best predictive model.
