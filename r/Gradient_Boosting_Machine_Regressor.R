#' @title Supervised Regression ML Algorithm using Gradient Boosting Machine
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Gradient_Boosting_Machine_Regressor(x = all[, -1], y = all[, 1])
#' @export Gradient_Boosting_Machine_Regressor
#'
#' # Define function
Gradient_Boosting_Machine_Regressor <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  num.of.trees = 10,
  shrinkage = 0.05,
  interaction.depth = 3) {

  # Compile data
  all <- data.frame(cbind(y,x))

  # Split data:
  train <- all[1:round(cutoff*nrow(all),0),]; dim(train) # Training set
  test <- all[(round(cutoff*nrow(all),0)+1):nrow(all),]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- train[,-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- test[,-1]; dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))
  train <- data.frame(Y = train.y, train.x)

  # Model Fitting
  ### load libraries
  #library("gbm")
  model <- gbm::gbm(Y~.,                             # formula
                    data=train,                           # dataset
                    distribution="gaussian",              # see the help for other choices
                    n.trees=num.of.trees,                 # number of trees
                    #shrinkage=shrinkage,                  # shrinkage or learning rate,
                    #interaction.depth=interaction.depth,  # 1: additive model, 2: two-way interactions, etc.
                    #bag.fraction=0.5,                     # subsampling fraction, 0.5 is probably best
                    #train.fraction=1,                     # fraction of data for training,
                    #n.minobsinnode=10,                    # minimum total weight needed in each node
                    #cv.folds = 1,                         # do 3-fold cross-validation
                    #keep.data=TRUE,                       # keep a copy of the dataset with the object
                    verbose=FALSE                         # don't print out progress
  )
  best_iter <- gbm::gbm.perf(model, method="OOB", plot.it = FALSE)
  #Sum <- summary(model)

  # Make prediction on training:
  preds.train.prob <- model$fit
  train.mse <- sum((preds.train.prob - train.y)^2)/nrow(train)

  # Make prediction on testing:
  colnames(test.x) <- colnames(train.x)
  preds.prob <- gbm::predict.gbm(model, n.trees = num.of.trees, test.x)
  test.mse <- sum((preds.prob - test.y)^2)/nrow(test)

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, preds.prob)
  colnames(truth.vs.pred.prob) <- c("True_Test_Y", "Predicted_Test_Y")

  # Final output:
  return(
    list(
      Train.X = train.x,
      Train.Y = train.y,
      Test.X = test.x,
      Test.Y = test.y,
      Summary = model,
      Train = train,
      Test = test,
      Train.MSE = train.mse,
      Test.MSE = test.mse,
      Truth_and_Predicted = truth.vs.pred.prob
    )
  )
} # End of function
