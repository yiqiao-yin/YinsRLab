#' @title Supervised Regression ML Algorithm using SVM Predictor
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Support_Vector_Machine_Regressor()
#' @export Support_Vector_Machine_Regressor
#'
#' # Define function
Support_Vector_Machine_Regressor <- function(
  x = all[, -1],
  y = all[ ,1],
  cutoff = 0.9,
  c = 0.01,
  g = 0.01){

  # Compile
  all <- data.frame(cbind(y,x))

  # Split:
  train <- all[1:round(cutoff*nrow(all),0),]; dim(train) # Training set
  test <- all[(round(cutoff*nrow(all),0)+1):nrow(all),]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- train[,-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- test[,-1]; dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  ## Apply SVM
  # Ex: c<-1; g<-1
  svm.fit <- e1071::svm(
    formula = train.y ~.,
    data = train.x,
    type = "nu-regression",
    kernel = "sigmoid",
    cost = c,
    gamma = g
  )

  # Make prediction on training:
  preds.train.prob <- predict(svm.fit, train.x); preds.train.prob <- as.numeric(as.character(preds.train.prob))
  train.mse <- sum((preds.train.prob - train.y)^2)/nrow(train)

  # Make prediction on testing:
  colnames(test.x) <- colnames(train.x)
  preds.prob <- predict(svm.fit, test.x); preds.prob <- as.numeric(as.character(preds.prob))
  test.mse <- sum((preds.prob - test.y)^2)/nrow(test)

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, preds.prob)
  colnames(truth.vs.pred.prob) <- c("True_Test_Y", "Predicted_Test_Y")

  # Final output:
  return(
    list(
      Summary = list(svm.fit, summary(svm.fit)),
      Summary = sum,
      Train = train,
      Test = test,
      Train.MSE = train.mse,
      Test.MSE = test.mse,
      Truth_and_Predicted = truth.vs.pred.prob
    )
  )
} ## End of function
