#' @title Supervised Regression ML Algorithm using Linear Regression Predictor
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Linear_Regression_Regressor()
#' @export Linear_Regression_Regressor
#'
#' # Define function
Linear_Regression_Regressor <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  cutoff.coefficient = 1) {

  # Compile data
  all <- data.frame(cbind(y,x))

  # Split data:
  train <- all[1:round(cutoff*nrow(all),0),]; dim(train) # Training set
  test <- all[(round(cutoff*nrow(all),0)+1):nrow(all),]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- data.frame(train[,-1]); colnames(train.x) <- colnames(train)[-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- data.frame(test[,-1]); dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  # Modeling fitting:
  # GLM or # LM
  model <- lm(
    train.y ~.,
    data = train.x
  )
  sum <- summary(model)

  # Make prediction on training:
  preds.train.prob <- predict(model, train.x)
  train.mse <- sum((preds.train.prob - train.y)^2)/nrow(train)

  # Make prediction on testing:
  colnames(test.x) <- colnames(train.x)
  preds.prob <- predict(model, test.x)
  test.mse <- sum((preds.prob - test.y)^2)/nrow(test)

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, preds.prob)
  colnames(truth.vs.pred.prob) <- c("True_Test_Y", "Predicted_Test_Y")

  # Final output:
  return(
    list(
      Summary = sum,
      Train = train,
      Test = test,
      Train.MSE = train.mse,
      Test.MSE = test.mse,
      Truth_and_Predicted = truth.vs.pred.prob
    )
  )
} # End of function
