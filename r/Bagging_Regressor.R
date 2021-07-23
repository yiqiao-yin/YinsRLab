#' @title Supervised Regression ML Algorithm using Bagging
#' @description This function accepts input of data and parameteres and produce output of regression results.
#' @param symbol
#' @return NULL
#' @examples  Bagging_Regressor()
#' @export Bagging_Regressor
#'
#' # Define function
Bagging_Regressor <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  nbagg = 10) {

  # Compile data
  all <- data.frame(cbind(y,x))

  # Split data:
  train_idx <- 1:round(cutoff*nrow(all),0)
  train <- all[train_idx,]; dim(train) # Training set
  test <- all[-train_idx,]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- data.frame(train[,-1]); colnames(train.x) <- colnames(train)[-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- data.frame(test[,-1]); dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  # Modeling fitting:
  # Bagging
  model <- ipred::bagging(
    formula = train$y~.,
    data = train,
    nbagg = nbagg,
    coob = TRUE,
    keepX = TRUE
  ); sum <- summary(model)

  # Make prediction on training:
  preds.train.prob <- predict(model, train.x)
  train.mse <- sum((preds.train.prob - train.y)^2)/nrow(train.x)

  # Make prediction on testing:
  colnames(test.x) <- colnames(train.x)
  preds.prob <- predict(model, test.x) # nrow(test.x)
  test.mse <- sum((preds.prob - test.y)^2)/nrow(test.x)

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, predict(model, test.x))
  colnames(truth.vs.pred.prob) <- c("True Probability", "Predicted Probability")

  # Final output:
  return(
    list(
      Summary = sum,
      Model = model,
      train.y.hat = preds.train.prob,
      train.y = train.y,
      test.y.hat = preds.prob,
      test.y.truth = test.y,
      Truth.vs.Predicted.Probabilities = truth.vs.pred.prob,
      trainMSE = train.mse,
      testMSE = test.mse
    )
  )
} # End of function
