#' @title Supervised Regression ML Algorithm using Gradient Descent Predictor
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Gradient_Descent_Regressor(x = all[, -1], y = all[, 1])
#' @export Gradient_Descent_Regressor
#'
#' # Define function
Gradient_Descent_Regressor <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  alpha = 0.01,
  num_iters = 1000) {

  # Data
  all <- data.frame(cbind(y,x))

  # Split data:
  train <- all[1:round(cutoff*nrow(all),0),]; dim(train) # Training set
  test <- all[(round(cutoff*nrow(all),0)+1):nrow(all),]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- data.frame(train[,-1]); colnames(train.x) <- colnames(train)[-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- data.frame(test[,-1]); dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  # Gradient Descent:
  # squared error cost function
  cost <- function(X, y, theta) { sum( (as.matrix(X) %*% as.matrix(theta) - y)^2 ) / (2*length(y)) }

  # keep history
  cost_history <- double(num_iters)
  theta_history <- list(num_iters)

  # initialize coefficients
  number.of.coeff <- ncol(train.x)
  theta <- matrix(rep(0,number.of.coeff+1), nrow=(number.of.coeff+1))

  # add a column of 1's for the intercept coefficient
  X <- cbind(1, train.x)

  # gradient descent
  for (i in 1:num_iters) {
    error <- (as.matrix(X) %*% as.matrix(theta) - train.y)
    delta <- t(X) %*% as.matrix(error) / length(train.y)
    theta <- theta - alpha * delta
    cost_history[i] <- cost(X, train.y, theta)
    theta_history[[i]] <- theta
  } # Finished Gradient Descent

  # Make prediction on training:
  preds.train.prob <- as.matrix(X) %*% as.matrix(theta)
  train.mse <- sum((preds.train.prob - train.y)^2)/nrow(train)

  # Make prediction on testing:
  colnames(test.x) <- colnames(train.x)
  X.new <- cbind(1, test.x)
  preds.prob <- as.matrix(X.new) %*% as.matrix(theta)
  test.mse <- sum((preds.prob - test.y)^2)/nrow(test)

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, preds.prob)
  colnames(truth.vs.pred.prob) <- c("True_Test_Y", "Predicted_Test_Y")

  # Final output:
  return(
    list(
      Weights = theta,
      Train = train,
      Test = test,
      Train.MSE = train.mse,
      Test.MSE = test.mse,
      Truth_and_Predicted = truth.vs.pred.prob
    )
  )
} # End of function
