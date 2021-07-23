#' @title Supervised Regression ML Algorithm using Lasso/Ridge and Logistic Classifier
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Lasso_Ridge_Regression_Regressor()
#' @export Lasso_Ridge_Regression_Regressor
#'
#' # Define function
Lasso_Ridge_Regression_Regressor <- function(
  x = all[, -1],
  y = all[, 1],
  alpha = 1,
  cutoff = .67,
  cutoff.coefficient = 1) {

  # Library
  #library(glmnet); library(pROC)

  # Compile
  all <- data.frame(cbind(y,x))

  # Split data:
  train <- all[1:(cutoff*nrow(all)),]; dim(train) # Training set
  test <- all[(cutoff*nrow(all)+1):nrow(all),]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- data.frame(train[,-1]); colnames(train.x) <- colnames(train)[-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- data.frame(test[,-1]); dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  # Modeling fitting:
  # GLM or # LM
  model <- glmnet::cv.glmnet(x = as.matrix(train.x), y = train.y)
  MSE_Plot <- plot(model)
  penalty <- model$lambda.min # optimal lambda
  model_new <- glmnet::glmnet(as.matrix(train.x), train.y, alpha = alpha, lambda = penalty)
  selected_variable <- which(as.matrix(coef(model_new))[-1, ] != 0)
  all <- data.frame(cbind(all[,1], all[, -1][,c(selected_variable)]))
  colnames(all)[1] <- "y"

  # Split data:
  train <- all[1:(cutoff*nrow(all)),]; dim(train) # Training set
  test <- all[(cutoff*nrow(all)+1):nrow(all),]; dim(test) # Testing set

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
  test.mse <- sum((preds.prob - test.y)^2)/nrow(test.x)

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, preds.prob)
  colnames(truth.vs.pred.prob) <- c("True_Test_Y", "Predicted_Test_Y")

  # Final output:
  return(
    list(
      Summary = list(Optimal_Lambda = penalty, Selected_Variables = selected_variable),
      X = all[, -1],
      Y = all[, 1],
      LinearRegressionSummary = sum,
      Train = train,
      Test = test,
      Train.MSE = train.mse,
      Test.MSE = test.mse,
      Truth_and_Predicted = truth.vs.pred.prob
    )
  )
} # End of function
