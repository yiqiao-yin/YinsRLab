#' @title Supervised Parametric Feature Selection using QDA
#' @description This function accepts input of data and parameters and produce output of new data.
#' @param symbol
#' @return NULL
#' @examples  QDA_Predictor()
#' @export QDA_Predictor
#'
#' # Define function
QDA_Predictor <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = cutoff,
  scaleData = TRUE,
  plotGraph = TRUE
) {

  # Compile data
  all <- data.frame(cbind(y,x))
  if (scaleData) {all = data.frame(y=all[,1], scale(all[,-1]))}

  # Split
  train <- all[1:round(cutoff*nrow(all),0),]; dim(train) # Training set
  test <- all[(round(cutoff*nrow(all),0)+1):nrow(all),]; dim(test) # Testing set

  # Fit QDA model
  model <- MASS::qda(y~., data=train)
  y_train = train[, 1]
  trained = predict(model, train)
  y_train_hat = as.numeric(trained$class) - 1
  mse_train = mean((y_train - y_train_hat)^2)

  # Use LDA model to make predictions on test data
  predicted <- predict(model, test)
  y_test = test[, 1]
  y_test_hat = as.numeric(predicted$class) - 1
  mse_test = mean((y_test - y_test_hat)^2)

  # Plot
  if (plotGraph) {
    # create plot
    klaR::partimat(
      as.factor(y) ~ .,
      data = train,
      method = "qda",
      plot.matrix = TRUE,
      col.correct = 'green',
      col.wrong = 'red' )
    klaR::partimat(
      as.factor(y) ~ .,
      data = test,
      method = "qda",
      plot.matrix = TRUE,
      col.correct = 'green',
      col.wrong = 'red' )
  }

  # Return
  return(list(
    dataInput = cbind(y=y, x),
    dataUsed = all,
    model = model,
    y_train = y_train,
    y_test = y_test,
    y_train_hat = y_train_hat,
    y_test_hat = y_test_hat,
    mse = list(trainMSE = mse_train, testMSE = mse_test)
  ))
} # End of function
