#' @title Supervised Regression ML Algorithm using Decision Trees
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Decision_Tree_Regressor()
#' @export Decision_Tree_Regressor
#'
#' # Define function
Decision_Tree_Regressor <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = .9,
  num.tree = 10,
  num.try = sqrt(ncol(all)),
  cutoff.coefficient = 1,
  SV.cutoff = 1:ncol(all[, -1])
) {

  # Compile data
  all <- data.frame(cbind(y,x))

  # Split data:
  trainIdx <- 1:round(cutoff*nrow(all),0)
  train <- all[trainIdx, ]; dim(train) # Training set
  test <- all[-trainIdx, ]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- train[,-1]; dim(train.x)
  train.y <- train[,1]; length(train.y)
  test.x <- test[,-1]; dim(test.x)
  test.y <- test[,1]; length(test.y)

  # Modeling fitting:
  model <- rpart::rpart(
    y~.,
    data = train)
  sum <- summary(model)

  # Extract imporance
  feature.and.score <- data.frame(model$variable.importance)
  selected.variable <- feature.and.score
  selected.variable

  # Make prediction on training:
  preds.train <- predict(model, train.x)
  trainMSE <- sum((cbind(preds.train) - cbind(train.y))^2)

  # Make prediction on testing:
  #preds.binary <- model$test$predicted # colMeans(model$yhat.test)
  preds.probability <- predict(model, test.x)
  testMSE <- sum((cbind(preds.probability) - cbind(test.y))^2)

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(testY = test.y, testYhat = preds.probability)
  colnames(truth.vs.pred.prob) <- c("TrueProbability", "PredictedProbability")

  # Final output:
  return(
    list(
      Summary = model,
      Important.Variables = selected.variable,
      trainX = train.x,
      trainY = train.y,
      trainYhat = preds.train,
      testX = test.x,
      testY = test.y,
      testYhat = preds.probability,
      TrainingMSE = trainMSE,
      TestingMSE = testMSE,
      Truth.vs.Predicted.Probabilities = truth.vs.pred.prob
    )
  )
} # End of function
