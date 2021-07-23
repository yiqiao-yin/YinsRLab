#' @title Supervised Regression ML Algorithm using Baeysian Additive Regression Tree (BART)
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Bayesian_Additive_Regression_Tree_Regressor(x = all[, -1], y = all[, 1])
#' @export Bayesian_Additive_Regression_Tree_Regressor
#'
#' # Define function
Bayesian_Additive_Regression_Tree_Regressor <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = .9,
  num.tree = 5,
  num.cut = 100,
  SV.cutoff = 1:10
) {

  # Compile data:
  all <- as.matrix(cbind(y, x))

  # Split data:
  train_idx <- 1:round(cutoff*nrow(all),0)
  train <- all[train_idx,]; dim(train) # Training set
  test <- all[-train_idx,]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- train[,-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- test[,-1]; dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  # Modeling fitting:
  model <- BayesTree::bart(
    x.train = train.x,
    y.train = train.y,
    x.test  = test.x,
    verbose = FALSE,
    ntree = num.tree,
    numcut = num.cut)
  sum <- summary(model)

  # Feature Selection (by varcounts)
  important.features <- cbind(Variable = c(1:ncol(train.x)),
                              Count = apply(model$varcount, 2, sum))
  important.features <- important.features[order(
    as.numeric(as.character(important.features[, 2])), decreasing = TRUE), ]
  selected.variable <- important.features[SV.cutoff, 1]

  # Make prediction on training:
  preds.train.prob <- colMeans(model$yhat.train)
  train.mse <- sum((preds.train.prob - train.y)^2)/nrow(train)

  # Make prediction on testing:
  preds.prob <- colMeans(model$yhat.test)
  test.mse <- sum((preds.prob - test.y)^2)/nrow(test)

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, colMeans(model$yhat.test))
  colnames(truth.vs.pred.prob) <- c("True_Test_Y", "Predicted_Test_Y")

  # Final output:
  return(
    list(
      Summary = model,
      Important.Variables = colnames(train.x)[selected.variable],
      Train = train,
      Test = test,
      Train.MSE = train.mse,
      Test.MSE = test.mse,
      Truth_and_Predicted = truth.vs.pred.prob
    )
  )
} # End of function
