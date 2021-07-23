#' @title Supervised Classification ML Algorithm using Gradient Descent Classifier
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Gradient_Descent_Classifier(x = all[, -1], y = all[, 1])
#' @export Gradient_Descent_Classifier
#'
#' # Define function
Gradient_Descent_Classifier <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  cutoff.coefficient = 1,
  alpha = 1e-2,
  earlyStop = TRUE,
  err_threshold = 0.5,
  num_iters = 1000,
  verbose = TRUE) {

  # Data
  all <- data.frame(cbind(y,x))

  # Split data:
  train_idx <- 1:round(cutoff*nrow(all),0)
  train <- data.frame(all[train_idx,]); dim(train) # Training set
  test <- all[-train_idx,]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- data.frame(train[,-1]); colnames(train.x) <- colnames(train)[-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- data.frame(test[,-1]); dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  # Gradient Descent:
  # squared error cost function
  cost <- function(X, y, theta) { sum( (as.matrix(X) %*% as.matrix(theta) - y)^2 ) / (2*length(y)) }

  # learning rate and iteration limit
  #alpha <- 0.01
  #num_iters <- 1000

  # keep history
  cost_history <- double(num_iters)
  theta_history <- list(num_iters)

  # initialize coefficients
  number.of.coeff <- ncol(train.x)
  theta <- matrix(rep(0,number.of.coeff+1), nrow=(number.of.coeff+1))

  # add a column of 1's for the intercept coefficient
  X <- cbind(train.x, 1)

  # gradient descent
  if (verbose) {print("Starting training ...")}
  if (verbose) {pb <- txtProgressBar(min = 0, max = num_iters, style = 3)}
  for (i in 1:num_iters) {
    error <- (as.matrix(X) %*% as.matrix(theta) - train.y)
    delta <- t(X) %*% as.matrix(error) / length(train.y)
    theta <- theta - alpha * delta
    unitCost <- cost(X, train.y, theta)
    cost_history[i] <- unitCost
    theta_history[[i]] <- theta
    TITLE <- paste0("Round=", i, ", Cost=", round(unitCost, 4), " ")
    if (verbose) {setTxtProgressBar(pb, i)}
    if (earlyStop && (unitCost < err_threshold)) {if (verbose) {print("Error threshold satisfied. Break loop.")}}
    if (earlyStop && (unitCost < err_threshold)) {break}
  } # Finished Gradient Descent

  # Make prediction on training:
  preds.train.prob <- as.matrix(X) %*% as.matrix(theta)
  preds.mean.train <- mean(preds.train.prob)
  preds.train <- ifelse(preds.train.prob > cutoff.coefficient*preds.mean.train, 1, 0)
  table.train <- as.matrix( cbind(preds.train, train.y) )
  tab.train <- table(Y_Hat = table.train[,1], Y = table.train[,2])
  percent.train <- sum(diag(tab.train))/sum(tab.train)

  # ROC
  actuals <- train.y
  scores <- preds.train.prob
  roc_obj <- pROC::roc(response = actuals, predictor = c(scores))
  roc_obj_train <- roc_obj
  auc.train <- roc_obj$auc

  # Make prediction on testing:
  colnames(test.x) <- colnames(train.x)
  X.new <- cbind(1, test.x)
  preds.prob <- as.matrix(X.new) %*% as.matrix(theta)
  preds.mean <- mean(preds.prob)
  preds <- ifelse(preds.prob > cutoff.coefficient*preds.mean, 1, 0)
  table <- as.matrix( cbind(preds, test.y) )
  dim(table); head(table)

  # Compute accuracy:
  table <- table(Y_Hat = table[,1], Y = table[,2]); table
  percent <- sum(diag(table))/sum(table)

  # ROC
  actuals <- test.y
  scores <- preds
  roc_obj <- pROC::roc(response = actuals, predictor = c(scores))
  roc_obj_test <- roc_obj
  auc <- roc_obj$auc

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, preds.prob)
  raw_score <- truth.vs.pred.prob[, 2]
  probs <- exp(raw_score)/(1 + exp(raw_score))
  truth.vs.pred.prob <- cbind(truth.vs.pred.prob, probs)
  colnames(truth.vs.pred.prob) <- c("True Label", "Raw Score", "Probability")

  # Final output:
  return(
    list(
      Summary = list(
        theta=theta,
        cost_history=cost_history,
        theta_history=theta_history),
      Training.Accuracy = percent.train,
      Training.AUC = auc.train,
      Train.Y.Hat = preds.train.prob,
      Train.Y.Truth = train.y,
      Test.Y.Hat = preds,
      Test.Y.Truth = test.y,
      Truth.vs.Predicted.Probabilities = truth.vs.pred.prob,
      Prediction.Table = table,
      Testing.Accuracy = percent,
      Testing.Error = 1-percent,
      AUC = auc,
      Gini = auc*2 - 1,
      ROC_Obj = list(
        roc_obj_train=roc_obj_train,
        roc_obj_test=roc_obj_test )
    )
  )
} # End of function
