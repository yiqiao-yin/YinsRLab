#' @title Supervised Classification ML Algorithm using Adaboost Classifier
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Adaboost_Classifier()
#' @export Adaboost_Classifier
#'
#' # Define function
Adaboost_Classifier <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  tree_depth = 5,
  n_rounds = 100,
  cutoff.coefficient = 1) {

  # Compile Data
  all <- data.frame(cbind(y,x))

  # Split data:
  train_idx <- 1:round(cutoff*nrow(all),0)
  train <- data.frame(all[train_idx,]); dim(train) # Training set
  test <- all[-train_idx,]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- data.frame(train[,-1]);
  colnames(train.x) <- colnames(train)[-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  train <- data.frame(
    Y = train.y,
    train.x )
  test.x <- data.frame(test[,-1]); dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  # Modeling fitting:
  # GLM or # LM
  model <- JOUSBoost::adaboost(
    as.matrix(train.x),
    as.numeric(ifelse(train.y == 1, 1, -1)),
    tree_depth = tree_depth,
    n_rounds = n_rounds,
    verbose = TRUE)
  sum <- summary(model)

  # Make prediction on training:
  preds.train.prob <- predict(model, as.matrix(train.x))
  preds.train.prob.regular <- as.numeric(ifelse(preds.train.prob == 1, 1, 0))
  table.train <- as.matrix(cbind(preds.train.prob.regular,train.y))
  tab.train <- table(Y_Hat = table.train[,1], Y = table.train[,2])
  percent.train <- sum(diag(tab.train))/sum(tab.train)

  # ROC
  actuals <- train.y
  scores <- preds.train.prob.regular
  roc_obj.train <- pROC::roc(response = actuals, predictor = scores)
  auc.train <- roc_obj.train$auc

  # Make prediction on testing:
  colnames(test.x) <- colnames(train.x)
  preds.prob <- predict(model, test.x) # nrow(test.x)
  preds.test.prob.regular <- as.numeric(ifelse(preds.prob == 1, 1, 0))
  table <- as.matrix( cbind(preds,test.y) )

  # Compute accuracy:
  table <- table(Y_Hat = table[,1], Y = table[,2]); table
  percent <- sum(diag(table))/sum(table)

  # ROC
  actuals <- test.y
  scores <- preds.test.prob.regular
  roc_obj <- pROC::roc(response = actuals, predictor = scores)
  auc <- roc_obj$auc

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, preds.test.prob.regular)
  colnames(truth.vs.pred.prob) <- c("True Probability", "Predicted Probability")

  # Final output:
  return(
    list(
      Summary = model,
      Training.Accuracy = percent.train,
      Training.AUC = auc.train,
      Training.Confusion.Table = tab.train,
      Train_Y_Hat = preds.train.prob.regular,
      Train_Y_Truth = train.y,
      Test_Y_Hat = preds.test.prob.regular,
      Test_Y_Truth = test.y,
      Truth.vs.Predicted.Probabilities = truth.vs.pred.prob,
      Prediction.Table = table,
      Testing.Accuracy = percent,
      Testing.Error = 1 - percent,
      AUC = auc,
      Gini = auc * 2 - 1
    )
  )
} # End of function
