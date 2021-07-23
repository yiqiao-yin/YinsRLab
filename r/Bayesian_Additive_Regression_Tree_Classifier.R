#' @title Supervised Classification ML Algorithm using Baeysian Additive Regression Tree (BART)
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Bayesian_Additive_Regression_Tree_Classifier(x = all[, -1], y = all[, 1])
#' @export Bayesian_Additive_Regression_Tree_Classifier
#'
#' # Define function
Bayesian_Additive_Regression_Tree_Classifier <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = .9,
  num.tree = 5,
  num.cut = 100,
  cutoff.coefficient = 1,
  SV.cutoff = 1:3,
  verbatim = TRUE
) {

  # Library
  #library("BayesTree"); library("pROC")

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
    verbose = verbatim,
    ntree = num.tree,
    numcut = num.cut)
  sum <- summary(model)

  # Feature Selection (by varcounts)
  important.features <- cbind(Variable = c(1:ncol(train.x)), Count = apply(model$varcount, 2, sum))
  important.features <- important.features[order( as.numeric(as.character(important.features[, 2])), decreasing = TRUE), ]
  selected.variable <- important.features[SV.cutoff, 1]

  # Make prediction on training:
  preds.train.prob <- colMeans(model$yhat.train)
  preds.mean.train <- mean(preds.train.prob)
  preds.train <- ifelse(preds.train.prob > cutoff.coefficient*preds.mean.train, 1, 0)
  table.train <- as.matrix(cbind(preds.train, train.y))
  tab.train <- table(Y_Hat = table.train[,1], Y = table.train[,2]); tab.train
  percent.train <- sum(diag(tab.train))/sum(tab.train)

  # ROC
  actuals <- train.y
  scores <- as.numeric(preds.train.prob)
  roc_obj <- pROC::roc(response = actuals, predictor =  scores)
  auc.train <- roc_obj$auc

  # Make prediction on testing:
  preds.prob <- colMeans(model$yhat.test)
  preds.mean <- mean(preds.prob)
  preds <- ifelse(preds.prob > cutoff.coefficient*preds.mean, 1, 0)
  table <- as.matrix(cbind(preds, test.y))
  dim(table); head(table)

  # Compute accuracy:
  table <- table(Y_Hat = table[,1], Y = table[,2]); table
  percent <- sum(diag(table))/sum(table)

  # ROC
  actuals <- test.y
  scores <- colMeans(model$yhat.test)
  roc_obj <- pROC::roc(response = actuals, predictor =  scores)
  auc <- roc_obj$auc

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, colMeans(model$yhat.test))
  colnames(truth.vs.pred.prob) <- c("True Probability", "Predicted Probability")

  # Final output:
  return(
    list(
      Summary = model,
      Important.Variables = colnames(train.x)[selected.variable],
      Training.Accuracy = percent.train,
      Training.AUC = auc.train,
      Train.Y.Hat = preds.train.prob,
      Train.Y.Truth = train.y,
      Test.Y.Hat = preds,
      Test.Y.Truth = test.y,
      Prediction.Table = table,
      Testing.Accuracy = percent,
      Testing.Error = 1-percent,
      AUC = auc,
      Gini = auc*2 - 1,
      Truth.vs.Predicted.Probabilities = truth.vs.pred.prob#,
      #AUC.Plot = plot(roc_obj, main = paste0("AUC=",round(auc,3)))
    )
  )
} # End of function
