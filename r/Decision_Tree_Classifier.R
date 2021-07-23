#' @title Supervised Classification ML Algorithm using Decision Trees
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Decision_Tree_Classifier()
#' @export Decision_Tree_Classifier
#'
#' # Define function
Decision_Tree_Classifier <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = .9,
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
  train.y <- train[,1]; head(train.y)
  test.x <- test[,-1]; dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

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
  preds.train.discrete <- as.numeric(preds.train > mean(preds.train))
  table.train <- as.matrix(cbind(preds.train.discrete, train.y))
  tab.train <- table(Y_Hat = table.train[,1], Y = table.train[,2]); tab.train
  percent.train <- sum(diag(tab.train))/sum(tab.train); percent.train

  # ROC
  actuals <- train.y
  scores <- as.numeric(preds.train)
  roc_obj <- pROC::roc(response = actuals, predictor =  scores)
  auc.train <- roc_obj$auc; auc.train

  # Make prediction on testing:
  preds.probability <- predict(model, test.x)
  preds.binary <- as.numeric(preds.probability > mean(preds.probability))
  table <- as.matrix(cbind(YHat = preds.binary, Y = test.y)); dim(table); head(table)

  # Compute accuracy:
  table <- table(Y_Hat = table[,1], Y = table[,2]); table
  percent <- sum(diag(table))/sum(table); percent

  # ROC
  actuals <- test.y
  scores <- preds.binary
  roc_obj <- pROC::roc(response = actuals, predictor =  scores)
  auc <- roc_obj$auc; auc

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(Y = test.y, YHat = preds.probability)
  colnames(truth.vs.pred.prob) <- c("TrueProbability", "PredictedProbability")

  # Final output:
  return(
    list(
      Summary = model,
      Training.Accuracy = percent.train,
      Training.AUC = auc.train,
      Important.Variables = selected.variable,
      train.y.hat = preds.train,
      train.y = train.y,
      test.y.hat.original = preds.probability,
      test.y.hat = preds.binary,
      test.y.truth = test.y,
      test.y.errors = plyr::count(preds.binary - test.y),
      Prediction.Table = table,
      Testing.Accuracy = percent,
      Testing.Error = 1-percent,
      AUC = auc,
      Gini = auc*2 - 1,
      Truth.vs.Predicted.Probabilities = truth.vs.pred.prob
    )
  )
} # End of function
