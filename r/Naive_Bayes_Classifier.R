#' @title Supervised Classification ML Algorithm using Naive Bayes
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Naive_Bayes_Classifier(x = all[, -1], y = all[, 1])
#' @export Naive_Bayes_Classifier
#'
#' # Define function
Naive_Bayes_Classifier <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  cutoff.coefficient = 1) {

  # Compile data
  all <- data.frame(cbind(y,x))

  # Split data:
  train <- all[1:round(cutoff*nrow(all),0),]; dim(train) # Training set
  test <- all[(round(cutoff*nrow(all),0)+1):nrow(all),]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- train[,-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- test[,-1]; dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))
  train <- data.frame(Y = train.y, train.x)

  # Model Fitting
  NB.Classifier <- naivebayes::naive_bayes(as.factor(Y)~.,usekernel=T, data=train)
  #NBclassfier <- e1071::naiveBayes(Y~., data = train)
  #Conditional.Prob.from.NBClassifier <- NBclassfier

  # Make prediction on training:
  preds.train.prob <- as.numeric(as.character(predict(NB.Classifier, newdata = train.x, type = "class")))
  preds.mean.train <- mean(preds.train.prob)
  preds.train <- ifelse(preds.train.prob > cutoff.coefficient*preds.mean.train, 1, 0)
  table.train <- as.matrix( cbind(preds.train, train.y) )
  tab.train <- table(Naive_Bayes_Classifier = table.train[,1], Y = table.train[,2])
  percent.train <- sum(diag(tab.train))/sum(tab.train)

  # ROC
  actuals <- train.y
  scores <- preds.train.prob
  roc_obj <- pROC::roc(response = actuals, predictor =  scores)
  auc.train <- roc_obj$auc

  # Make prediction on testing:
  colnames(test.x) <- colnames(train.x)
  preds.prob <- as.numeric(as.character(predict(NB.Classifier, newdata = test.x, type = "class")))
  preds.mean <- mean(preds.prob)
  preds <- ifelse(preds.prob > cutoff.coefficient*preds.mean, 1, 0)
  table <- as.matrix( cbind(preds, test.y) )
  dim(table); head(table)

  # Compute accuracy:
  table <- table(Naive_Bayes_Classifier = table[,1], Y = table[,2]); table
  percent <- sum(diag(table))/sum(table)

  # ROC
  actuals <- test.y
  scores <- preds
  roc_obj <- pROC::roc(response = actuals, predictor =  scores)
  auc <- roc_obj$auc

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, preds.prob)
  colnames(truth.vs.pred.prob) <- c("True Probability", "Predicted Probability")

  # Final output:
  return(
    list(
      Train.X = train.x,
      Train.Y = train.y,
      Train.Y.Hat = preds.train.prob,
      Test.X = test.x,
      Test.Y = test.y,
      Test.Y.Hat = preds.prob,
      Summary = NB.Classifier,
      Train.AUC = auc.train,
      Train.Confusion.Matrix = tab.train,
      Train.Accuracy = percent.train,
      Test.Confusion.Matrix = table,
      Test.Accuracy = percent,
      Test.AUC = auc,
      Turh.Pred.Prob = truth.vs.pred.prob
    )
  )
} # End of function
