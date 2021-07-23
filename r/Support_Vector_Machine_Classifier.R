#' @title Supervised Classification ML Algorithm using SVM Classifier
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Support_Vector_Machine_Classifier()
#' @export Support_Vector_Machine_Classifier
#'
#' # Define function
Support_Vector_Machine_Classifier <- function(
  x = all[, -1],
  y = all[ ,1],
  cutoff = 0.9,
  c = 0.01,
  g = 0.01,
  cutoff.coefficient = 1){

  # Compile
  all <- data.frame(cbind(y,x))

  # Split:
  train <- all[1:round(cutoff*nrow(all),0),]; dim(train) # Training set
  test <- all[(round(cutoff*nrow(all),0)+1):nrow(all),]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- train[,-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- test[,-1]; dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  ## Apply SVM
  # Ex: c<-1; g<-1
  svm.fit <- e1071::svm(
    formula = train.y ~.,
    data = train.x,
    type = "C-classification",
    kernel = "sigmoid",
    cost = c,
    gamma = g
  )

  # Make prediction on training:
  preds.train.prob <- predict(svm.fit, train.x); preds.train.prob <- as.numeric(as.character(preds.train.prob))
  preds.mean.train <- mean(preds.train.prob)
  preds.train <- ifelse(preds.train.prob > cutoff.coefficient*preds.mean.train, 1, 0)
  table.train <- as.matrix(cbind(preds.train, train.y))
  tab.train <- table(Y_Hat = table.train[,1], Y = table.train[,2])
  percent.train <- sum(diag(tab.train))/sum(tab.train)

  # ROC
  actuals <- train.y
  scores <- preds.train.prob
  roc_obj.train <- pROC::roc(response = actuals, predictor =  scores)
  auc.train <- roc_obj.train$auc

  # Make prediction on testing:
  colnames(test.x) <- colnames(train.x)
  preds.prob <- predict(svm.fit, test.x); preds.prob <- as.numeric(as.character(preds.prob))
  preds.mean <- mean(preds.prob)
  preds <- ifelse(preds.prob > cutoff.coefficient*preds.mean, 1, 0)
  table <- as.matrix(cbind(preds, test.y))
  dim(table); head(table)

  # Compute accuracy:
  table <- table(Y_Hat = table[,1], Y = table[,2]); table
  percent <- sum(diag(table))/sum(table)

  # ROC
  actuals <- test.y
  scores <- preds.prob
  roc_obj <- pROC::roc(response = actuals, predictor =  scores)
  auc <- roc_obj$auc

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, preds.prob)
  colnames(truth.vs.pred.prob) <- c("True Probability", "Predicted Probability")

  # Final output:
  return(
    list(
      Summary = list(svm.fit, summary(svm.fit)),
      Training.Accuracy = percent.train,
      Training.AUC = auc.train,
      #Train.ROC = plot(roc_obj.train),
      train.y.hat = preds.train.prob,
      train.y = train.y,
      test.y.hat = preds,
      test.y.truth = test.y,
      Truth.vs.Predicted.Probabilities = truth.vs.pred.prob,
      Prediction.Table = table,
      Testing.Accuracy = percent,
      Testing.Error = 1-percent,
      AUC = auc,
      Gini = auc*2 - 1#,
      #Test.ROC = plot(roc_obj)
    )
  )
} ## End of function
