#' @title Supervised Classification ML Algorithm using Logistic Regression
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Logistic(all = all)
#' @export Logistic
#'
#' # Define function
Logistic <- function(
  all = all,
  cutoff = 0.9,
  fam = binomial,
  cutoff.coefficient = 1) {

  # Library
  library(pROC)

  # Split data:
  train <- all[1:round(cutoff*nrow(all),0),]; dim(train) # Training set
  test <- all[(round(cutoff*nrow(all),0)+1):nrow(all),]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- data.frame(train[,-1]); colnames(train.x) <- colnames(train)[-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- data.frame(test[,-1]); dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  # Modeling fitting:
  # GLM or # LM
  model <- glm(
    train.y ~.,
    data = train.x,
    family = fam
    # gaussian
    # binomial
    # quasibinomial
  )
  sum <- summary(model)

  # Make prediction on training:
  preds.train.prob <- predict(model, train.x)
  preds.mean.train <- mean(preds.train.prob)
  preds.train <- ifelse(preds.train.prob > cutoff.coefficient*preds.mean.train, 1, 0)
  table.train <- as.matrix(cbind(preds.train, train.y))
  tab.train <- table(Y_Hat = table.train[,1], Y = table.train[,2])
  percent.train <- sum(diag(tab.train))/sum(tab.train)

  # ROC
  actuals <- train.y
  scores <- preds.train.prob
  roc_obj.train <- roc(response = actuals, predictor =  scores)
  auc.train <- roc_obj.train$auc

  # Make prediction on testing:
  colnames(test.x) <- colnames(train.x)
  preds.prob <- predict(model, test.x) # nrow(test.x)
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
  roc_obj <- roc(response = actuals, predictor =  scores)
  auc <- roc_obj$auc

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, predict(model, test.x))
  colnames(truth.vs.pred.prob) <- c("True Probability", "Predicted Probability")

  # Final output:
  return(
    list(
      Summary = sum,
      Training.Accuracy = percent.train,
      Training.AUC = auc.train,
      #Train.ROC = plot(roc_obj.train),
      Train.Y = train.y,
      Train.Y.Hat = preds.train.prob,
      Test.Y = test.y,
      Test.Y.Hat = preds,
      Truth.vs.Predicted.Probabilities = truth.vs.pred.prob,
      Prediction.Table = table,
      Testing.Accuracy = percent,
      Testing.Error = 1 - percent,
      AUC = auc,
      Gini = auc * 2 - 1#,
      #Test.ROC = plot(roc_obj)
    )
  )
} # End of function
