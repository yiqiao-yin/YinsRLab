#' @title Supervised Classification ML Algorithm using Lasso/Ridge and Logistic Classifier
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Lasso_Ridge_Logistic_Classifier()
#' @export Lasso_Ridge_Logistic_Classifier
#'
#' # Define function
Lasso_Ridge_Logistic_Classifier <- function(
  x = all[, -1],
  y = all[, 1],
  alpha = 1,
  cutoff = .67,
  fam = binomial,
  cutoff.coefficient = 1) {

  # Library
  #library(glmnet); library(pROC)

  # Compile
  all <- data.frame(cbind(y,x))

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
  model <- glmnet::cv.glmnet(x = as.matrix(train.x), y = train.y)
  MSE_Plot <- plot(model)
  penalty <- model$lambda.min # optimal lambda
  model_new <- glmnet::glmnet(as.matrix(train.x), train.y, alpha = alpha, lambda = penalty)
  selected_variable <- which(as.matrix(coef(model_new))[-1, ] != 0)
  all <- data.frame(cbind(all[,1], all[, -1][,c(selected_variable)]))
  colnames(all)[1] <- "y"

  # Split data:
  train <- all[1:(cutoff*nrow(all)),]; dim(train) # Training set
  test <- all[(cutoff*nrow(all)+1):nrow(all),]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- data.frame(train[,-1]); colnames(train.x) <- colnames(train)[-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- data.frame(test[,-1]); dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  # Modeling fitting:
  # GLM or # LM
  log.model <- glm(
    train.y ~.,
    data = train.x,
    family = fam)

  # Make prediction on training:
  preds.train.prob <- predict(log.model, train.x)
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
  preds.prob <- predict(log.model, test.x) # nrow(test.x)
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
      Summary = list(Optimal_Lambda = penalty, Selected_Variables = selected_variable),
      X = all[, -1],
      Y = all[, 1],
      Training.Accuracy = percent.train,
      Training.AUC = auc.train,
      #Train.ROC = plot(roc_obj.train),
      #MSE_Plot = MSE_Plot,
      train.y.hat = preds.train.prob,
      train.y = train.y,
      y.hat = preds,
      y.truth = test.y,
      Truth.vs.Predicted.Probabilities = truth.vs.pred.prob,
      Prediction.Table = table,
      Testing.Accuracy = percent,
      Testing.Error = 1-percent,
      AUC = auc,
      Gini = auc*2 - 1#,
      #Test.ROC = plot(roc_obj)
    )
  )
} # End of function
