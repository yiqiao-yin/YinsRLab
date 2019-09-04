#' @title Supervised Classification ML Algorithm using Random Forest
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Random_Forest_Classifier(x = all[, -1], y = all[, 1])
#' @export Random_Forest_Classifier
#'
#' # Define function
Random_Forest_Classifier <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = .9,
  num.tree = 10,
  num.try = sqrt(ncol(all)),
  cutoff.coefficient = 1,
  SV.cutoff = 1:10
) {

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

  # Modeling fitting:
  model <- randomForest::randomForest(
    x = as.matrix(train.x),
    y = as.factor(train.y),
    xtest = as.matrix(test.x),
    ytest = as.factor(test.y),
    ntree = num.tree,
    mtry = num.try
  )
  sum <- summary(model)

  # Extract imporance
  feature.and.score <- data.frame(model$importance)
  feature.score <- feature.and.score[order(feature.and.score, decreasing = TRUE), ]
  feature.order <- rownames(feature.and.score)[order(feature.and.score, decreasing = TRUE)]
  new.feature.and.score <- data.frame(cbind(feature.order, feature.score))
  head(new.feature.and.score)
  #SV.cutoff = 1:5
  selected.variable <- feature.order[SV.cutoff]
  selected.variable

  # Make prediction on training:
  preds.train <- model$predicted
  preds.train[is.na(preds.train) == TRUE] <- 0
  #preds.mean.train <- mean(preds.train)
  #preds.train <- ifelse(preds.train > preds.mean.train, 1, 0)
  table.train <- as.matrix(cbind(preds.train, train.y))
  tab.train <- table(Y_Hat = table.train[,1], Y = table.train[,2]); tab.train
  percent.train <- sum(diag(tab.train))/sum(tab.train); percent.train

  # ROC
  actuals <- train.y
  scores <- as.numeric(preds.train)
  roc_obj <- pROC::roc(response = actuals, predictor =  scores)
  auc.train <- roc_obj$auc; auc.train

  # Make prediction on testing:
  #preds.binary <- model$test$predicted # colMeans(model$yhat.test)
  preds.probability <- model$test$votes[,2]
  preds.mean <- mean(preds.probability)
  preds.binary <- ifelse(preds.probability > cutoff.coefficient*preds.mean, 1, 0)
  table <- as.matrix(cbind(preds.binary, test.y)); dim(table); head(table)

  # Compute accuracy:
  table <- table(Y_Hat = table[,1], Y = table[,2]); table
  percent <- sum(diag(table))/sum(table); percent

  # ROC
  actuals <- test.y
  scores <- preds.binary
  roc_obj <- pROC::roc(response = actuals, predictor =  scores)
  auc <- roc_obj$auc; auc

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, model$test$votes[,2])
  colnames(truth.vs.pred.prob) <- c("True Probability", "Predicted Probability")

  # Final output:
  return(
    list(
      Summary = model,
      Training.Accuracy = percent.train,
      Training.AUC = auc.train,
      Important.Variables = selected.variable,
      train.y.hat = preds.train,
      train.y = train,
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
      #AUC.Plot = plot(
      #  1 - spec, sens, type = "l", col = "red",
      #  ylab = "Sensitivity", xlab = "1 - Specificity")
    )
  )
} # End of function
