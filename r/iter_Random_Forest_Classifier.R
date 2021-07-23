#' @title Supervised Classification ML Algorithm using iterative Random Forest
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  iter_Random_Forest_Classifier(x = all[, -1], y = all[, 1])
#' @export iter_Random_Forest_Classifier
#'
#' # Define function
iter_Random_Forest_Classifier <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = .9,
  num.tree = 5,
  num.iter = 5,
  num.bootstrap = 30,
  SV.cutoff = 1:10,
  verbatim = TRUE
) {

  # Library
  #library("iRF"); library("pROC")

  # Compile data:
  all <- data.frame(cbind(y, x))

  # Split data:
  train <- all[1:round(cutoff*nrow(all),0),]; dim(train) # Training set
  test <- all[(round(cutoff*nrow(all),0)+1):nrow(all),]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- train[,-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- test[,-1]; dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  # Modeling fitting:
  model <- iRF::iRF(
    x = as.matrix.data.frame(train.x),
    y = as.factor(train.y),
    xtest = as.matrix.data.frame(test.x),
    ytest = as.factor(test.y),
    n.iter = num.iter,
    ntree = num.tree,
    n.bootstrap = num.bootstrap,
    verbose = verbatim
  )
  sum <- model$rf.list[[num.iter]]

  # Important Features
  important.features <- model$rf.list[[length(model$rf.list)]]$importance
  important.features <- cbind(rownames(important.features), important.features)
  important.features <- important.features[order( as.numeric(as.character(important.features[, 2])), decreasing = TRUE), ]
  selected.variable <- important.features[SV.cutoff, 1]

  # Make prediction on training:
  preds.train <- sum$predicted  # colMeans(model$yhat.train)
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
  auc.train <- roc_obj$auc

  # Make prediction on testing:
  preds <- sum$test$predicted # colMeans(model$yhat.test)
  #preds.mean <- mean(preds)
  #preds <- ifelse(preds > preds.mean, 1, 0)
  table <- as.matrix(cbind(preds, test.y)); dim(table); head(table)

  # Compute accuracy:
  table <- table(Y_Hat = table[,1], Y = table[,2]); table
  percent <- sum(diag(table))/sum(table); percent

  # ROC
  actuals <- test.y
  scores <- sum$test$votes[,2] # colMeans(model$yhat.test)
  roc_obj <- pROC::roc(response = actuals, predictor =  scores)
  auc <- roc_obj$auc

  # Truth.vs.Predicted.Probabilities
  truth.vs.pred.prob <- cbind(test.y, sum$test$votes[,2])
  colnames(truth.vs.pred.prob) <- c("True Probability", "Predicted Probability")

  # Final output:
  return(
    list(
      Summary = sum,
      Important.Variables = selected.variable,
      Training.Accuracy = percent.train,
      Training.AUC = auc.train,
      train.y.hat = preds.train,
      train.y = train.y,
      test.y.hat = as.numeric(preds),
      test.y.truth = test.y,
      test.y.errors = plyr::count(as.numeric(preds) - test.y),
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
