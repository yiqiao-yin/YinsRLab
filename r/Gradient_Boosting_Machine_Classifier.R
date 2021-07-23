#' @title Supervised Classification ML Algorithm using Gradient Boosting Machine
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  Gradient_Boosting_Machine_Classifier(x = all[, -1], y = all[, 1])
#' @export Gradient_Boosting_Machine_Classifier
#'
#' # Define function
Gradient_Boosting_Machine_Classifier <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  cutoff.coefficient = 1,
  num.of.trees = 10,
  bag.fraction = 0.5,
  shrinkage = 0.05,
  interaction.depth = 3,
  cv.folds = 5,
  verbatim = TRUE) {

  # Compile data
  all <- data.frame(cbind(y,x))

  # Split data:
  train_idx <- 1:round(cutoff*nrow(all),0)
  train <- data.frame(all[train_idx,]); dim(train) # Training set
  test <- all[-train_idx,]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- train[,-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- test[,-1]; dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))
  train <- data.frame(Y = train.y, train.x)

  # Model Fitting
  ### load libraries
  #library("gbm")
  model <- gbm::gbm(Y~.,                             # formula
               data=train,                           # dataset
               distribution="gaussian",              # see the help for other choices
               n.trees=num.of.trees,                 # number of trees
               shrinkage=shrinkage,                  # shrinkage or learning rate,
               interaction.depth=interaction.depth,  # 1: additive model, 2: two-way interactions, etc.
               bag.fraction=bag.fraction,            # subsampling fraction, 0.5 is probably best
               train.fraction=1,                     # fraction of data for training,
               n.minobsinnode=10,                    # minimum total weight needed in each node
               cv.folds = cv.folds,                  # do 3-fold cross-validation
               keep.data=TRUE,                       # keep a copy of the dataset with the object
               verbose=verbatim                         # don't print out progress
  )
  best_iter <- gbm::gbm.perf(model, method="OOB", plot.it = FALSE)
  #Sum <- summary(model)

  # Make prediction on training:
  preds.train.prob <- model$fit
  preds.mean.train <- mean(preds.train.prob)
  preds.train <- ifelse(preds.train.prob > cutoff.coefficient*preds.mean.train, 1, 0)
  table.train <- as.matrix( cbind(preds.train, train.y) )
  tab.train <- table(Y_Hat = table.train[,1], Y = table.train[,2])
  percent.train <- sum(diag(tab.train))/sum(tab.train)

  # ROC
  actuals <- train.y
  scores <- preds.train.prob
  roc_obj <- pROC::roc(response = actuals, predictor =  scores)
  auc.train <- roc_obj$auc

  # Make prediction on testing:
  colnames(test.x) <- colnames(train.x)
  preds.prob <- gbm::predict.gbm(model, n.trees = num.of.trees, test.x)
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
      Test.Y.Hat.Original = preds.prob,
      Test.Y.Hat.Discrete = preds,
      Test.Y.Errors = plyr::count(preds - test.y),
      Summary = model,
      Train.AUC = auc.train,
      Train.Confusion.Matrix = tab.train,
      Train.Accuracy = percent.train,
      Test.Confusion.Matrix = table,
      Test.Accuracy = percent,
      Test.AUC = auc,
      Truth.Pred.Prob = truth.vs.pred.prob
    )
  )
} # End of function
