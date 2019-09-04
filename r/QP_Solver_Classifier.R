#' @title Supervised Classification ML Algorithm using Adaboost Classifier
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  QP_Solver_Classifier()
#' @export QP_Solver_Classifier
#'
#' # Define function
QP_Solver_Classifier <- function(
  x = all[, -1],
  y = all[, 1],
  seed = 1,
  cutoff.coefficient = 1,
  cutoff = .9
) {
  # Define function
  search.for.weights <- function(
    seed = 1,
    all = all,
    cutoff.coefficient = 1
  ) {
    # Set up
    set.seed(seed)
    X <- as.matrix(all[,-1], nrow=nrow(all[,-1]))
    Y <- all[,1]
    num.col <- ncol(X)

    # Solve Optimization Problem
    Dmat = t(X) %*% X
    Amat = t(diag(num.col))
    bvec = rep(0, num.col); # bvec[num.col] <- 1 # Constraints: only positive weights
    dvec = t(Y) %*% X
    solution <- quadprog::solve.QP(Dmat = Dmat, dvec = dvec, Amat = Amat, bvec = bvec, meq = 0, factorized = T)

    # Output
    weights.normal.sol <- solution$solution
    weights.un.sol <- solution$unconstrained.solution
    weights.lagrange <- solution$Lagrangian
    # Here user can change to any one of the three types of weights
    weights.sol <- weights.normal.sol
    weights.sol <- weights.sol/sum(weights.sol) # Normalized; and sum to 1

    # Compute Response
    Y_hat <- matrix(0, nrow = nrow(X))
    for (i in 1:nrow(X)) {
      Y_hat[i, ] <- sum(X[i, ]*t(weights.sol))
    }
    compare_Y_Y_hat_original <- rbind(Y_hat = t(Y_hat), Truth = Y);
    compare_Y_Y_hat_original = t(compare_Y_Y_hat_original)
    Y_hat = ifelse(Y_hat > cutoff.coefficient*mean(Y_hat), 1, 0)

    # Output
    compare_Y_Y_hat <- rbind(t(Y_hat), Y); compare_Y_Y_hat = t(compare_Y_Y_hat)
    prediction.table <- table(Y_Hat = Y_hat, Y = Y)
    type.I.II.error.table <- plyr::count(Y_hat - Y)
    type.I.II.error.table$name <- ifelse(
      type.I.II.error.table[,1] == -1,
      "Type I Error",
      ifelse(
        type.I.II.error.table[,1] == 0,
        "True Pos+Neg",
        "Type II Error"
      )
    )
    train.accuracy = type.I.II.error.table[type.I.II.error.table$x == 0, 2] / sum(type.I.II.error.table$freq)

    # ROC
    actuals <- Y
    scores <- Y_hat
    roc_obj <- pROC::roc(response = actuals, predictor =  c(as.numeric(scores)))
    auc <- pROC::auc(roc_obj); auc

    # Output
    return(list(
      Weights.for.Variables = cbind(Name = colnames(X), Weights = as.numeric(weights.sol)),
      Weights.Lagrange.for.Variables = weights.lagrange,
      Y.and.Y.hat.Table.Original = compare_Y_Y_hat_original,
      Y.and.Y.hat.Table.Binary = compare_Y_Y_hat,
      I.II.Table = type.I.II.error.table,
      Prediction.Truth.Table = prediction.table,
      Accuracy = train.accuracy,
      AUC = auc,
      Gini.Co = 2*auc - 1
    ))
  } # End of function

  # Compile data
  all <- data.frame(cbind(y, x))

  # Split data:
  train <- all[1:(cutoff*nrow(all)),]; dim(train) # Training set
  test <- all[(cutoff*nrow(all)+1):nrow(all),]; dim(test) # Testing set

  # Identify Response and Explanatory:
  train.x <- train[,-1]; dim(train.x)
  train.y <- train[,1]; head(train.y)
  test.x <- test[,-1]; dim(test.x)
  test.y <- test[,1]; dim(data.frame(test.y))

  # Train:
  all = data.frame(cbind(train.y, train.x))
  Result <- search.for.weights(seed, all, cutoff.coefficient)
  Result$Weights.for.Variables;
  Train.List <- list(
    Weights.for.Variables = Result$Weights.for.Variables,
    Y.and.Y.hat.Table.Original = Result$Y.and.Y.hat.Table.Original,
    Y.and.Y.hat.Table.Binary = Result$Y.and.Y.hat.Table.Binary,
    I.II.Table = Result$I.II.Table,
    Prediction.Truth.Table = Result$Prediction.Truth.Table,
    Accuracy = Result$Accuracy,
    AUC = Result$AUC,
    Gini.Co = 2*Result$AUC - 1
  )

  # Test
  Y_test_hat <- matrix(0, nrow = nrow(test.x))
  for (i in 1:nrow(test.x)) {
    Y_test_hat[i, ] <- sum(test.x[i, ]*t(as.numeric(Train.List$Weights.for.Variables[, 2])))
  }
  compare_Y_Y_hat_original <- rbind(Y_test_hat = t(Y_test_hat), Truth = test.y);
  compare_Y_Y_hat_original = t(compare_Y_Y_hat_original)
  Y_test_hat = ifelse(Y_test_hat > cutoff.coefficient*mean(Y_test_hat), 1, 0)
  compare_Y_Y_hat <- rbind(t(Y_test_hat), test.y); compare_Y_Y_hat = t(compare_Y_Y_hat)
  prediction.table <- table(Y_Hat = Y_test_hat, Y = test.y)
  type.I.II.error.table <- plyr::count(Y_test_hat - test.y)
  test.accuracy = type.I.II.error.table[type.I.II.error.table$x == 0, 2] / sum(type.I.II.error.table$freq)

  # ROC
  roc_obj <- pROC::roc(response = test.y, predictor = c(as.numeric(Y_test_hat)))
  auc.plot = plot(roc_obj)
  auc <- pROC::auc(roc_obj); auc

  # Test Output
  Test.List <- list(
    Y.and.Y.hat.Table.Original = compare_Y_Y_hat_original,
    Y.and.Y.hat.Table.Binary = compare_Y_Y_hat,
    Prediction.Truth.Table = prediction.table,
    Prediction.Accuracy = test.accuracy,
    AUC = auc,
    Plot.AUC = auc.plot,
    Gini.Co = 2*auc - 1
  )

  # Final Output
  return(list(
    Train.Result = Train.List,
    Test.Result = Test.List
  ))
} # End of function
