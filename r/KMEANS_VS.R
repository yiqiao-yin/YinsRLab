#' @title Unsupervised Non-parametric Feature Selection using K-means Clustering
#' @description This function accepts input of data and parameters and produce output of new data.
#' @param symbol
#' @return NULL
#' @examples  KMEANS_VS()
#' @export KMEANS_VS
#'
#' # Define function
KMEANS_VS <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  k = 3
) {
  # Compile data
  all <- data.frame(cbind(y,x))

  # Split
  train <- all[1:round(cutoff*nrow(all),0),]; dim(train) # Training set
  test <- all[(round(cutoff*nrow(all),0)+1):nrow(all),]; dim(test) # Testing set
  train_x_pc <- data.frame(t(kmeans(t(train[, -1]), k)$centers)); dim(train_x_pc)
  test_x_pc <- data.frame(t(kmeans(t(test[, -1]), k)$centers)); dim(test_x_pc)
  X <- rbind(train_x_pc, test_x_pc); Y <- cbind(all$y)
  all <- data.frame(cbind(Y, X))
  colnames(all)[1] <- "y"

  # Return
  return(list(
    all = data.frame(rbind(train, test)),
    train = train,
    test = test,
    train.x = train[, -1],
    train.x.new = train_x_pc,
    train.y = train[, 1],
    test.x = test[, -1],
    test.x.new = test_x_pc,
    test.y = test[, 1]
  ))
} # End of function
