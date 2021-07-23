#' @title Unsupervised Non-parametric Feature Selection using Principle Component
#' @description This function accepts input of data and parameteres and produce output of new data.
#' @param symbol
#' @return NULL
#' @examples  PC_VS()
#' @export PC_VS
#'
#' # Define function
PC_VS <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  select = 1:ncol(all[,-1])
) {
  # Compile data
  all <- data.frame(cbind(y,x))

  # Split data
  train <- all[1:round(cutoff*nrow(all),0),]; dim(train) # Training set
  test <- all[(round(cutoff*nrow(all),0)+1):nrow(all),]; dim(test) # Testing set
  train_x_pc_total <- princomp(train[, -1])
  train_x_pc <- princomp(train[, -1])$scores
  test_x_pc_total <- princomp(test[, -1])
  test_x_pc <- princomp(test[, -1])$scores
  X <- rbind(train_x_pc, test_x_pc); Y <- cbind(all$y)
  all <- data.frame(cbind(Y, X))[, c(select)]
  colnames(all)[1] <- "y"

  # Return
  return(list(
    all = data.frame(rbind(train, test)),
    train = train,
    test = test,
    train.x = train_x_pc,
    train.x.pc.loading = train_x_pc_total$loadings,
    train.y = train[, 1],
    test.x = test_x_pc,
    test.x.pc.loading = test_x_pc_total$loadings,
    test.y = test[, 1]
  ))
} # End of function
