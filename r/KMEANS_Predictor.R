#' @title Unsupervised Non-parametric Feature Selection using K-means Clustering
#' @description This function accepts input of data and parameters and produce output of new data.
#' @param symbol
#' @return NULL
#' @examples  KMEANS_Predictor()
#' @export KMEANS_Predictor
#'
#' # Define function
KMEANS_Predictor <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  k = 3,
  nstart = 20,
  plotGraph = TRUE
) {

  # Compile data
  all <- data.frame(cbind(y,x))

  # Split
  train <- all[1:round(cutoff*nrow(all),0),]; dim(train) # Training set
  test <- all[(round(cutoff*nrow(all),0)+1):nrow(all),]; dim(test) # Testing set

  # Fit KNN Model:
  k.model = kmeans(train[, -1], k, nstart = nstart)
  plt = factoextra::fviz_cluster(k.model, data = train[, -1])
  if (plotGraph == TRUE) {plt}
  y_train = train[, 1]
  y_hat_train = k.model$cluster

  # Prediction
  y_test = test[, 1]
  predict.kmeans <- function(object, newdata){
    centers <- object$centers
    n_centers <- nrow(centers)
    dist_mat <- as.matrix(dist(rbind(centers, newdata)))
    dist_mat <- dist_mat[-seq(n_centers), seq(n_centers)]
    max.col(-dist_mat) } # helper function for predict()
  y_hat_test = predict(k.model, test[,-1])

  # Errors
  mse_train = mean((y_train - y_hat_train)^2)
  mse_test = mean((y_test - y_hat_test)^2)

  # Return
  return(list(
    model = k.model,
    training_plot = plt,
    y_train = y_train,
    y_test = y_test,
    y_hat_train = y_hat_train,
    MSE = list(trainMSE = mse_train, testMSE = mse_test)
  ))
} # End of function
