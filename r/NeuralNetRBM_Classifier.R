#' @title Supervised Classification ML Algorithm using RBM
#' @description This function accepts input of data and parameteres and produce output of classification results.
#' @param symbol
#' @return NULL
#' @examples  NeuralNetRBM_Classifier()
#' @export NeuralNetRBM_Classifier
#'
#' # Define function
NeuralNetRBM_Classifier <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = cutoff,
  linear.output = TRUE,
  startweights = NULL,
  algorithm = "rprop+", # c("rprop+", "rprop-", "slr", "sag", "backprop")
  act.fct.name = "softplus", # c("logistic", "tanh")
  err.fct = "sse",
  hidden = c(10),
  stepmax = 1e6,
  threshold = 1,
  verbose = TRUE
) {

  # Data
  all = data.frame(cbind(y, x))

  # Setup
  train_idx = 1:round(cutoff*nrow(all),0)
  x_train = as.matrix(all[train_idx, -1]); y_train = as.matrix(all[train_idx, 1])
  train = data.frame(cbind(y_train, x_train)); colnames(train) = colnames(all)
  x_test = as.matrix(all[-train_idx, -1]); y_test = as.matrix(all[-train_idx, 1])
  test = data.frame(cbind(y_test, x_test)); colnames(test) = colnames(all)

  # Activation Function:
  if (act.fct.name == "softplus") {
    act.fct = function(x) {log(1 + exp(x))} }
  if (act.fct.name == "sigmoid") {
    act.fct = function(x) {exp(x)/(1+exp(x))} }

  # Deep Learning: Training
  nn = neuralnet::neuralnet(
    y~., train,
    linear.output = linear.output,
    hidden = hidden,
    startweights = startweights,
    act.fct = act.fct,
    algorithm = algorithm,
    stepmax = stepmax,
    threshold = threshold,
    err.fct = err.fct)
  yhat_train = predict(nn, x_train)
  AUC_train = pROC::roc(response=y_train, predictor=yhat_train)

  # Deep Learning: Prediction
  yhat_test = predict(nn, x_test)
  AUC_test = pROC::roc(response=y_test, predictor=yhat_test)

  # Visualiation
  if (verbose) {
    print("Print training set AUC:")
    plot(AUC_train)
    print("Print testing set AUC:")
    plot(AUC_test)    #plot(yhat_train)
    print("Trained neural network architecture:")
    plot(nn)
  }

  # Confusion Matrix
  trainingConfusionMatrix = table(yhat_train = as.numeric(yhat_train > mean(yhat_train)), y = y_train)
  testingConfusionMatrix = table(yhat_test = as.numeric(yhat_test > mean(yhat_test)), y = y_test)

  # Output
  return(list(
    model = nn,
    model_brief = nn$result.matrix,
    trainAUC = AUC_train$auc,
    trainingConfusionMatrix = trainingConfusionMatrix,
    trainingAccuracy = sum(diag(trainingConfusionMatrix))/sum(trainingConfusionMatrix),
    testAUC = AUC_test$auc,
    testingConfusionMatrix = testingConfusionMatrix,
    testingAccuracy = sum(diag(testingConfusionMatrix))/sum(testingConfusionMatrix),
    yhat = list(yhat_train, yhat_test)
  ))
} # End function
