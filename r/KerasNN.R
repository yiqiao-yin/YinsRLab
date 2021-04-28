#' @title Yin's Version of Neural Network through Keras Framework
#' @description This package uses Keras as framework which provides a high-level neural networks API developed with a focus on enabling fast experimentation.
#' @param symbol
#' @return NULL
#' @examples
#' @export KerasNN
#'
#' # Define function
KerasNN <- function(
  x = x,
  y = y,
  cutoff = .9,
  validation_split = 1 - cutoff,
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  batch_size = 128,
  numberOfLayers = 1,
  activation = 'relu',
  useBias = FALSE,
  finalactivation = 'softmax',
  l1.units = 128,
  l2.units = 64,
  epochs = 30
) {

  # Package
  library(keras)

  # Data
  all <- data.frame(cbind(y, x))

  # Setup
  train_idx <- 1:round(cutoff*nrow(all),0)
  x_train <- as.matrix(all[train_idx, -1])
  y_train <- as.matrix(all[train_idx, 1])
  x_test <- as.matrix(all[-train_idx, -1])
  y_test <- as.matrix(all[-train_idx, 1])

  # Check levels for response
  number.of.levels <- nrow(plyr::count(y_train))
  num_classes <- number.of.levels

  # To prepare this data for training we one-hot encode the
  # vectors into binary class matrices using the Keras to_categorical() function
  y_train <- to_categorical(y_train, number.of.levels)
  y_test <- to_categorical(y_test, number.of.levels)

  # Defining the Model
  if (numberOfLayers == 1) {
    model <- keras_model_sequential()
    model %>%
      layer_dense(
        units = number.of.levels,
        input_shape = c(ncol(x_train)),
        activation = finalactivation,
        use_bias = useBias)
    summary(model)
  } else if (numberOfLayers == 2) {
    model <- keras_model_sequential()
    model %>%
      layer_dense(units = l1.units, activation = activation, use_bias = useBias, input_shape = c(ncol(x_train))) %>%
      layer_dense(units = number.of.levels, use_bias = useBias, activation = finalactivation)
    summary(model)
  } else if (numberOfLayers == 3) {
    model <- keras_model_sequential()
    model %>%
      layer_dense(units = l1.units, activation = activation, use_bias = useBias, input_shape = c(ncol(x_train))) %>%
      layer_dense(units = l2.units, activation = activation, use_bias = useBias) %>%
      layer_dense(units = number.of.levels, use_bias = useBias, activation = finalactivation)
    summary(model)
  } else {
    print("============== WARNING ==============")
    print("Input value for [numberOfLayers] must be 1, 2, or 3.")
    print("Since none of the values above are entered, the default is set to 1.")
    print("=====================================")
    model <- keras_model_sequential()
    model %>%
      layer_dense(
        units = number.of.levels,
        input_shape = c(ncol(x_train)),
        activation = finalactivation,
        use_bias = useBias)
    summary(model)
  }

  # Next, compile the model with appropriate loss function, optimizer, and metrics:
  model %>% compile(
    loss = loss,
    optimizer = optimizer,
    metrics = c('accuracy') )

  # Training and Evaluation
  history <- model %>% fit(
    x_train, y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = validation_split
  ); plot(history)

  # Evaluate the model's performance on the test data:
  scores = model %>% evaluate(x_test, y_test)

  # Generate predictions on new data:
  y_test_hat <- model %>% predict_classes(x_test)
  y_test_hat_raw <- model %>% predict_proba(x_test); colnames(y_test_hat_raw) = c(0:(num_classes-1))
  y_test <- as.matrix(all[-train_idx, 1])
  y_test <- as.numeric(as.character(y_test))
  confusion.matrix <- table(Y_Hat = y_test_hat, Y = y_test)
  test.acc <- sum(diag(confusion.matrix))/sum(confusion.matrix)
  all.error <- plyr::count(y_test - cbind(y_test_hat))
  y_test_eval_matrix <- cbind(
    y_test=y_test,
    y_test_hat=y_test_hat,
    y_test_hat_raw=y_test_hat_raw )

  # AUC/ROC
  if ((num_classes == 2) && (nrow(plyr::count(y_test_hat)) == 2)) {
    y_test_for_roc = c(y_test)
    y_test_class1_for_roc = c(y_test_hat_raw[, 2])
    AUC_test <- pROC::roc(response = y_test_for_roc, predictor = y_test_class1_for_roc)
  } else {
    print("Estimate do not have enough levels.")
    AUC_test <- 0.5
  }

  # Return
  return(
    list(
      Model = list(model = model, scores = scores),
      x_train = x_train,
      y_train = y_train,
      x_test = x_test,
      y_test = y_test,
      y_test_hat = y_test_hat,
      y_test_eval_matrix = y_test_eval_matrix,
      Training.Plot = plot(history),
      Confusion.Matrix = list(regularTable = confusion.matrix, prettyTable = knitr::kable(confusion.matrix)),
      Testing.Accuracy = test.acc,
      All.Types.of.Error = all.error,
      Test_AUC = AUC_test
    )
  )
} # End of function
