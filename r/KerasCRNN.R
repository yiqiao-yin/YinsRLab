#' @title Yin's Version of Recurrent Neural Network through Keras Framework
#' @description This package uses Keras as framework which provides a high-level neural networks API developed with a focus on enabling fast experimentation.
#' @param symbol
#' @return NULL
#' @examples
#' @export KerasCRNN
#'
#' # Define function
KerasCRNN <- function(
  # Data
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,

  # Embedding
  max_features = 10000,
  maxlen = ncol(all[, -1]),
  embedding_size = 128,

  # Convolution
  kernel_size = 5,
  filters = 64,
  pool_size = 4,

  # LSTM
  lstm_output_size = 70,

  # Training
  batch_size = 30,
  epochs = 2,

  # Model
  activation = "relu",
  finalActivation = "sigmoid",
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy",

  # Comment
  verbose = 1
) {
  # Library
  library(keras)

  # Compile data
  all <- data.frame(cbind(y,x))
  num_classes <- nrow(plyr::count(all[,1])); num_classes

  # The data, shuffled and split between train and test sets
  all <- as.matrix(all)
  index <- 1:(cutoff*nrow(all))
  x_train <- all[index, -1]
  y_train <- all[index, 1]
  x_test <- all[-index, -1]
  y_test <- all[-index, 1]; y_test_backup <- y_test
  dim(x_train); dim(y_train); dim(x_test); dim(y_test)

  # Model
  model <- keras_model_sequential()
  model %>%
    layer_embedding(max_features, embedding_size, input_length = maxlen) %>%
    layer_dropout(0.25) %>%
    layer_conv_1d(
      filters,
      kernel_size,
      padding = "valid",
      activation = activation,
      strides = 1
    ) %>%
    layer_max_pooling_1d(pool_size) %>%
    layer_lstm(lstm_output_size) %>%
    layer_dense(1) %>%
    layer_activation(finalActivation)

  model %>% compile(
    loss = loss,
    optimizer = optimizer,
    metrics = metrics
  )

  # Create a validation set
  x_val <- x_test
  partial_x_train <- x_train

  y_val <- y_test
  partial_y_train <- y_train

  history <- model %>% fit(
    partial_x_train,
    partial_y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = list(x_val, y_val),
    verbose = verbose
  )

  #### Evaluate the model ####
  scores <- model %>% evaluate(x_test, y_test)
  scores

  # Create a graph of accuracy and loss over time
  Plt = plot(history)

  # Prediction
  predictions <- model %>% predict(x_test)
  y_hat <- as.numeric(predictions > mean(predictions)); y_test_hat <- y_hat
  y_test <- y_test_backup; y_test <- as.numeric(as.character(y_test))
  confusion.matrix <- table(Y_Hat = y_hat, Y = y_test); confusion.matrix
  test.acc <- sum(diag(confusion.matrix))/sum(confusion.matrix)
  all.error <- plyr::count(cbind(y_test) - cbind(y_hat))

  # AUC/ROC
  if ((num_classes == 2) && (nrow(plyr::count(y_test_hat)) > 1)) {
    AUC_test <- pROC::roc(y_test_hat, c(y_test))
  } else {
    AUC_test <- c("Estimate do not have enough levels.")
  }

  # Comment
  return(list(
    Model = list(model = model, scores = scores),
    Y_Hat = y_hat,
    Y_Hat_Test = y_test,
    Confusion_Matrix = confusion.matrix,
    Test_Accuracy = test.acc,
    All_Error_Types = all.error,
    Test_AUC = AUC_test
  ))
} # End of function
