#' @title Yin's Version of Recurrent Neural Network through Keras Framework
#' @description This package uses Keras as framework which provides a high-level neural networks API developed with a focus on enabling fast experimentation.
#' @param symbol
#' @return NULL
#' @examples
#' @export KerasRNN3
#'
#' # Define function
KerasRNN3 <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.5,
  batch_size = 32,
  loss = 'binary_crossentropy',
  optimizer = 'rmsprop',
  activation = 'softmax',
  max_features = 10,
  maxlen = 10,
  size = 32,
  lstm_layer1 = 64,
  nn_layer1 = 128,
  nn_layer2 = 64,
  nn_layer3 = 32,
  epochs = 10
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

  cat('Pad sequences (samples x time)\n')
  x_train <- pad_sequences(x_train, maxlen = maxlen)
  x_test <- pad_sequences(x_test, maxlen = maxlen)

  # Define Model ------------------------------------------------------------------------
  model <- keras_model_sequential()
  model %>%
    layer_embedding(input_dim = max_features, output_dim = size) %>%
    layer_lstm(units = lstm_layer1, dropout = 0.2, recurrent_dropout = 0.2) %>%
    layer_dense(units = nn_layer1, activation = activation) %>%
    layer_dense(units = nn_layer2, activation = activation) %>%
    layer_dense(units = nn_layer3, activation = activation) %>%
    layer_dense(units = 1, activation = activation)

  # Try using different optimizers and different optimizer configs
  model %>% compile(
    loss = loss,
    optimizer = optimizer,
    metrics = c('accuracy')
  )

  # Train
  model %>% fit(
    x_train, y_train,
    batrch_size = batch_size,
    epochs = epochs,
    validation_data = list(x_test, y_test)
  )

  # Scores
  scores <- model %>% evaluate(
    x_test, y_test,
    batch_size = batch_size)

  # Prediction
  predictions <- model %>% predict(x_test)
  head(predictions); which.max(predictions[1, ])
  y_hat <- apply(predictions, 1, which.max) - 1L; y_test_hat <- y_hat
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
