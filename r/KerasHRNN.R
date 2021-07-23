#' @title Yin's Version of Hierarchical Neural Network through Keras Framework
#' @description This package uses Keras as framework which provides a high-level neural networks API developed with a focus on enabling fast experimentation.
#' @param symbol
#' @return NULL
#' @examples
#' @export KerasHRNN
#'
#' # Define function
KerasHRNN <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.5,
  batch_size = 32,
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  activation = 'softmax',
  img_row = 9,
  img_col = 9,
  row_hidden = 128,
  col_hidden = 128,
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

  # Reshapes data to 4D for Hierarchical RNN.
  x_train <- array_reshape(x_train, c(nrow(x_train), img_row, img_col, 1))
  x_test <- array_reshape(x_test, c(nrow(x_test), img_row, img_col, 1))
  dim(x_train); dim(y_train); dim(x_test); dim(y_test)

  # Check
  dim_x_train <- dim(x_train)
  cat('x_train_shape: ', dim_x_train)
  cat('train samples: ', nrow(x_train))
  cat('test samples: ', nrow(x_test))

  # Converts class vectors to binary class matrices
  y_train <- to_categorical(y_train, num_classes)
  y_test <- to_categorical(y_test, num_classes)

  # Define input dimensions
  row <- dim_x_train[[2]]
  col <- dim_x_train[[3]]
  pixel <- dim_x_train[[4]]

  # Model input (4D)
  input <- layer_input(shape = c(row, col, pixel))

  # Encodes a row of pixels using TimeDistributed Wrapper
  encoded_rows <- input %>% time_distributed(layer_lstm(units = row_hidden))

  # Encodes columns of encoded rows
  encoded_columns <- encoded_rows %>% layer_lstm(units = col_hidden)

  # Model output
  prediction <- encoded_columns %>%
    layer_dense(units = num_classes, activation = activation)

  # Define Model ------------------------------------------------------------------------

  model <- keras_model(input, prediction)
  model %>% compile(
    loss = loss,
    optimizer = optimizer,
    metrics = c('accuracy')
  )

  # Training
  model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 1,
    validation_data = list(x_test, y_test)
  )

  # Evaluation
  scores <- model %>% evaluate(x_test, y_test, verbose = 1)
  cat('Test loss:', scores[[1]], '\n')
  cat('Test accuracy:', scores[[2]], '\n')

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
    Data = all,
    test_Y_Hat = y_hat,
    test_Y = y_test,
    Confusion_Matrix = confusion.matrix,
    Test_Accuracy = test.acc,
    All_Error_Types = all.error,
    Test_AUC = AUC_test
  ))
} # End of function
