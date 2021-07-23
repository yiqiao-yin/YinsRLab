#' @title Yin's Version of Convolutional Neural Network through Keras Framework
#' @description This package uses Keras as framework which provides a high-level neural networks API developed with a focus on enabling fast experimentation.
#' @param symbol
#' @return NULL
#' @examples
#' @export KerasC6NN3
#'
#' # Define function
KerasC6NN3 <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  img_rows = 8,
  img_cols = 8,
  batch_size = 128,
  convl1 = 8, convl2 = 6,
  convl3 = 6, convl4 = 4,
  convl5 = 4, convl6 = 4,
  convl1kernel = c(2,2),
  convl2kernel = c(2,2),
  maxpooll1 = c(2,2),
  activation = 'relu',
  finalActivation = 'softmax',
  l1.units = 256,
  l2.units = 128,
  l3.units = 64,
  epochs = 12
) {

  # Package
  library(keras); library(knitr)

  # Data Preparation -----------------------------------------------------
  all <- as.matrix(data.frame(cbind(y,x)))

  # The data, shuffled and split between train and test sets
  train_idx <- 1:round(cutoff*nrow(all),0)
  x_train <- all[train_idx, -1]
  y_train <- all[train_idx, 1]
  x_test <- all[-train_idx, -1]
  y_test <- all[-train_idx, 1]

  # Redefine  dimension of train/test inputs
  x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
  x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
  input_shape <- c(img_rows, img_cols, 1)
  #cat('x_train_shape:', dim(x_train), '\n')
  #cat(nrow(x_train), 'train samples\n')
  #cat(nrow(x_test), 'test samples\n')

  # Convert class vectors to binary class matrices
  num_classes <- nrow(plyr::count(all[,1]))
  y_train <- to_categorical(y_train, num_classes)
  y_test <- to_categorical(y_test, num_classes)

  # Define Model -----------------------------------------------------------
  model <- keras_model_sequential() %>%
    layer_conv_2d(filters = convl1, kernel_size = convl1kernel, activation = activation, input_shape = input_shape) %>%
    layer_conv_2d(filters = convl2, kernel_size = convl2kernel, activation = activation) %>%
    layer_conv_2d(filters = convl3, kernel_size = convl2kernel, activation = activation) %>%
    layer_conv_2d(filters = convl4, kernel_size = convl2kernel, activation = activation) %>%
    layer_conv_2d(filters = convl5, kernel_size = convl2kernel, activation = activation) %>%
    layer_conv_2d(filters = convl6, kernel_size = convl2kernel, activation = activation) %>%
    layer_max_pooling_2d(pool_size = maxpooll1) %>% layer_dropout(rate = 0.25) %>%
    layer_flatten() %>%
    layer_dense(units = l1.units, activation = activation) %>% layer_dropout(rate = 0.5) %>%
    layer_dense(units = l2.units, activation = activation) %>% layer_dropout(rate = 0.5) %>%
    layer_dense(units = l3.units, activation = activation) %>% layer_dropout(rate = 0.5) %>%
    layer_dense(units = num_classes, activation = finalActivation)
  # Compile model
  model %>% compile(
    loss = loss_categorical_crossentropy,
    optimizer = optimizer_adadelta(),
    metrics = c('accuracy')
  )
  # Train model
  history <- model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.2 )
  scores <- model %>% evaluate( x_test, y_test, verbose = 0 )

  # Generate predictions on new data:
  y_test_hat <- model %>% predict_classes(x_test)
  y_test <- as.matrix(all[-train_idx, 1]); y_test <- as.numeric(as.character(y_test))
  confusion.matrix <- table(Y_Hat = y_test_hat, Y = y_test)
  test.acc <- sum(diag(confusion.matrix))/sum(confusion.matrix)
  all.error <- plyr::count(as.numeric(as.character(y_test)) - cbind(y_test_hat))

  # AUC/ROC
  if ((num_classes == 2) && (nrow(plyr::count(y_test_hat)) > 1)) {
    AUC_test <- pROC::roc(y_test_hat, c(y_test))
  } else {
    AUC_test <- c("Estimate do not have enough levels.")
  }

  # Output metrics
  return(
    list(
      Model = list(model = model, scores = scores),
      Summary = c(
        paste0('x_train_shape:', dim(x_train), '\n'),
        paste0(nrow(x_train), 'train samples\n'),
        paste0(nrow(x_test), 'test samples\n'),
        paste0('Test loss:', scores[[1]], '\n'),
        paste0('Test accuracy:', scores[[2]], '\n') ),
      x_train = x_train,
      y_train = y_train,
      x_test = x_test,
      y_test = y_test,
      y_test_hat = y_test_hat,
      Training.Plot = plot(history),
      Confusion.Matrix = confusion.matrix,
      Confusion.Matrix.Pretty = knitr::kable(confusion.matrix),
      Testing.Accuracy = test.acc,
      All.Types.of.Error = all.error,
      Test_AUC = AUC_test
    )
  )
} # End of function
