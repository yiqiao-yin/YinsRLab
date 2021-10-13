#' @title Yin's Version of KerasVGG16 through Keras Framework
#' @description This package uses Keras as framework which provides a high-level neural networks API developed with a focus on enabling fast experimentation.
#' @param symbol
#' @return NULL
#' @examples
#' @export KerasVGG16
#'
#' # Define function
KerasVGG16 <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  img_rows = 48,
  img_cols = 48,
  depth = 1,
  dropout = 1e-5,
  alpha = 1,
  batch_size = 128,
  lr = 0.1,
  momentum = 0.9,
  epochs = 12
) {

  # Package
  library(keras); library(densenet); library(knitr)

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

  # Model Definition -------------------------------------------------------
  input_img <- layer_input(shape = input_shape)
  model <- application_vgg16(
    include_top = TRUE, weights = NULL,
    input_tensor = input_img, input_shape = input_shape, pooling = TRUE,
    classes = num_classes)
  opt <- optimizer_sgd(lr = lr, momentum = momentum, nesterov = TRUE)
  model %>% compile(
    optimizer = opt,
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )

  # Model fitting -----------------------------------------------------------
  # callbacks for weights and learning rate
  lr_schedule <- function(epoch, lr) { if(epoch <= 150) { 0.1 } else if(epoch > 150 && epoch <= 225){ 0.01 } else { 0.001 } }
  lr_reducer <- callback_learning_rate_scheduler(lr_schedule)
  history <- model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(x_test, y_test),
    callbacks = list( lr_reducer ) )
  scores <- model %>% evaluate(x_test, y_test, verbose = 0)

  # Generate predictions on new data:
  y_test_hat_scores <- cbind(apply(predict(model, x_test), 1, max))
  y_test_hat_mean <- mean(y_test_hat_scores)
  y_test_hat <- ifelse(y_test_hat_scores > y_test_hat_mean, 1, 0)
  y_test <- as.matrix(all[-train_idx, 1]); y_test <- as.numeric(as.character(y_test))
  confusion.matrix <- table(Y_Hat = y_test_hat, Y = y_test)
  test.acc <- sum(diag(confusion.matrix))/sum(confusion.matrix)
  all.error <- plyr::count(as.numeric(as.character(y_test)) - cbind(y_test_hat))

  # AUC/ROC
  if ((num_classes == 2) && (nrow(plyr::count(y_test_hat_scores)) > 1)) {
    AUC_test <- pROC::roc(y_test_hat_scores, c(y_test))
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
      history = history,
      Training.Plot = plot(history),
      Confusion.Matrix = confusion.matrix,
      Confusion.Matrix.Pretty = knitr::kable(confusion.matrix),
      Testing.Accuracy = test.acc,
      All.Types.of.Error = all.error,
      Test_AUC = AUC_test
    )
  )
} # End of function
