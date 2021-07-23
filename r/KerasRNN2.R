#' @title Yin's Version of Recurrent Neural Network through Keras Framework
#' @description This package uses Keras as framework which provides a high-level neural networks API developed with a focus on enabling fast experimentation.
#' @param symbol
#' @return NULL
#' @examples
#' @export KerasRNN3
#'
#' # Define function
KerasRNN2 <- function(
  x = all[, -1],
  y = all[, 1],
  cutoff = 0.9,
  vocab_size = 10000,
  output_dim = 16,
  l1.units = 16,
  l2.units = 1,
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy'),
  epochs = 40,
  batch_size = 512,
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

  #### Build the model ####

  # The neural network is created by stacking layers - this requires two main architectural decisions:
  # How many layers to use in the model?
  # How many hidden units to use for each layer?

  # In this example, the input data consists of an array of
  # word-indices. The labels to predict are either 0 or 1. Let's build a model for this problem:

  # input shape is the vocabulary count used for the movie reviews (10,000 words)
  vocab_size <- vocab_size

  model <- keras_model_sequential()
  model %>%
    layer_embedding(input_dim = vocab_size, output_dim = output_dim) %>%
    layer_global_average_pooling_1d() %>%
    layer_dense(units = l1.units, activation = "relu") %>%
    layer_dense(units = l2.units, activation = "sigmoid")

  model %>% summary()

  # Loss function and optimizer
  model %>% compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metrics
  )

  # Create a validation set
  x_val <- x_test
  partial_x_train <- x_train

  y_val <- y_test
  partial_y_train <- y_train

  #### Train the model ####
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
