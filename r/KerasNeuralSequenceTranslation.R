#' @title Yin's Version of Neural Network through Keras Framework
#' @description This package uses Keras as framework which provides a high-level neural networks API developed with a focus on enabling fast experimentation.
#' @param symbol
#' @return NULL
#' @examples
#' @export KerasNeuralSequenceTranslation
#'
#' # Define function
KerasNeuralSequenceTranslation = function(
  X = X,
  y = y,
  cutoff = 0.8,
  validation_split = 1 - cutoff,
  max_len = 1,
  useModel = "lstm",
  num_hidden = 2,
  l1.units = 2,
  l2.units = 4,
  l3.units = 6,
  activation = 'tanh',
  loss = 'loss_mean_squared_error',
  useDice = TRUE,
  optimizer = optimizer_rmsprop(),
  batch_size = 128,
  epochs = 10,
  verbatim = TRUE
) {

  # Check shapes
  # this ensures that max_len can be divided by number of columns
  # in the explanatory data matrix
  print("First, check if max_len can be divided by number of cols.")
  if (ncol(X) %% max_len == 0) {
    print("... checking ...")
    print(ncol(X) %% max_len == 0)
    print("... it means it is divisible, pass and continue ...")
  } else {
    print("Warning: number of col in X cannot divide max_len!")
    print("Reset to 1.")
    max_len = 1
  } # Done

  # Package
  library(keras)

  # Separate scenarios:
  # if X is NULL, this is not allowed
  # if X is filled as a data frame (we only take data frame),
  # then we pursue X (this input data frame) as covariate matrix
  # and in this case the vector y (assuming it has the same length as X)
  # will be response variable
  # training data
  train_idx = 1:round(cutoff*nrow(y),0)
  x_train <- array(X[train_idx,], dim = c(length(train_idx), max_len, round(ncol(X)/max_len)))
  y_train <- array(y[train_idx,], dim = c(length(train_idx), ncol(y)))

  # testing data
  x_test <- array(X[-train_idx,], dim = c(nrow(y) - length(train_idx), max_len, round(ncol(X)/max_len)))
  y_test <- array(y[-train_idx,], dim = c(nrow(y) - length(train_idx), ncol(y)))

  # shape
  dim(x_train); dim(x_test); dim(y_train); dim(y_test)

  # Defining the Model
  if (tolower(useModel) == "lstm") {
    if (num_hidden == 1) {
      model <- keras_model_sequential() %>%
        layer_lstm(l1.units, input_shape = c(max_len, round(ncol(X)/max_len))) %>%
        layer_dense(ncol(y)) %>% layer_activation(activation)
      summary(model)
    } else if (num_hidden == 2) {
      model <- keras_model_sequential() %>%
        layer_lstm(l1.units, input_shape = c(max_len, round(ncol(X)/max_len))) %>%
        layer_dense(units = l2.units, activation = activation, use_bias = FALSE) %>%
        layer_dense(ncol(y)) %>% layer_activation(activation)
      summary(model)
    } else if (num_hidden == 3) {
      model <- keras_model_sequential() %>%
        layer_lstm(l1.units, input_shape = c(max_len, round(ncol(X)/max_len))) %>%
        layer_dense(units = l2.units, activation = activation, use_bias = FALSE) %>%
        layer_dense(units = l3.units, activation = activation, use_bias = FALSE) %>%
        layer_dense(ncol(y)) %>% layer_activation(activation)
      summary(model)
    } else {
      print("Too many layers implemented, set to default: one hidden layer")
      model <- keras_model_sequential() %>%
        layer_lstm(l1.units, input_shape = c(max_len, round(ncol(X)/max_len))) %>%
        layer_dense(ncol(y)) %>% layer_activation(activation)
      summary(model)
    }
  } else if (tolower(useModel) == "gru") {
    if (num_hidden == 1) {
      model <- keras_model_sequential() %>%
        layer_gru(units = l1.units, return_sequences = TRUE, input_shape = dim(x_train)[-1]) %>%
        bidirectional(layer_gru(units = l1.units)) %>%
        layer_dense(units = ncol(y_test), activation = activation)
      summary(model)
    } else if (num_hidden == 2) {
      model <- keras_model_sequential() %>%
        layer_gru(units = l1.units, return_sequences = TRUE,input_shape = dim(x_train)[-1]) %>%
        bidirectional(layer_gru(units = l1.units)) %>%
        layer_dense(units = l2.units, activation = activation) %>%
        layer_dense(units = ncol(y_test), activation = activation)
      summary(model)
    } else if (num_hidden == 3) {
      model <- keras_model_sequential() %>%
        layer_gru(units = l1.units, return_sequences = TRUE,input_shape = dim(x_train)[-1]) %>%
        bidirectional(layer_gru(units = l1.units)) %>%
        layer_dense(units = l2.units, activation = activation) %>%
        layer_dense(units = l3.units, activation = activation) %>%
        layer_dense(units = ncol(y_test), activation = activation)
      summary(model)
    }
  }

  # Dice Loss
  # Wiki: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
  dice <- custom_metric("dice", function(y_true, y_pred, smooth = 1.0) {
    y_true_f <- k_flatten(y_true)
    y_pred_f <- k_flatten(y_pred)
    intersection <- k_sum(y_true_f * y_pred_f)
    (2 * intersection + smooth) / (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
  })

  # Compile and Train
  # compile the model with appropriate loss function, optimizer, and metrics:
  if (useDice) {
    model %>% compile(
      loss = loss,
      optimizer = optimizer,
      metrics = dice )
  } else {
    model %>% compile(
      loss = loss,
      optimizer = optimizer )
  } # done
  history = model %>% fit(
    x_train, y_train,
    batch_size = batch_size,
    validation_split = validation_split,
    epochs = epochs ); if (verbatim) {plot(history)}

  # Prediction
  yhat_train_mat = predict(model, x_train)
  yhat_test_mat = predict(model, x_test)
  if (verbatim) {
    print("Note:")
    print("-- Use yhat_test_mat = predict(model, x_test) to make prediction")
    print(paste0("-- This assumes x_test has dimension: ", paste0(dim(x_test)[-1], collapse = " x "))) }
  head(yhat_test_mat)
  if (verbatim) {
    par(mfrow=c(1,2))
    matplot(
      y_test,
      type = 'l',
      xaxs = "i",
      yaxs = "i" ,
      lwd = 3,
      xlab = "Sequential Index (Time or Day)",
      ylab = "Real Values",
      main = paste0("Real Data: Y Matrix"))
    eachLoss = sapply(1:ncol(yhat_test_mat), function(s) mean(abs(y_test[,s] - yhat_test_mat[,s]), na.rm = TRUE))
    matplot(
      yhat_test_mat,
      type = 'l',
      xaxs = "i",
      yaxs = "i" ,
      lwd = 3,
      xlab = "Sequential Index (Time or Day)",
      ylab = "Predicted Values",
      main = paste0("Prediction: YHat Matrix \n(average of the MAEs for all paths: ", mean(round(eachLoss,3), na.rm = TRUE), ")"))
  } # Done

  # Return
  return(
    list(
      Model = list(model = model, weights = model$weights),
      X = X, # original explanatory data matrix
      y = y, # original response data matrix
      x_train = x_train,
      y_train = y_train,
      x_test = x_test,
      y_test = y_test,
      yhat_train_mat = yhat_train_mat,
      yhat_test_mat = yhat_test_mat
    )
  )
}
