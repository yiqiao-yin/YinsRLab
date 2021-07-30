#' @title Yin's Version of Neural Network through Keras Framework
#' @description This package uses Keras as framework which provides a high-level neural networks API developed with a focus on enabling fast experimentation.
#' @param symbol
#' @return NULL
#' @examples
#' @export KerasRNN
#'
#' # Define function
KerasRNN <- function(
  X = NULL,
  y = y,
  cutoff = 0.8,
  validation_split = 1 - cutoff,
  max_len = 6,
  useModel = "lstm",
  num_hidden = 2,
  l1.units = 2,
  l2.units = 4,
  l3.units = 6,
  activation = 'tanh',
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  batch_size = 128,
  epochs = 10
) {

  # Package
  library(keras)

  # Separate scenarios:
  # if X is NULL, we pursue to consider the input vector y
  # as a long time-series data (one big vector), the code
  # will then create a autoregressor with max_len (another input)
  # to be the parameter of AR(max_len), but the algorithm
  # will follow LSTM or Bidirectional RNN
  # if X is filled as a data frame (we only take data frame),
  # then we pursue X (this input data frame) as covariate matrix
  # and in this case the vector y (assuming it has the same length as X)
  # will be response variable
  if (is.null(X)) {
    # CLEAN DATA:
    # get a list of start indexes for our (overlapping) chunks
    start_indexes <- seq(1, length(y) - (max_len + 1), by = 1)

    # create an empty matrix to store our data in
    data_matrix <- matrix(nrow = length(start_indexes), ncol = max_len + 1)

    # fill our matrix with the overlapping slices of our dataset
    for (i in 1:length(start_indexes)){
      data_matrix[i,] <- y[start_indexes[i]:(start_indexes[i] + max_len)]
    }

    # split our data into the day we're predict (y), and the
    # sequence of days leading up to it (X)
    X <- data_matrix[,-ncol(data_matrix)]
    y <- data_matrix[,ncol(data_matrix)]

    # training data
    train_idx = 1:round(cutoff*length(y),0)
    x_train <- array(X[train_idx,], dim = c(length(train_idx), max_len, 1))
    y_train <- y[train_idx]

    # testing data
    x_test <- array(X[-train_idx,], dim = c(length(y) - length(train_idx), max_len, 1))
    y_test <- y[-train_idx]
  } else {
    # training data
    train_idx = 1:round(cutoff*length(y),0)
    x_train <- array(X[train_idx,], dim = c(length(train_idx), ncol(X), 1))
    y_train <- y[train_idx]

    # testing data
    x_test <- array(X[-train_idx,], dim = c(length(y) - length(train_idx), ncol(X), 1))
    y_test <- y[-train_idx]
  } # Done

  # Defining the Model
  if (tolower(useModel) == "lstm") {
    if (num_hidden == 1) {
      model <- keras_model_sequential() %>%
        layer_lstm(units = l1.units, input_shape = dim(x_train)[-1], return_sequences = FALSE, stateful = FALSE, use_bias = FALSE) %>%
        layer_dense(1, activation = activation, use_bias = FALSE)
      summary(model)
    } else if (num_hidden == 2) {
      model <- keras_model_sequential() %>%
        layer_lstm(units = l1.units, input_shape = dim(x_train)[-1], return_sequences = FALSE, stateful = FALSE, use_bias = FALSE) %>%
        layer_dense(units = l2.units, activation = activation, use_bias = FALSE) %>%
        layer_dense(units = 1, activation = activation, use_bias = FALSE)
      summary(model)
    } else if (num_hidden == 3) {
      model <- keras_model_sequential() %>%
        layer_lstm(units = l1.units, input_shape = dim(x_train)[-1], return_sequences = FALSE, stateful = FALSE, use_bias = FALSE) %>%
        layer_dense(units = l2.units, activation = activation, use_bias = FALSE) %>%
        layer_dense(units = l3.units, activation = activation, use_bias = FALSE) %>%
        layer_dense(units = 1, activation = activation, use_bias = FALSE)
      summary(model)
    } else {
      print("Too many layers implemented, set to default: one hidden layer")
      model <- keras_model_sequential() %>%
        layer_lstm(units = l1.units, input_shape = dim(x_train)[-1], return_sequences = FALSE, stateful = FALSE, use_bias = FALSE) %>%
        layer_dense(units = 1, activation = activation, use_bias = FALSE)
      summary(model)
    }
  } else if (tolower(useModel) == "gru") {
    if (num_hidden == 1) {
      model <- keras_model_sequential() %>%
        layer_gru(units = l1.units, return_sequences = TRUE,input_shape = dim(x_train)[-1]) %>%
        bidirectional(layer_gru(units = l1.units)) %>%
        layer_dense(units = 1, activation = activation)
      summary(model)
    } else if (num_hidden == 2) {
      model <- keras_model_sequential() %>%
        layer_gru(units = l1.units, return_sequences = TRUE,input_shape = dim(x_train)[-1]) %>%
        bidirectional(layer_gru(units = l1.units)) %>%
        layer_dense(units = l2.units, activation = activation) %>%
        layer_dense(units = 1, activation = activation)
      summary(model)
    } else if (num_hidden == 3) {
      model <- keras_model_sequential() %>%
        layer_gru(units = l1.units, return_sequences = TRUE,input_shape = dim(x_train)[-1]) %>%
        bidirectional(layer_gru(units = l1.units)) %>%
        layer_dense(units = l2.units, activation = activation) %>%
        layer_dense(units = l3.units, activation = activation) %>%
        layer_dense(units = 1, activation = activation)
      summary(model)
    }
  }

  # Next, compile the model with appropriate loss function, optimizer, and metrics:
  if (nrow(plyr::count(y_test)) == 2) {
    num_classes = nrow(plyr::count(y_test))

    model %>% compile(
      loss = loss,
      optimizer = optimizer,
      metrics = c('accuracy') )

    # Training and Evaluation
    history <- model %>% fit(
      x = x_train, # sequence we're using for prediction
      y = y_train, # sequence we're predicting
      epochs = epochs,
      batch_size = batch_size,
      validation_split = validation_split
    ); plot(history)

    # Evaluate the model's performance on the test data:
    scores = model %>% evaluate(x_test, y_test)

    # Generate predictions on new data:
    y_test_hat <- model %>% predict_classes(x_test)
    y_train_hat_raw <- model %>% predict_proba(x_train)
    y_test_hat_raw <- model %>% predict_proba(x_test)
    confusion.matrix <- table(Y_Hat = y_test_hat, Y = y_test)
    test.acc <- sum(diag(confusion.matrix))/sum(confusion.matrix)
    all.error <- plyr::count(y_test - cbind(y_test_hat))
    y_test_eval_matrix <- cbind(
      y_test=y_test,
      y_test_hat=y_test_hat,
      y_test_hat_raw=y_test_hat_raw )
    colnames(y_test_eval_matrix) = c("y_test", "y_test_hat_class", "y_test_hat_prob")

    p = NA
  } else {
    num_classes = nrow(plyr::count(y_test))

    model %>% compile(
      loss = loss,
      optimizer = optimizer,
      metrics = c('mae') )

    # Training and Evaluation
    history <- model %>% fit(
      x = x_train, # sequence we're using for prediction
      y = y_train, # sequence we're predicting
      epochs = epochs,
      batch_size = batch_size,
      validation_split = validation_split
    ); plot(history)

    # Evaluate the model's performance on the test data:
    scores = model %>% evaluate(x_test, y_test)

    # Generate predictions on new data:
    y_train_hat_raw <- model %>% predict_proba(x_train)
    y_test_hat_raw <- model %>% predict_proba(x_test)
    y_test_hat = y_test_hat_raw
    y_test_eval_matrix <- data.frame(
      y_test = y_test,
      y_test_hat_raw = y_test_hat_raw )
    colnames(y_test_eval_matrix) = c("y_test", "y_test_hat")
    confusion.matrix = NA
    test.acc = NA
    all.error = NA

    library(ggplot2)
    y_test_eval_matrix_for_gg = y_test_eval_matrix
    y_test_eval_matrix_for_gg$dates = 1:nrow(y_test_eval_matrix)
    y_test_eval_matrix_for_gg = reshape2::melt(y_test_eval_matrix_for_gg, id.var='dates')
    p = ggplot(data = y_test_eval_matrix_for_gg,
           aes(x = dates, y = value, col = variable)) +
      geom_line() +
      theme(legend.title = element_blank())
  }

  # AUC/ROC
  if ((num_classes == 2) && (nrow(plyr::count(y_test_hat)) == 2)) {
    y_test_for_roc = c(y_test)
    y_test_class1_for_roc = c(y_test_hat_raw)
    AUC_test <- pROC::roc(response = y_test_for_roc, predictor = y_test_class1_for_roc)
  } else {
    print("Estimate do not have enough levels.")
    AUC_test = list()
    AUC_test$auc <- 0.5
  }

  # Return
  return(
    list(
      Model = list(model = model, scores = scores),
      x_train = x_train,
      y_train = y_train,
      x_test = x_test,
      y_test = y_test,
      y_train_hat = y_train_hat_raw,
      y_test_hat = y_test_hat,
      y_test_eval_matrix = y_test_eval_matrix,
      Training.Plot = plot(history),
      Testing.Plot = p,
      Confusion.Matrix = list(regularTable = confusion.matrix, prettyTable = knitr::kable(confusion.matrix)),
      Testing.Accuracy = test.acc,
      All.Types.of.Error = all.error,
      Test_AUC = AUC_test
    )
  )
} # End of function
