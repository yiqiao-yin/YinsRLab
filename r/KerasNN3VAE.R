#' @title Yin's Version of VAE through Keras Framework
#' @description This package uses Keras as framework which provides a high-level neural networks API developed with a focus on enabling fast experimentation.
#' @param NULL
#' @return NULL
#' @examples
#' @export KerasNN3VAE
#'
#' # Define function
KerasNN3VAE <- function(
  X = all[, -1],
  y = all[, 1],
  cutoff = 0.8,
  batch_size = 128L,
  latent_dim = 2L,
  intermediate_dim = 20L,
  epochs = 20L,
  epsilon_std = 1.0,
  verbatim = TRUE
) {
  #' This script demonstrates how to build a variational autoencoder with Keras.
  #' Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114

  # Note: This code reflects pre-TF2 idioms.
  # For an example of a TF2-style modularized VAE, see e.g.: https://github.com/rstudio/keras/blob/master/vignettes/examples/eager_cvae.R
  # Also cf. the tfprobability-style of coding VAEs: https://rstudio.github.io/tfprobability/

  # With TF-2, you can still run this code due to the following line:
  if (verbatim) {print("Initiating environment ...")}
  if (tensorflow::tf$executing_eagerly()) {
    tensorflow::tf$compat$v1$disable_eager_execution()}

  library(keras)
  if (verbatim) {print("Initiating Keras TF backend ...")}
  K <- keras::backend()

  # Data preparation --------------------------------------------------------

  all <- cbind(y, X); all <- as.matrix(all)
  trainIdx <- 1:round(cutoff*nrow(all))
  alltrain <- all[trainIdx, ]; alltest <- all[-trainIdx, ]
  x_train <- alltrain[, -1]; x_test <- alltest[, -1]
  x_train <- scale(x_train); x_test <- scale(x_test)
  y_train <- alltrain[, 1]; y_test <- alltest[, 1]

  # Ensure type of data is "matrix":
  if (class(x_train) == "matrix" && class(x_test) == "matrix") {

    # Parameters --------------------------------------------------------------
    # all parameters are entered as input
    # batch_size = 128L
    # latent_dim = 2L
    # intermediate_dim = 20L
    # epochs = 20L
    # epsilon_std = 1.0
    original_dim = ncol(x_train)

    # Model definition --------------------------------------------------------
    x <- layer_input(shape = c(original_dim))
    h <- layer_dense(x, intermediate_dim, activation = "relu")
    z_mean <- layer_dense(h, latent_dim)
    z_log_var <- layer_dense(h, latent_dim)

    sampling <- function(arg){
      z_mean <- arg[, 1:(latent_dim)]
      z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
      epsilon <- k_random_normal(
        shape = c(k_shape(z_mean)[[1]]),
        mean=0.,
        stddev=epsilon_std)
      z_mean + k_exp(z_log_var/2)*epsilon
    }

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z <- layer_concatenate(list(z_mean, z_log_var)) %>%
      layer_lambda(sampling)

    # we instantiate these layers separately so as to reuse them later
    decoder_h <- layer_dense(units = intermediate_dim, activation = "relu")
    decoder_mean <- layer_dense(units = original_dim, activation = "sigmoid")
    h_decoded <- decoder_h(z)
    x_decoded_mean <- decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae <- keras_model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder <- keras_model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input <- layer_input(shape = latent_dim)
    h_decoded_2 <- decoder_h(decoder_input)
    x_decoded_mean_2 <- decoder_mean(h_decoded_2)
    generator <- keras_model(decoder_input, x_decoded_mean_2)

    vae_loss <- function(x, x_decoded_mean){
      xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
      kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
      xent_loss + kl_loss}

    vae %>% compile(optimizer = "rmsprop", loss = vae_loss)

    # Model training ----------------------------------------------------------
    if (verbatim) {
      print("... Preview summary of architecture ...")
      print(vae)
      print("... training in progress ...")}
    vae %>% fit(
      x_train, x_train,
      shuffle = TRUE,
      epochs = epochs,
      batch_size = batch_size,
      validation_data = list(x_test, x_test) )

    # Visualizations ----------------------------------------------------------
    library(ggplot2)
    library(dplyr)
    x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)
    colnames(x_test_encoded) <- c(paste0("V", 1:ncol(x_test_encoded)))
    PLT <- x_test_encoded %>%
      as_data_frame() %>%
      mutate(class = as.factor(y_test)) %>% # choose mnist$test$y or mnist$test$y[selIdxTest]
      ggplot(aes(x = V1, y = V2, colour = class)) + geom_point() +
      ggtitle(paste0(
        "Ep:", epochs,
        ", BatchSize:", batch_size,
        ", InputDim:", original_dim,
        ", IntermediateDim:", intermediate_dim,
        ", LatenDim:", latent_dim))
    if (verbatim) {plot(PLT)}

    # New Data ----------------------------------------------------------------
    x_train_encoded <- predict(encoder, x_train, batch_size = batch_size)
    colnames(x_train_encoded) <- c(paste0("V", 1:ncol(x_train_encoded)))
    newDataTrain <- data.frame(cbind(Y = as.numeric(y_train), x_train_encoded))
    newDataTest <- data.frame(cbind(Y = as.numeric(y_test), x_test_encoded))
  } else {
    print(paste0("Class of data is not in matrix format. Please reload X and y."))
  }

  # Output
  return(list(
    INPUT = list(
      X = X,
      y = y,
      cutoff = cutoff,
      batch_size = batch_size,
      latent_dim = latent_dim,
      intermediate_dim = intermediate_dim,
      epochs = epochs,
      epsilon_std = epsilon_std,
      verbatim = verbatim ),
    VAE = list(
      encoder = encoder,
      generator = generator,
      vae = vae,
      x_train_encoded = x_train_encoded,
      x_test_encoded = x_test_encoded,
      PLT = PLT),
    OUTPUT = list(
      newDataTrain = newDataTrain,
      newDataTest = newDataTest )
  ))
} # End of function
