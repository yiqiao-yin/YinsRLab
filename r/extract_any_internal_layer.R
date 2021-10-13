#' @title Supervised Learning Evaluation: Extract Output
#' @description This function accepts input of a fitted neural network model and extract input
#' @param
#' @return NULL
#' @examples extract_any_internal_layer()
#' @export multiAUC
#'
#' # Define function
extract_any_internal_layer <- function(
  model = model,
  what_layer = what_layer,
  X = X
) {

  # Remak:
  # model: this is the actual model | Entering summary(model) should give the sequential print
  # what_layer: this must be a name in the model
  # X: this must be in matrix or array format

  # Library
  library(keras)

  # Starter
  K <- backend()

  # Get Internal Structure
  get_conv_layer <- model %>% get_layer(what_layer)
  iterate <- K$`function`(list(model$input), list(get_conv_layer$output))
  conv_layer_output_value %<-% iterate(list(X))

  # Output
  return(
    list(
      summary = summary(model),
      output = conv_layer_output_value
    )
  )
} # End
