#' @title Yin's Version of Feature Engineer Using Convolutional Neural Network through Keras Framework
#' @description This package uses Keras as framework which provides a high-level neural networks API developed with a focus on enabling fast experimentation.
#' @param symbol
#' @return NULL
#' @examples
#' @export KerasFeatureEngineerVGG19
#'
#' # Define function
KerasFeatureEngineerVGG19 <- function(
  X = X,
  y = y,
  verbatim = TRUE
) {

  # LIBRARY
  library(keras)
  K <- backend()

  # LOAD MODEL
  model_VGG19 <- application_vgg19(weights = "imagenet") # 512
  # model_VGG19 <- application_vgg19(weights = "imagenet") # 512
  # model_DenseNet121 <- application_densenet121(weights = "imagenet") # 32
  # model_DenseNet169 <- application_densenet169(weights = "imagenet") # 32
  # model_DenseNet201 <- application_densenet201(weights = "imagenet") # 32
  # model_ResNet50 <- application_resnet50(weights = "imagenet") # 2048
  # model_Xception <- application_xception(weights = "imagenet") # 2048

  # RESHAPE
  i = 1; newX = rbind()
  if (verbatim) {pb <- txtProgressBar(min = 0, max = nrow(X), style = 3)}
  for (i in 1:nrow(X)) {
    curr_img <- EBImage::Image(array(X[i, ], dim = c(224, 224, 3)))
    curr_img_resized <- EBImage::resize(curr_img, 224, 224)
    currIMAGE = curr_img_resized
    image <- array_reshape(currIMAGE, dim = c(1, 224, 224, 3))

    # POST PREDICTION WORKFLOW
    # Model: VGG16
    # last_conv_layer_VGG16 <- model_VGG16 %>% get_layer("block5_conv3")
    # iterate <- K$`function`(list(model_VGG16$input), list(last_conv_layer_VGG16$output[1,,,]))
    # conv_layer_output_value %<-% iterate(list(image))
    # currGAPvalue_VGG16 = sapply(1:512, function(kk) {mean(conv_layer_output_value[[1]][,,kk])})
    # # Model: VGG19
    last_conv_layer_VGG19 <- model_VGG19 %>% get_layer("block5_conv4")
    iterate <- K$`function`(list(model_VGG19$input), list(last_conv_layer_VGG19$output[1,,,]))
    conv_layer_output_value %<-% iterate(list(image))
    currGAPvalue_VGG19 = sapply(1:512, function(kk) {mean(conv_layer_output_value[[1]][,,kk])})
    # # Model: DenseNet121
    # last_conv_layer_DenseNet121 <- model_DenseNet121 %>% get_layer("conv5_block16_2_conv")
    # iterate <- K$`function`(list(model_DenseNet121$input), list(last_conv_layer_DenseNet121$output[1,,,]))
    # conv_layer_output_value %<-% iterate(list(image))
    # currGAPvalue_DenseNet121 = sapply(1:32, function(kk) {mean(conv_layer_output_value[[1]][,,kk])})
    # # Model: DenseNet169
    # last_conv_layer_DenseNet169 <- model_DenseNet169 %>% get_layer("conv5_block32_2_conv")
    # iterate <- K$`function`(list(model_DenseNet169$input), list(last_conv_layer_DenseNet169$output[1,,,]))
    # conv_layer_output_value %<-% iterate(list(image))
    # currGAPvalue_DenseNet169 = sapply(1:32, function(kk) {mean(conv_layer_output_value[[1]][,,kk])})
    # # Model: DenseNet201
    # last_conv_layer_DenseNet201 <- model_DenseNet201 %>% get_layer("conv5_block32_2_conv")
    # iterate <- K$`function`(list(model_DenseNet201$input), list(last_conv_layer_DenseNet201$output[1,,,]))
    # conv_layer_output_value %<-% iterate(list(image))
    # currGAPvalue_DenseNet201 = sapply(1:32, function(kk) {mean(conv_layer_output_value[[1]][,,kk])})
    # # Model: ResNet50
    # last_conv_layer_ResNet50 <- model_ResNet50 %>% get_layer("conv5_block3_3_conv")
    # iterate <- K$`function`(list(model_ResNet50$input), list(last_conv_layer_ResNet50$output[1,,,]))
    # conv_layer_output_value %<-% iterate(list(image))
    # currGAPvalue_ResNet50 = sapply(1:2048, function(kk) {mean(conv_layer_output_value[[1]][,,kk])})
    # # Model: Xception
    # last_conv_layer_Xception <- model_Xception %>% get_layer("block14_sepconv2")
    # iterate <- K$`function`(list(model_Xception$input), list(last_conv_layer_Xception$output[1,,,]))
    # conv_layer_output_value %<-% iterate(list(image))
    # currGAPvalue_Xception = sapply(1:2048, function(kk) {mean(conv_layer_output_value[[1]][,,kk])})

    # Store Values
    newX <- rbind(
      newX,
      c(currGAPvalue_VGG19))

    # Checkpoint
    if (verbatim) {setTxtProgressBar(pb, i)}
  }; if (verbatim) {close(pb); print(paste0("Finished generating features!"))} # Done

  # Output
  return(
    list(
      data = list(X, y),
      newData = newX,
      models = list( model_VGG19 )
    )
  )
} # Finished
