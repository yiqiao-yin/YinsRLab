#' @title  Unsupervised Representation Learning Algorithm Gabor Learning
#' @description This function accepts input of data and produce Gabor Features.
#' @param symbol
#' @return NULL
#' @examples  GaborFeatures()
#' @export GaborFeatures
#'
#' # Define function
GaborFeatures <- function(
  x = x,
  y = y,
  gb_bank = list(
    scales = 5,
    orientations = 8,
    gabor_rows = 39,
    gabor_columns = 39,
    plot_data = TRUE ),
  gabor_feature_input = list(
    img_nrow = 28,
    img_ncol = 28,
    scales = 6,
    orientations = 12,
    gabor_rows = 28,
    gabor_columns = 28,
    downsample_gabor = FALSE,
    downsample_rows = NULL,
    downsample_cols = NULL,
    normalize_features = FALSE,
    threads = 6,
    verbose = TRUE),
  SVD_nv = 10,
  SVD_nu = 10
  ) {

  # Library
  library(OpenImageR)
  init_gb = GaborFeatureExtract$new()

  # Gabor Filter Bank
  gb_f = init_gb$gabor_filter_bank(
    scales = gb_bank$scales,
    orientations = gb_bank$orientations,
    gabor_rows = gb_bank$gabor_rows,
    gabor_columns = gb_bank$gabor_columns,
    plot_data = gb_bank$plot_data )

  # Plot of Real Part of Gabor Filter Bank
  plt_f = init_gb$plot_gabor(
    real_matrices = gb_f$gabor_real, margin_btw_plots = 0.65,
    thresholding = FALSE )

  # Use Input Data
  dat = init_gb$gabor_feature_engine(
    img_data = as.matrix(x),
    img_nrow = gabor_feature_input$img_nrow, img_ncol = gabor_feature_input$img_ncol,
    scales = gabor_feature_input$scales, orientations = gabor_feature_input$orientations,
    gabor_rows = gabor_feature_input$gabor_rows, gabor_columns = gabor_feature_input$gabor_columns,
    downsample_gabor = gabor_feature_input$downsample_gabor,
    downsample_rows = gabor_feature_input$downsample_rows,
    downsample_cols = gabor_feature_input$downsample_cols,
    normalize_features = gabor_feature_input$normalize_features,
    threads = gabor_feature_input$threads,
    verbose = gabor_feature_input$verbose )

  # Extract SVD from Magnitude
  svd_irlb = irlba::irlba(as.matrix(dat$magnitude), nv = SVD_nv, nu = SVD_nu, verbose = TRUE)
  new_x = as.matrix(dat$magnitude) %*% svd_irlb$v
  data = ClusterR::center_scale(cbind(new_x, dat$energy_aptitude))

  # Output
  return(list(
    GaborData = dat,
    newX = data,
    newData = cbind(y=y, data)
  ))
} # End
