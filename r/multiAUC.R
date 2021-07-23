#' @title Supervised Learning Evaluation: Muliple ROC Objects Visualized
#' @description This function accepts input of a list of ROC objects and output a plot with multiple AUCs.
#' @param
#' @return NULL
#' @examples multiAUC()
#' @export multiAUC
#'
#' # Define function
multiAUC <- function(
  aucObjList = sample_list_of_roc_objects,
  plotTITLE = "Comparison of Multiple AUCs",
  midpoint = length(sample_list_of_roc_objects)/2
) {
  plot(aucObjList[[1]], col = 1, lty = 2, main = plotTITLE)
  text(aucObjList[[1]]$sensitivities[midpoint], aucObjList[[1]]$specificities[midpoint],
       paste0("L1, AUC=", round(as.numeric(aucObjList[[1]]$auc), 2)), col = 1, cex = 1)
  # to add to the same graph: add=TRUE
  for (iii in 2:length(aucObjList)) {
    plot(aucObjList[[iii]], col = iii, lty = iii + 1, add = TRUE)
    text(
      aucObjList[[iii]]$sensitivities[midpoint],
      aucObjList[[iii]]$specificities[midpoint],
      paste0("L", iii, ", AUC=", round(as.numeric(aucObjList[[iii]]$auc), 2)), col = iii, cex = 1)} # finish graph
} # end of function
