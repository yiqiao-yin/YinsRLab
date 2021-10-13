#' @title Supervised Learning Evaluation: Muliple Training Paths Visualized
#' @description This function accepts input of a list of history objects and print the paths for training / validating
#' @param
#' @return NULL
#' @examples plot_training_validating_path()
#' @export multiAUC
#'
#' # Define function
plot_training_validating_path <- function(
  input = list(history1, history2, history3, history4),
  name = list(title1, title2, title3, title4)
) {

  # Library
  library(tidyverse) # for dplyr, ggplot2, etc.
  library(testthat)  # unit testing
  library(glue)      # easy print statements

  # Define functions
  get_min_loss <- function(output) {
    output %>%
      filter(data == "validation", metric == "loss") %>%
      summarize(min_loss = min(value, na.rm = TRUE)) %>%
      pull(min_loss) %>%
      round(3)
  }

  # Process Output
  output1 <- as.data.frame(input$history1) %>% mutate(model = name$title1)
  min_loss_1 = get_min_loss(output1)

  output2 <- as.data.frame(input$history2) %>% mutate(model = name$title2)
  min_loss_2 = get_min_loss(output2)

  output3 <- as.data.frame(input$history3) %>% mutate(model = name$title3)
  min_loss_3 = get_min_loss(output3)

  output4 <- as.data.frame(input$history4) %>% mutate(model = name$title4)
  min_loss_4 = get_min_loss(output4)

  # Combine Results
  results <- rbind(output1, output2, output3, output4)

  # Get Loss
  min_loss <- results %>%
    filter(metric == "loss" & data == "validation") %>%
    summarize(min_loss = min(value, na.rm = TRUE)) %>%
    pull()

  # Plot
  results %>%
    filter(metric == "loss") %>%
    ggplot(aes(epoch, value, color = data)) +
    geom_line() +
    geom_hline(yintercept = min_loss, lty = "dashed") +
    facet_wrap(~ model) +
    theme_bw()
} # done

