# ==============================================================================
# COMBINED ARC-AGI PSYCHOMETRIC ANALYSIS (V1 + V2) - FIXED PLOTTING
# ==============================================================================

if (!require("pacman")) install.packages("pacman")
pacman::p_load(jsonlite, tidyverse, psych, mirt, reshape2, ggplot2, ggpubr, ggdendro)

# --- 1. SETUP & PATHS ---
get_script_dir <- function() {
  cmdArgs <- commandArgs(trailingOnly = FALSE)
  match <- grep("--file=", cmdArgs)
  if (length(match) > 0) dirname(sub("--file=", "", cmdArgs[match])) else getwd()
}

BASE_DIR <- get_script_dir()

DATASETS <- list(
  V1 = list(
    preds = file.path(BASE_DIR, "arc_agi_v1_public_eval"),
    truth = file.path(BASE_DIR, "ARC-AGI", "data", "evaluation")
  ),
  V2 = list(
    preds = file.path(BASE_DIR, "arc_agi_v2_public_eval"),
    truth = file.path(BASE_DIR, "ARC-AGI-2", "data", "evaluation")
  )
)

# --- 2. DATA INGESTION ---
normalize_grid <- function(grid) {
  if (is.null(grid) || length(grid) == 0) return("EMPTY")
  tryCatch(paste(unlist(grid), collapse = ","), error = function(e) "ERROR")
}

load_dataset_matrix <- function(name, paths) {
  cat(sprintf("\n[LOAD] Loading %s...\n", name))
  if (!dir.exists(paths$preds)) return(NULL)
  
  truth_files <- list.files(paths$truth, pattern = "*.json", full.names = TRUE)
  truth_cache <- list()
  for (f in truth_files) {
    data <- tryCatch(fromJSON(f, simplifyVector = FALSE), error = function(e) NULL)
    if (!is.null(data)) {
      truth_cache[[basename(f)]] <- lapply(data$test, function(x) normalize_grid(x$output))
    }
  }
  
  task_ids <- names(truth_cache)
  model_dirs <- list.dirs(paths$preds, full.names = TRUE, recursive = FALSE)
  mat <- matrix(NA, nrow = length(model_dirs), ncol = length(task_ids))
  rownames(mat) <- basename(model_dirs)
  colnames(mat) <- paste0(name, "_", task_ids)
  
  for (i in seq_along(model_dirs)) {
    model_name <- basename(model_dirs[i])
    pred_files <- list.files(model_dirs[i], pattern = "*.json", full.names = TRUE)
    for (pf in pred_files) {
      tid <- basename(pf)
      if (!tid %in% task_ids) next
      pdata <- tryCatch(fromJSON(pf, simplifyVector = FALSE), error = function(e) NULL)
      if (is.null(pdata)) next
      true_outputs <- truth_cache[[tid]]
      is_correct <- TRUE
      for (j in seq_along(true_outputs)) {
        pred_entry <- NULL
        for (item in pdata) {
          if (!is.null(item$metadata$pair_index) && as.character(item$metadata$pair_index) == as.character(j-1)) {
            pred_entry <- item; break
          }
        }
        if (is.null(pred_entry) && j <= length(pdata)) pred_entry <- pdata[[j]]
        if (is.null(pred_entry)) { is_correct <- FALSE; break }
        
        ans <- NULL
        if (!is.null(pred_entry$attempt_1$answer)) ans <- pred_entry$attempt_1$answer
        if ((is.null(ans) || length(ans)==0) && !is.null(pred_entry$attempt_2$answer)) ans <- pred_entry$attempt_2$answer
        
        if (normalize_grid(ans) != true_outputs[[j]]) { is_correct <- FALSE; break }
      }
      mat[i, which(task_ids == tid)] <- ifelse(is_correct, 1, 0)
    }
  }
  mat[is.na(mat)] <- 0
  return(mat)
}

mat_v1 <- load_dataset_matrix("V1", DATASETS$V1)
mat_v2 <- load_dataset_matrix("V2", DATASETS$V2)
common_models <- intersect(rownames(mat_v1), rownames(mat_v2))
combined_mat <- cbind(mat_v1[common_models, ], mat_v2[common_models, ])
df_clean <- as.data.frame(combined_mat)
item_means <- colMeans(df_clean)
df_clean <- df_clean[, item_means > 0 & item_means < 1]

cat("\n[IRT] Fitting Rasch Model (1PL)...\n")
mod <- mirt(df_clean, 1, itemtype = 'Rasch', verbose = FALSE)
scores <- fscores(mod, full.scores = TRUE)

# --- FORENSIC ANALYSIS ---

# A. PERSON-FIT
fit_stats <- personfit(mod, stats.only = FALSE)
fit_df <- data.frame(Model = rownames(df_clean), Zh = fit_stats$Zh, Theta = scores[,1])
fit_df$Anomaly <- ifelse(fit_df$Zh < -2.0, "Aberrant", "Normal")
anomalies <- fit_df[fit_df$Zh < -2.0, ]
if (nrow(anomalies) > 0) {
  print(knitr::kable(anomalies, caption = "Models with Weird Response Patterns (Zh < -2)"))
}

# B. CLUSTER ANALYSIS
dist_mat <- dist(df_clean, method = "binary")
hc <- hclust(dist_mat, method = "ward.D2")
png("model_family_tree.png", width=1000, height=600)
plot(hc, main="The Genealogy of AI Models", xlab="", sub="", hang = -1)
rect.hclust(hc, k=3, border="red")
dev.off()

# C. TEST INFORMATION FUNCTION (FIXED MANUAL PLOT)
cat("\n--- C. TEST INFORMATION FUNCTION (FIXED) ---\n")

# Generate Theta Grid
theta_grid <- matrix(seq(-6, 6, length.out=100))

# Extract Information for ALL items
info_matrix <- testinfo(mod, theta_grid)

# Create Dataframe for Plotting
plot_data <- data.frame(
  Theta = theta_grid,
  Information = info_matrix
)

# Plot using ggplot2 (Guaranteed to work)
p_info <- ggplot(plot_data, aes(x=Theta, y=Information)) +
  geom_line(color="blue", size=1.5) +
  geom_area(fill="blue", alpha=0.1) +
  labs(title = "Test Information Function: Precision of the ARC Benchmark",
       subtitle = "Higher peaks = Where the test is most accurate at measuring IQ",
       x = "Intelligence (Theta)",
       y = "Test Information") +
  theme_minimal() +
  geom_vline(xintercept = 0, linetype="dashed", color="grey") +
  annotate("text", x=0, y=max(plot_data$Information)*0.1, label="Average Difficulty", angle=90, vjust=-0.5)

ggsave("test_information_curve.png", p_info, width=10, height=6)
cat("Generated 'test_information_curve.png'.\n")

cat("\n[DONE] Forensic Analysis Complete.\n")