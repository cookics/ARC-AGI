# ==============================================================================
# ARC-AGI PSYCHOMETRIC ANALYSIS SUITE (ADVANCED)
# "The Psychologist's Toolkit for AI"
# ==============================================================================

# 1. SETUP & LIBRARY MANAGEMENT -----------------------------------------------
if (!require("pacman")) install.packages("pacman")
pacman::p_load(jsonlite, tidyverse, psych, mirt, reshape2, ggplot2, corrplot, ggpubr, lavaan)

# Path Handling
get_script_dir <- function() {
  cmdArgs <- commandArgs(trailingOnly = FALSE)
  match <- grep("--file=", cmdArgs)
  if (length(match) > 0) {
    return(dirname(sub("--file=", "", cmdArgs[match])))
  } else {
    return(getwd())
  }
}

BASE_DIR <- get_script_dir()
PREDS_DIR <- file.path(BASE_DIR, "arc_agi_v1_public_eval") 
TRUTH_DIR <- file.path(BASE_DIR, "ARC-AGI", "data", "evaluation")

cat(sprintf("\n[INIT] Looking for data in:\n - Preds: %s\n - Truth: %s\n", PREDS_DIR, TRUTH_DIR))

if (!dir.exists(PREDS_DIR) || !dir.exists(TRUTH_DIR)) {
  stop("CRITICAL ERROR: Folders not found.")
}

# --- CLASSIFICATION HELPER (UPDATED) ---
get_model_type <- function(model_names) {
  # Default to Standard
  types <- rep("Standard", length(model_names))
  
  # 1. Identify Thinking Models based on keywords and user rules
  # - "thinking", "deep", "reasoning" keywords
  # - ALL "gemini" models
  # - "gpt-5-pro"
  is_thinking <- grepl("thinking|deep|reasoning|gemini|gpt-5-pro", model_names, ignore.case = TRUE)
  types[is_thinking] <- "Thinking"
  
  # 2. Override for specific 'fake' thinking models
  # - "thinking-none" is Standard
  is_fake_thinking <- grepl("thinking-none", model_names, ignore.case = TRUE)
  types[is_fake_thinking] <- "Standard"
  
  return(types)
}

# 2. DATA INGESTION (Robust List Method) --------------------------------------
normalize_grid <- function(grid) {
  if (is.null(grid) || length(grid) == 0) return("EMPTY")
  tryCatch({
    paste(unlist(grid), collapse = ",")
  }, error = function(e) "ERROR")
}

truth_files <- list.files(TRUTH_DIR, pattern = "*.json", full.names = TRUE)
truth_cache <- list()
cat("[DATA] Caching Ground Truths...\n")

for (f in truth_files) {
  data <- tryCatch(fromJSON(f, simplifyVector = FALSE), error = function(e) NULL)
  if (!is.null(data)) {
    truth_cache[[basename(f)]] <- lapply(data$test, function(x) normalize_grid(x$output))
  }
}

task_ids <- names(truth_cache)
model_dirs <- list.dirs(PREDS_DIR, full.names = TRUE, recursive = FALSE)
response_matrix <- matrix(0, nrow = length(model_dirs), ncol = length(task_ids))
rownames(response_matrix) <- basename(model_dirs)
colnames(response_matrix) <- task_ids

cat(sprintf("[DATA] Scoring %d Models against %d Tasks...\n", length(model_dirs), length(task_ids)))

for (i in seq_along(model_dirs)) {
  model_name <- basename(model_dirs[i])
  cat(sprintf("  Processing: %-40s\r", model_name))
  
  pred_files <- list.files(model_dirs[i], pattern = "*.json", full.names = TRUE)
  
  for (pf in pred_files) {
    tid <- basename(pf)
    if (!tid %in% task_ids) next
    
    pdata <- tryCatch(fromJSON(pf, simplifyVector = FALSE), error = function(e) NULL)
    if (is.null(pdata)) next
    
    true_outputs <- truth_cache[[tid]]
    is_task_correct <- TRUE
    
    for (j in seq_along(true_outputs)) {
      target_grid_str <- true_outputs[[j]]
      pred_entry <- NULL
      
      for (item in pdata) {
        if (!is.null(item$metadata$pair_index)) {
          if (as.character(item$metadata$pair_index) == as.character(j-1)) {
            pred_entry <- item; break
          }
        }
      }
      if (is.null(pred_entry)) {
        if (j <= length(pdata)) pred_entry <- pdata[[j]]
      }
      
      if (is.null(pred_entry)) { is_task_correct <- FALSE; break }
      
      ans_grid <- NULL
      if (!is.null(pred_entry$attempt_1) && !is.null(pred_entry$attempt_1$answer)) {
        ans_grid <- pred_entry$attempt_1$answer
      }
      if ((is.null(ans_grid) || length(ans_grid) == 0) && 
          !is.null(pred_entry$attempt_2) && !is.null(pred_entry$attempt_2$answer)) {
        ans_grid <- pred_entry$attempt_2$answer
      }
      
      if (normalize_grid(ans_grid) != target_grid_str) { is_task_correct <- FALSE; break }
    }
    if (is_task_correct) response_matrix[i, tid] <- 1
  }
}
cat("\n[DATA] Matrix construction complete.\n")

# 3. ADVANCED PSYCHOMETRICS ---------------------------------------------------
df <- as.data.frame(response_matrix)

# Variance Filter (Keep items where at least one passed and one failed)
item_means <- colMeans(df)
df_clean <- df[, item_means > 0 & item_means < 1]
cat(sprintf("\n[CLEAN] Remaining Items: %d (Removed %d flat items)\n", ncol(df_clean), ncol(df) - ncol(df_clean)))

if (ncol(df_clean) < 3) stop("Not enough variance to run Psychometrics.")

# --- A. EXTENDED FACTOR ANALYSIS (PCA) ---
cat("\n--- A. COGNITIVE STRUCTURE (PCA) ---\n")
pca_res <- prcomp(df_clean, scale. = FALSE)
var_explained <- (pca_res$sdev^2) / sum(pca_res$sdev^2)
cum_var <- cumsum(var_explained)

# Plot top 20 components
png("pca_full_scree_plot.png", width=800, height=600)
par(mar=c(5,4,4,2) + 0.1)
bp <- barplot(var_explained[1:20] * 100, 
              names.arg = 1:20, 
              main="Cognitive Structure: Variance Explained by Top 20 PC Factors",
              xlab="Principal Component", ylab="% Variance Explained",
              col="steelblue", ylim=c(0, max(var_explained)*100 + 5))
lines(x = bp, y = cum_var[1:20] * 100 / 2, col = "red", type = "b", pch = 19, lwd=2) # Scaled line
legend("topright", legend=c("Individual Var %", "Cumulative Trend"), col=c("steelblue", "red"), lwd=2, pch=c(15,19))
dev.off()
cat("Generated 'pca_full_scree_plot.png'.\n")

# PC1 vs PC2 Biplot (The "Cognitive Map")
scores_pca <- as.data.frame(pca_res$x)
scores_pca$Model <- rownames(df)
# APPLY NEW CLASSIFICATION LOGIC
scores_pca$Type <- get_model_type(scores_pca$Model)

p_pca <- ggplot(scores_pca, aes(x=PC1, y=PC2, color=Type, label=Model)) +
  geom_point(size=3) +
  geom_text(vjust=-0.5, size=3) +
  labs(title="The AI Cognitive Map (PC1 vs PC2)", subtitle="Clustering of Intelligence Types") +
  theme_minimal()
ggsave("pca_cognitive_map.png", p_pca, width=10, height=8)

# --- B. MEASUREMENT INVARIANCE (Configural / Scalar) ---
cat("\n--- B. MEASUREMENT INVARIANCE (Standard vs Thinking) ---\n")
# APPLY NEW CLASSIFICATION LOGIC
group_vec <- get_model_type(rownames(df_clean))
cat(sprintf("Groups: Thinking (N=%d), Standard (N=%d)\n", sum(group_vec=="Thinking"), sum(group_vec=="Standard")))

# We use multiple-group IRT analysis in 'mirt' to test invariance.
# Step 1: Configural Model (Free parameters for both groups)
# Using Rasch (1PL) due to small sample size to assist convergence
tryCatch({
  model_configural <- multipleGroup(df_clean, 1, group = group_vec, itemtype = 'Rasch', verbose = FALSE)
  cat("Configural Model (Baseline): Converged.\n")
  
  # Step 2: Scalar Invariance (Constrain Difficulty 'd' parameters to be equal)
  # If this model fits WORSE, then the test is biased (Difficulty varies by group).
  model_scalar <- multipleGroup(df_clean, 1, group = group_vec, itemtype = 'Rasch', 
                                invariance = c('free_means', 'free_var', 'intercepts'), verbose = FALSE)
  cat("Scalar Model (Constrained): Converged.\n")
  
  # Likelihood Ratio Test
  ano <- anova(model_scalar, model_configural)
  print(ano)
  
  p_val <- ano[2, "p"]
  if (!is.na(p_val) && p_val < 0.05) {
    cat("\nRESULT: Scalar Invariance REJECTED. The test difficulty behaves differently for Thinking vs Standard models.\n")
  } else {
    cat("\nRESULT: Scalar Invariance SUPPORTED. The test measures the same construct for both groups.\n")
  }
}, error = function(e) {
  cat("\n[WARNING] Invariance analysis failed (Sample size likely too small). Skipping.\n")
  print(e)
})

# --- C. DIFFERENTIAL ITEM FUNCTIONING (DIF) ---
cat("\n--- C. DIFFERENTIAL ITEM FUNCTIONING (DIF) ---\n")
# Identify specific items that favor one group over the other
tryCatch({
  # Using 'loop' method for small sample stability
  dif_res <- DIF(model_configural, which.par = c('d'), scheme = 'add', p.adjust = 'fdr')
  sig_dif <- dif_res[dif_res$p < 0.05, ]
  
  if (nrow(sig_dif) > 0) {
    cat(sprintf("Found %d items with significant DIF (Bias).\n", nrow(sig_dif)))
    print(head(sig_dif))
    cat("(Negative 'd' bias usually favors the focal group 'Thinking')\n")
  } else {
    cat("No items showed significant bias (DIF) between groups.\n")
  }
}, error = function(e) cat("DIF Analysis skipped due to convergence issues.\n"))

# --- D. FINAL IQ SCORES (Rasch Model) ---
cat("\n--- D. FINAL IQ SCORES ---\n")
mod_irt <- mirt(df_clean, 1, itemtype = 'Rasch', verbose = FALSE)
scores <- fscores(mod_irt, full.scores = TRUE)

scores_df <- data.frame(
  Model = rownames(df),
  Theta = scores[,1]
)
scores_df$AI_IQ <- round(100 + (scale(scores_df$Theta) * 15))
scores_df$Raw_Acc <- round(rowMeans(df) * 100, 2)
# APPLY NEW CLASSIFICATION LOGIC
scores_df$Type <- get_model_type(scores_df$Model)

final_table <- scores_df[order(-scores_df$AI_IQ), ]
print(knitr::kable(final_table, row.names=FALSE))

# Save Ability Plot
p <- ggplot(scores_df, aes(x=reorder(Model, Theta), y=Theta, fill=Type)) +
  geom_bar(stat="identity") +
  coord_flip() +
  labs(title="Latent Ability (Theta) of AI Models", 
       subtitle="Comparison of Thinking vs Standard Architectures",
       x="Model", y="Latent Ability (Theta)") +
  theme_minimal()
ggsave("irt_ability_plot.png", p, width=10, height=8)

cat("\n[DONE] All analyses complete. Check PNG files for plots.\n")