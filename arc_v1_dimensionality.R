# ==============================================================================
# ARC-AGI 1: DIMENSIONALITY & HETEROGENEITY ANALYSIS
# ==============================================================================

if (!require("pacman")) install.packages("pacman")
pacman::p_load(jsonlite, tidyverse, psych, ggplot2, ggrepel)

# --- 1. SETUP & CONFIG ---
BASE_DIR <- getwd() # Assumptions: Running from project root
V1_PREDS <- file.path(BASE_DIR, "arc_agi_v1_public_eval")
V1_TRUTH <- file.path(BASE_DIR, "ARC-AGI", "data", "evaluation")

# Masterpiece Theme Definition
theme_masterpiece <- function() {
    theme_minimal(base_size = 18) +
        theme(
            plot.title = element_text(size = 30, face = "bold", hjust = 0.5, color = "#2C3E50"),
            plot.subtitle = element_text(size = 20, hjust = 0.5, color = "#555555"),
            axis.title = element_text(size = 20, face = "bold", color = "#2C3E50"),
            axis.text = element_text(size = 16, color = "black"),
            legend.position = "top",
            panel.grid.minor = element_blank(),
            panel.background = element_rect(fill = "white", color = NA),
            plot.background = element_rect(fill = "white", color = NA)
        )
}

# --- 2. DATA LOADING (Adapted from Previous Work) ---
# Helper to normalize grids for comparison
normalize_grid <- function(grid) {
    if (is.null(grid) || length(grid) == 0) {
        return("EMPTY")
    }
    tryCatch(paste(unlist(grid), collapse = ","), error = function(e) "ERROR")
}

# Loading Function
load_v1_matrix <- function() {
    cat("[LOAD] Scanning ARC-AGI V1 Data...\n")

    if (!dir.exists(V1_PREDS)) stop("Prediction directory not found: ", V1_PREDS)

    # Load Truth
    truth_files <- list.files(V1_TRUTH, pattern = "*.json", full.names = TRUE)
    truth_cache <- list()
    for (f in truth_files) {
        data <- tryCatch(fromJSON(f, simplifyVector = FALSE), error = function(e) NULL)
        if (!is.null(data)) {
            truth_cache[[basename(f)]] <- lapply(data$test, function(x) normalize_grid(x$output))
        }
    }

    task_ids <- names(truth_cache)
    model_dirs <- list.dirs(V1_PREDS, full.names = TRUE, recursive = FALSE)

    # Initialize Matrix
    mat <- matrix(NA, nrow = length(model_dirs), ncol = length(task_ids))
    rownames(mat) <- basename(model_dirs)
    colnames(mat) <- task_ids

    # Fill Matrix
    for (i in seq_along(model_dirs)) {
        if (i %% 10 == 0) cat(sprintf("\rProcessing model %d/%d...", i, length(model_dirs)))
        model_name <- basename(model_dirs[i])
        pred_files <- list.files(model_dirs[i], pattern = "*.json", full.names = TRUE)

        for (pf in pred_files) {
            tid <- basename(pf)
            if (!tid %in% task_ids) next

            pdata <- tryCatch(fromJSON(pf, simplifyVector = FALSE), error = function(e) NULL)
            if (is.null(pdata)) next

            true_outputs <- truth_cache[[tid]]
            is_correct <- TRUE

            # Check all test pairs
            for (j in seq_along(true_outputs)) {
                start_j <- j - 1 # JSON is 0-indexed sometimes
                pred_entry <- NULL

                # Find entry
                for (item in pdata) {
                    # Try flexible matching for pair index
                    idx <- item$metadata$pair_index
                    if (!is.null(idx) && as.character(idx) == as.character(start_j)) {
                        pred_entry <- item
                        break
                    }
                }
                # Fallback to positional
                if (is.null(pred_entry) && j <= length(pdata)) pred_entry <- pdata[[j]]

                if (is.null(pred_entry)) {
                    is_correct <- FALSE
                    break
                }

                # Check answer (Attempt 1 then 2)
                ans <- pred_entry$attempt_1$answer
                if ((is.null(ans) || length(ans) == 0) && !is.null(pred_entry$attempt_2$answer)) ans <- pred_entry$attempt_2$answer

                if (normalize_grid(ans) != true_outputs[[j]]) {
                    is_correct <- FALSE
                    break
                }
            }
            mat[i, which(task_ids == tid)] <- ifelse(is_correct, 1, 0)
        }
    }
    mat[is.na(mat)] <- 0
    cat("\n[LOAD] Complete.\n")
    return(mat)
}

# Execute Load
raw_mat <- load_v1_matrix()
df_clean <- as.data.frame(raw_mat)

# Filter: Remove items with 0 variance (everyone passed or everyone failed)
item_means <- colMeans(df_clean)
df_valid <- df_clean[, item_means > 0 & item_means < 1]
cat(sprintf("Retained %d items (out of %d) with variance.\n", ncol(df_valid), ncol(df_clean)))


# --- 3. DIMENSIONALITY ANALYSIS ---

cat("\n[ANALYSIS] Running Parallel Analysis...\n")
# Parallel analysis to suggest number of factors
pa <- fa.parallel(df_valid, fm = "minres", fa = "fa", n.iter = 50, show.legend = FALSE)
suggested_factors <- pa$nfact
cat(sprintf("Parallel Analysis suggests %d factors.\n", suggested_factors))

# generate masterpiece scree plot
png("arc_v1_scree.png", width = 1400, height = 1000)
scree_data <- data.frame(
    Factor = 1:length(pa$fa.values),
    Eigenvalue = pa$fa.values,
    Simulated = pa$fa.sim
)
p_scree <- ggplot(scree_data[1:20, ], aes(x = Factor)) +
    geom_line(aes(y = Eigenvalue, color = "Actual Data"), size = 2) +
    geom_point(aes(y = Eigenvalue, color = "Actual Data"), size = 5) +
    geom_line(aes(y = Simulated, color = "Simulated (Random)"), linetype = "dashed", size = 1.5) +
    scale_color_manual(values = c("Actual Data" = "#2C3E50", "Simulated (Random)" = "#E31A1C")) +
    labs(
        title = "ARC-AGI 1: Dimensionality Check",
        subtitle = "Scree Plot: Is Intelligence One Thing (g) or Many?",
        y = "Eigenvalue (Variance Explained)", x = "Factor Number", color = ""
    ) +
    theme_masterpiece() +
    theme(legend.position = c(0.8, 0.8))
print(p_scree)
dev.off()


# --- 4. FACTOR MODELING (2 FACTORS) ---
# We force 2 factors to test the "Heterogeneity" hypothesis
cat("\n[ANALYSIS] Fitting 2-Factor Model...\n")
fa_model <- fa(df_valid, nfactors = 2, rotate = "oblimin", fm = "minres")

# Extract Scores
scores <- as.data.frame(fa_model$scores)
colnames(scores) <- c("Factor1", "Factor2") # Note: Rotation might Flip them, usually F1 is strongest
scores$Model <- rownames(df_valid)

# Identify Potential "Heterogeneous" Models
# Look for models with high discrepancy between Factor 1 (General) and Factor 2 (Specific?)
# Normalizing scores for comparison
scores$F1_Z <- scale(scores$Factor1)
scores$F2_Z <- scale(scores$Factor2)
scores$Discrepancy <- abs(scores$F1_Z - scores$F2_Z)

# Highlight Top Models & Anomalies
top_models <- scores %>%
    arrange(desc(Factor1)) %>%
    head(10)
anomalies <- scores %>%
    arrange(desc(Discrepancy)) %>%
    head(10)
interesting <- unique(rbind(top_models, anomalies))

# --- 5. VISUALIZATION: THE HETEROGENEITY MAP ---

p_factor <- ggplot(scores, aes(x = Factor1, y = Factor2)) +
    geom_point(color = "#2C3E50", alpha = 0.6, size = 3) +
    geom_smooth(method = "lm", color = "#E31A1C", fill = "#E31A1C", alpha = 0.1) +
    geom_text_repel(
        data = interesting, aes(label = Model),
        size = 5, fontface = "bold", box.padding = 0.5, max.overlaps = 20
    ) +
    labs(
        title = "The Landscape of ARC-AGI Intelligence",
        subtitle = "Checking for distinct problem-solving clusters (F1 vs F2)",
        x = "Factor 1 (General Ability)",
        y = "Factor 2 (Specific Skill / Residual)"
    ) +
    theme_masterpiece()

ggsave("arc_v1_heterogeneity.png", p_factor, width = 14, height = 10, dpi = 300)

# --- 6. REPORTING ---
cat("\n[REPORT] Analysis Complete.\n")
print(fa_model)
cat("\n-- Factor Correlations --\n")
print(fa_model$Phi)
cat("\n-- Top Heterogeneous Models (High F1/F2 Split) --\n")
print(anomalies[, c("Model", "Factor1", "Factor2", "Discrepancy")])
