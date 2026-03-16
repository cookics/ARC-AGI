# ==============================================================================
# ARC-AGI HETEROGENEITY ANALYSIS (V3: Permutation-Based Inference)
# Addressing the "Over-Consistency" (P=1.000) Anomaly
# ==============================================================================

if (!require("pacman")) install.packages("pacman")
# We'll try to use 'vegan' for proper binary matrix shuffling if available
pacman::p_load(jsonlite, tidyverse, psych, ggplot2, ggrepel, vegan)

# --- 1. CONFIGURATION ---
BASE_DIR <- getwd()
V1_PREDS <- file.path(BASE_DIR, "arc_agi_v1_public_eval")
V1_TRUTH <- file.path(BASE_DIR, "ARC-AGI", "data", "evaluation")

# --- 2. DATA LOADING ---
normalize_grid <- function(grid) {
    if (is.null(grid) || length(grid) == 0) {
        return("EMPTY")
    }
    paste(unlist(grid), collapse = ",")
}

load_matrix <- function() {
    cat("[LOAD] Loading Truth Data...\n")
    truth_files <- list.files(V1_TRUTH, pattern = "*.json", full.names = TRUE)
    truth_cache <- list()
    for (f in truth_files) {
        data <- tryCatch(fromJSON(f, simplifyVector = FALSE), error = function(e) NULL)
        if (!is.null(data)) truth_cache[[basename(f)]] <- lapply(data$test, function(x) normalize_grid(x$output))
    }

    cat("[LOAD] Loading Predictions...\n")
    model_dirs <- list.dirs(V1_PREDS, full.names = TRUE, recursive = FALSE)
    task_ids <- names(truth_cache)

    mat <- matrix(0, nrow = length(model_dirs), ncol = length(task_ids))
    rownames(mat) <- basename(model_dirs)
    colnames(mat) <- task_ids

    for (i in seq_along(model_dirs)) {
        m_name <- basename(model_dirs[i])
        cat(sprintf("\rScanning %s (%d/%d)...", m_name, i, length(model_dirs)))
        pred_files <- list.files(model_dirs[i], pattern = "*.json", full.names = TRUE)
        for (pf in pred_files) {
            tid <- basename(pf)
            if (!tid %in% task_ids) next
            pdata <- tryCatch(fromJSON(pf, simplifyVector = FALSE), error = function(e) NULL)
            if (is.null(pdata)) next
            true_outputs <- truth_cache[[tid]]
            is_correct <- TRUE
            for (j in seq_along(true_outputs)) {
                pred_entry <- if (length(pdata) >= j) pdata[[j]] else NULL
                if (is.null(pred_entry)) {
                    is_correct <- FALSE
                    break
                }
                ans <- pred_entry$attempt_1$answer
                if ((is.null(ans) || length(ans) == 0) && !is.null(pred_entry$attempt_2$answer)) ans <- pred_entry$attempt_2$answer
                if (normalize_grid(ans) != true_outputs[[j]]) {
                    is_correct <- FALSE
                    break
                }
            }
            if (is_correct) mat[i, which(task_ids == tid)] <- 1
        }
    }
    cat("\n[LOAD] Complete.\n")
    return(mat)
}

raw_mat <- load_matrix()
item_means <- colMeans(raw_mat)
valid_idx <- item_means > 0 & item_means < 1
clean_mat <- raw_mat[, valid_idx]

# --- 3. CORE METRICS ---
calc_diagnostics <- function(m) {
    scores <- rowMeans(m)
    p_rates <- colMeans(m)
    # Rasch approx
    theta <- qnorm(pmin(pmax(scores, 0.01), 0.99))
    beta <- -qnorm(pmin(pmax(p_rates, 0.01), 0.99))
    P <- plogis(outer(theta, beta, "-"))
    W <- P * (1 - P)
    W[W < 1e-6] <- 1e-6
    Z <- (m - P) / sqrt(W)
    outfit <- rowMeans(Z^2)
    # Loevinger H
    N <- nrow(m)
    n_items <- ncol(m)
    # Simpler H calc for speed during permutations
    # Correct H = 1 - (Shuffled Obs Errors / Expected Errors)
    # But since we'll use actual shuttling, we just need a metric.
    # Let's use the mean inter-item covariance as a proxy in the simulation
    # if full H is too slow. But let's try full H.

    # Mokken H helper (Internal)
    h_metric <- function(mat) {
        # Using a subset of items if too many to speed up permutation cycles?
        # No, let's keep it complete.
        cv <- cov(mat)
        obs_c <- sum(cv[lower.tri(cv)])
        # Max Cov for binary is min(p1,p2) - p1*p2
        p <- colMeans(mat)
        max_c <- 0
        for (i in 1:(length(p) - 1)) {
            for (j in (i + 1):length(p)) {
                max_c <- max_c + (min(p[i], p[j]) - p[i] * p[j])
            }
        }
        return(obs_c / max_c)
    }

    return(list(outfit = outfit, h = h_metric(m)))
}

obs_results <- calc_diagnostics(clean_mat)

# --- 4. NULL MODEL: PERMUTATION TEST (Fixed Marginals) ---
# Previous bias (P=1.0) came from Bernoulli (Rasch) expecting coin-flips.
# Permutation shuffling tests if the *cohesion* is better than random given the scores.

n_sims <- 200 # Lower sims for speed, but more rigorous algorithm
cat("Running Permutation Shuffling (Swap Algorithm)...\n")

# Use vegan's permatfull if available, else a simpler fallback
# permatfull(m, fixedmar="both", mtype="count") works for binary if binary=TRUE
null_mats <- permatfull(clean_mat, fixedmar = "both", mtype = "count", times = n_sims)

sim_outfits <- matrix(0, nrow = nrow(clean_mat), ncol = n_sims)
sim_h <- numeric(n_sims)

for (i in 1:n_sims) {
    if (i %% 20 == 0) cat(sprintf("\rPermutation %d/%d...", i, n_sims))
    sim_m <- null_mats$perm[[i]]
    # Recalculating metrics under shuffled null
    res <- calc_diagnostics(sim_m)
    sim_outfits[, i] <- res$outfit
    sim_h[i] <- res$h
}
cat("\n")

# Correct P-Values: What % of random shuffles show MORE surprise than observed?
# P = (Sum(Sim_Outfit >= Obs_Outfit) + 1) / (N_sims + 1)
model_diag <- data.frame(
    Model = rownames(clean_mat),
    Score = rowMeans(clean_mat),
    Outfit_Obs = obs_results$outfit,
    # P-value for "Misfit" (High Outfit)
    P_Misfit = sapply(1:nrow(clean_mat), function(idx) (sum(sim_outfits[idx, ] >= obs_results$outfit[idx]) + 1) / (n_sims + 1)),
    # P-value for "Over-Consistency" (Low Outfit)
    P_Consistency = sapply(1:nrow(clean_mat), function(idx) (sum(sim_outfits[idx, ] <= obs_results$outfit[idx]) + 1) / (n_sims + 1))
)

p_h_global <- (sum(sim_h >= obs_results$h) + 1) / (n_sims + 1)

# --- 5. FINAL REPORT ---
sink("heterogeneity_report.txt")
cat("====================================================\n")
cat("ARC-AGI-1: STATISTICAL AUDIT & HETEROGENEITY REPORT\n")
cat("====================================================\n\n")

cat("--- AUDIT: Addressing the P=1.000 Anomaly ---\n")
cat("Previously, P-values clustered at 1.0 because the Rasch null model assumes STOCHASTIC behavior.\n")
cat("LLMs are largely DETERMINISTIC. They solve items they know and fail items they don't, with very little 'luck' noise.\n")
cat("This results in 'Infranormal' variance (MSQ < 1).\n\n")
cat("New Method: Permutation Test (Swap Algorithm).\n")
cat("We shuffled the response matrix 200 times while keeping exact model and task scores constant.\n")
cat("This tests if the models' successes are 'locally grouped' or 'scattered' compared to random permutations.\n\n")

cat("--- GLOBAL SCALE METRICS (Permutation Based) ---\n")
cat(sprintf("Observed Loevinger's H: %.3f\n", obs_results$h))
cat(sprintf("Empirical P-value for Scale Cohesion: %.4f\n", p_h_global))
cat("Interpretation: If P < .05, the benchmark is significantly more structured than random.\n\n")

cat("--- MODEL-SPECIFIC DIAGNOSTICS ---\n")
cat("P_Misfit < .05: Significant Heterogeneity (Unusually messy performance).\n")
cat("P_Consistency < .05: Unusually Guttman-like performance (No deviations at all).\n\n")
print(model_diag %>% arrange(P_Misfit))

cat("\n--- TARGET CASE ANALYSIS ---\n")
target_models <- c("qwen3-235b-a22b-instruct-2507", "gpt-5-pro-2025-10-06")
for (tm in target_models) {
    entry <- model_diag %>% filter(Model == tm)
    if (nrow(entry) > 0) {
        cat(sprintf("%s:\n", tm))
        cat(sprintf("  Outfit MSQ: %.3f\n", entry$Outfit_Obs))
        cat(sprintf("  P-Value (Heterogeneity Signal): %.4f\n", entry$P_Misfit))
        if (entry$P_Misfit < 0.05) {
            cat("  VERDICT: SIGNAL DETECTED. This model has specific skill pockets.\n")
        } else {
            cat("  VERDICT: NO SIGNAL. Performance matches the general ability scale.\n")
        }
    }
}
sink()

# --- 6. VISUALIZATION ---
# Plot P-Values to confirm uniform distribution or signal
p_dist <- ggplot(model_diag, aes(x = P_Misfit)) +
    geom_histogram(bins = 10, fill = "#2C3E50", alpha = 0.8) +
    labs(title = "Distribution of P-Values (Permutation Null)", subtitle = "Expected: Uniform if no signal. Peaks at ends indicate misfits or over-consistency.", x = "P-Value (Misfit)", y = "Count") +
    theme_minimal()

ggsave("p_value_distribution.png", p_dist, width = 10, height = 6)

cat("[DONE] Enhanced Permutation Analysis complete.\n")
