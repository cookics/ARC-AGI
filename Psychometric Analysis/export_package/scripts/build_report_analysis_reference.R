.libPaths(c(file.path(getwd(), "Rlib"), .libPaths()))

suppressPackageStartupMessages({
  library(jsonlite)
  library(lavaan)
  library(psych)
  library(GPArotation)
  library(ggplot2)
})

options(stringsAsFactors = FALSE)

report_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
source_root <- normalizePath(file.path(report_dir, ".."), winslash = "/", mustWork = TRUE)

for (subdir in c("figures", "tables", "notes")) {
  dir.create(file.path(report_dir, subdir), recursive = TRUE, showWarnings = FALSE)
}

clean_names <- function(x) {
  x <- tolower(x)
  x <- gsub("\\(.*\\)", "", x)
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("_+", "_", x)
  x <- gsub("^_|_$", "", x)
  x
}

sign_align <- function(x, reference = NULL) {
  x_names <- names(x)
  x <- as.numeric(x)
  names(x) <- x_names
  if (is.null(reference)) {
    if (sum(x, na.rm = TRUE) < 0) {
      x <- -x
    }
    return(x)
  }
  shared <- intersect(names(x), names(reference))
  if (length(shared) > 1 && suppressWarnings(cor(x[shared], reference[shared], use = "pairwise.complete.obs")) < 0) {
    x <- -x
  }
  x
}

tucker_congruence <- function(x, y) {
  shared <- intersect(names(x), names(y))
  x <- x[shared]
  y <- y[shared]
  sum(x * y) / sqrt(sum(x^2) * sum(y^2))
}

pc1_summary <- function(df) {
  cor_mat <- cor(df, use = "pairwise.complete.obs")
  eig <- eigen(cor_mat)
  loadings <- eig$vectors[, 1] * sqrt(eig$values[1])
  names(loadings) <- colnames(df)
  loadings <- sign_align(loadings)
  list(
    cor = cor_mat,
    eigenvalues = eig$values,
    pc1_variance = unname(eig$values[1] / sum(eig$values)),
    loadings = loadings
  )
}

fit_summary <- function(fit, model_name, dataset_name) {
  fm <- fitMeasures(fit, c("cfi", "tli", "rmsea", "srmr", "aic", "bic", "chisq", "df", "pvalue"))
  data.frame(
    dataset = dataset_name,
    model = model_name,
    cfi = unname(fm["cfi"]),
    tli = unname(fm["tli"]),
    rmsea = unname(fm["rmsea"]),
    srmr = unname(fm["srmr"]),
    aic = unname(fm["aic"]),
    bic = unname(fm["bic"]),
    chisq = unname(fm["chisq"]),
    df = unname(fm["df"]),
    pvalue = unname(fm["pvalue"])
  )
}

rank_gaussian <- function(df) {
  out <- lapply(df, function(col) {
    r <- rank(col, na.last = "keep", ties.method = "average")
    p <- r / (sum(!is.na(col)) + 1)
    qnorm(p)
  })
  as.data.frame(out)
}

minmax_scale <- function(df) {
  out <- lapply(df, function(col) {
    rng <- range(col, na.rm = TRUE)
    if (diff(rng) == 0) {
      return(rep(0, length(col)))
    }
    (col - rng[1]) / diff(rng)
  })
  as.data.frame(out)
}

row_standardize <- function(df) {
  mat <- as.matrix(df)
  scaled <- t(scale(t(mat)))
  as.data.frame(scaled)
}

extract_item_g_loadings <- function(std_solution, domain_names, general_name = "g") {
  out <- c()
  for (domain in domain_names) {
    domain_loading <- std_solution$est.std[
      std_solution$lhs == general_name &
        std_solution$rhs == domain &
        std_solution$op == "=~"
    ]
    item_rows <- std_solution[
      std_solution$lhs == domain &
        std_solution$op == "=~",
    ]
    for (i in seq_len(nrow(item_rows))) {
      out[item_rows$rhs[i]] <- item_rows$est.std[i] * domain_loading
    }
  }
  out
}

load_latest_llm_bench <- function(path) {
  raw <- fromJSON(path)
  evals <- as.data.frame(raw$evals)
  rownames(evals) <- make.unique(raw$model$Model)
  for (nm in names(evals)) {
    evals[[nm]] <- as.numeric(evals[[nm]])
  }
  keep_rows <- rowMeans(is.na(evals)) < 0.5
  evals <- evals[keep_rows, , drop = FALSE]
  keep_cols <- colMeans(is.na(evals)) < 0.3
  evals <- evals[, keep_cols, drop = FALSE]
  keep_cols <- vapply(evals, function(x) sd(x, na.rm = TRUE) > 0, logical(1))
  evals[, keep_cols, drop = FALSE]
}

load_web_bench_203 <- function(path) {
  raw <- fromJSON(path)
  df <- as.data.frame(raw$evaluations)
  colnames(df) <- clean_names(colnames(df))
  for (nm in names(df)) {
    df[[nm]] <- as.numeric(df[[nm]])
  }
  rownames(df) <- make.unique(raw$name)
  df
}

load_arena_matrix <- function(path) {
  df <- read.csv(path, check.names = TRUE)
  rownames(df) <- df$model
  df$model <- NULL
  for (nm in names(df)) {
    df[[nm]] <- as.numeric(df[[nm]])
  }
  keep_cols <- vapply(df, function(x) sum(!is.na(x)) > 10 && sd(x, na.rm = TRUE) > 0, logical(1))
  df[, keep_cols, drop = FALSE]
}

# -----------------------------------------------------------------------------
# Project chronology
# -----------------------------------------------------------------------------

timeline <- data.frame(
  folder = c("AI Bench re do", "AI bench g", "AI bench API", "AI Bench more", "HumanData", "Report"),
  last_modified = c(
    "2026-01-25 03:28",
    "2026-01-25 13:31",
    "2026-01-27 02:07",
    "2026-01-27 19:45",
    "2026-01-27 21:45",
    "2026-03-15 21:16"
  ),
  role = c(
    "Earliest clean factor-analysis reset",
    "Matrix reconstruction and early manifold experiments",
    "API/web benchmark synthesis and robustness tests",
    "Latest benchmark refresh plus expert-arena edge analysis",
    "Human psychometric comparison analyses",
    "Consolidated reproducible report workspace"
  )
)
write.csv(timeline, file.path(report_dir, "tables", "project_timeline.csv"), row.names = FALSE)

# -----------------------------------------------------------------------------
# Latest benchmark-only analysis (AI Bench more)
# -----------------------------------------------------------------------------

latest_llm <- load_latest_llm_bench(file.path(source_root, "AI Bench more", "filtered_evals_floats.json"))
latest_pc1 <- pc1_summary(latest_llm)
latest_fa1 <- fa(latest_llm, nfactors = 1, rotate = "none", fm = "minres")

latest_loadings <- data.frame(
  benchmark = rownames(unclass(latest_fa1$loadings)),
  g_loading = as.numeric(unclass(latest_fa1$loadings)[, 1]),
  avg_correlation = (colSums(latest_pc1$cor, na.rm = TRUE) - 1) / (ncol(latest_pc1$cor) - 1),
  stringsAsFactors = FALSE
)
latest_loadings$g_loading <- sign_align(setNames(latest_loadings$g_loading, latest_loadings$benchmark))[latest_loadings$benchmark]
latest_loadings <- latest_loadings[!is.na(latest_loadings$g_loading), ]
latest_loadings <- latest_loadings[order(latest_loadings$g_loading, decreasing = TRUE), ]
write.csv(latest_loadings, file.path(report_dir, "tables", "latest_llm_g_loadings.csv"), row.names = FALSE)

latest_cor_df <- as.data.frame(as.table(latest_pc1$cor), stringsAsFactors = FALSE)
colnames(latest_cor_df) <- c("benchmark_a", "benchmark_b", "correlation")
write.csv(latest_cor_df, file.path(report_dir, "tables", "latest_llm_correlation_matrix.csv"), row.names = FALSE)

latest_cor_pairs <- latest_cor_df[latest_cor_df$benchmark_a < latest_cor_df$benchmark_b, ]
latest_cor_pairs <- latest_cor_pairs[order(latest_cor_pairs$correlation, decreasing = TRUE), ]
write.csv(latest_cor_pairs, file.path(report_dir, "tables", "latest_llm_correlation_pairs.csv"), row.names = FALSE)

heatmap_plot <- ggplot(latest_cor_df, aes(x = benchmark_a, y = benchmark_b, fill = correlation)) +
  geom_tile(color = "white", linewidth = 0.15) +
  scale_fill_gradient2(low = "#2f4858", mid = "white", high = "#bc4749", midpoint = 0, limits = c(-1, 1)) +
  labs(
    title = "Benchmark Correlation Matrix",
    subtitle = sprintf("Latest benchmark panel: %d models x %d benchmarks", nrow(latest_llm), ncol(latest_llm)),
    x = NULL,
    y = NULL,
    fill = "r"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  )
ggsave(
  filename = file.path(report_dir, "figures", "benchmark_correlation_heatmap.png"),
  plot = heatmap_plot,
  width = 9.5,
  height = 8,
  dpi = 300
)

holdout_results <- data.frame()
for (target in names(latest_llm)) {
  others <- latest_llm[, setdiff(names(latest_llm), target), drop = FALSE]
  cc <- complete.cases(cbind(others, latest_llm[[target]]))
  x <- scale(others[cc, , drop = FALSE])
  y <- scale(latest_llm[cc, target])[, 1]
  pca <- prcomp(x, center = FALSE, scale. = FALSE)
  g <- pca$x[, 1]
  if (cor(g, y) < 0) {
    g <- -g
  }
  fit <- lm(y ~ g)
  holdout_results <- rbind(
    holdout_results,
    data.frame(
      benchmark = target,
      n = sum(cc),
      r = cor(g, y),
      r2 = summary(fit)$r.squared
    )
  )
}
holdout_results <- holdout_results[order(holdout_results$r2, decreasing = TRUE), ]
write.csv(holdout_results, file.path(report_dir, "tables", "holdout_g_prediction.csv"), row.names = FALSE)

holdout_plot <- ggplot(holdout_results, aes(x = reorder(benchmark, r2), y = r2)) +
  geom_col(fill = "#386641", width = 0.72) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Held-Out Benchmark Prediction From Common Benchmark Ability",
    subtitle = sprintf("Mean R^2 = %.3f; median R^2 = %.3f", mean(holdout_results$r2), median(holdout_results$r2)),
    x = NULL,
    y = expression(R^2)
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.major.y = element_blank()
  )
ggsave(
  filename = file.path(report_dir, "figures", "holdout_g_prediction.png"),
  plot = holdout_plot,
  width = 8.5,
  height = 6.5,
  dpi = 300
)

efa_rows <- list()
for (nf in seq_len(min(5, floor(ncol(latest_llm) / 2)))) {
  fit <- fa(latest_llm, nfactors = nf, rotate = "oblimin", fm = "minres")
  efa_rows[[length(efa_rows) + 1]] <- data.frame(
    factors = nf,
    variance_explained = sum(fit$values[seq_len(nf)]) / ncol(latest_llm),
    rmsr = fit$rms,
    tli = fit$TLI,
    rmsea = fit$RMSEA[1],
    bic = fit$BIC,
    complexity = mean(fit$complexity)
  )
}
latest_efa_sweep <- do.call(rbind, efa_rows)
write.csv(latest_efa_sweep, file.path(report_dir, "tables", "latest_llm_efa_sweep.csv"), row.names = FALSE)

latest_plot <- ggplot(latest_loadings, aes(x = reorder(benchmark, g_loading), y = g_loading)) +
  geom_col(fill = "#274060", width = 0.72) +
  coord_flip() +
  labs(
    title = "Latest LLM Benchmark g-Loadings",
    subtitle = sprintf("Filtered benchmark matrix: %d models x %d benchmarks", nrow(latest_llm), ncol(latest_llm)),
    x = NULL,
    y = "One-factor loading"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.major.y = element_blank()
  )
ggsave(
  filename = file.path(report_dir, "figures", "latest_llm_g_loadings.png"),
  plot = latest_plot,
  width = 9,
  height = 6,
  dpi = 300
)

# -----------------------------------------------------------------------------
# Standard benchmark robustness analysis (AI bench API)
# -----------------------------------------------------------------------------

web_raw <- load_web_bench_203(file.path(source_root, "AI bench API", "WebDataConverted_203.json"))
web_z <- as.data.frame(scale(web_raw))

transformations <- list(
  Raw = function(x) x,
  ZScore = function(x) as.data.frame(scale(x)),
  RankGaussian = rank_gaussian,
  MinMax = minmax_scale,
  RowStandardized = row_standardize
)

baseline_pc1 <- pc1_summary(web_raw)
transform_rows <- list()
for (name in names(transformations)) {
  transformed <- transformations[[name]](web_raw)
  transformed_pc1 <- pc1_summary(transformed)
  aligned <- sign_align(transformed_pc1$loadings, baseline_pc1$loadings)
  shared <- intersect(names(aligned), names(baseline_pc1$loadings))
  transform_rows[[length(transform_rows) + 1]] <- data.frame(
    transformation = name,
    pc1_variance = transformed_pc1$pc1_variance,
    loading_congruence = tucker_congruence(setNames(aligned, names(transformed_pc1$loadings)), baseline_pc1$loadings),
    loading_correlation = cor(aligned[shared], baseline_pc1$loadings[shared], use = "pairwise.complete.obs")
  )
}
transform_summary <- do.call(rbind, transform_rows)
write.csv(transform_summary, file.path(report_dir, "tables", "web_transform_generality.csv"), row.names = FALSE)

jackknife_rows <- list()
for (benchmark in colnames(web_z)) {
  subset_df <- web_z[, setdiff(colnames(web_z), benchmark), drop = FALSE]
  subset_pc1 <- pc1_summary(subset_df)
  jackknife_rows[[length(jackknife_rows) + 1]] <- data.frame(
    dropped_benchmark = benchmark,
    pc1_variance = subset_pc1$pc1_variance
  )
}
jackknife_summary <- do.call(rbind, jackknife_rows)
write.csv(jackknife_summary, file.path(report_dir, "tables", "web_jackknife_generality.csv"), row.names = FALSE)

single_g_syntax <- paste0("g =~ ", paste(colnames(web_z), collapse = " + "))
bifactor_syntax <- "
g =~ gdpval_aa + terminal_bench_hard + t2_bench_telecom + aa_lcr + aa_omniscience_accuracy +
     aa_omniscience_non_hallucination_rate + humanity_s_last_exam + gpqa_diamond + livecodebench +
     scicode + ifbench + aime_2025 + critpt

Technical =~ livecodebench + scicode + ifbench
Reasoning =~ gpqa_diamond + aime_2025 + humanity_s_last_exam + critpt
Knowledge =~ aa_omniscience_accuracy + aa_omniscience_non_hallucination_rate
Agentic =~ t2_bench_telecom + terminal_bench_hard + aa_lcr + gdpval_aa

g ~~ 0*Technical
g ~~ 0*Reasoning
g ~~ 0*Knowledge
g ~~ 0*Agentic
Technical ~~ 0*Reasoning
Technical ~~ 0*Knowledge
Technical ~~ 0*Agentic
Reasoning ~~ 0*Knowledge
Reasoning ~~ 0*Agentic
Knowledge ~~ 0*Agentic
"

single_g_fit <- cfa(single_g_syntax, data = web_z, std.lv = TRUE, estimator = "MLM")
bifactor_fit <- cfa(bifactor_syntax, data = web_z, std.lv = TRUE, estimator = "MLM")

web_fit_summary <- rbind(
  fit_summary(single_g_fit, "Single g", "AA Web Benchmarks (z-score)"),
  fit_summary(bifactor_fit, "Bifactor 4F", "AA Web Benchmarks (z-score)")
)
write.csv(web_fit_summary, file.path(report_dir, "tables", "web_fit_summary.csv"), row.names = FALSE)

raw_model_cor <- cor(t(as.matrix(web_raw)), use = "pairwise.complete.obs")
z_model_cor <- cor(t(as.matrix(web_z)), use = "pairwise.complete.obs")
diag(raw_model_cor) <- NA_real_
diag(z_model_cor) <- NA_real_

profile_paradox <- data.frame(
  metric = c("Mean pairwise model correlation", "Median pairwise model correlation"),
  raw = c(mean(raw_model_cor, na.rm = TRUE), median(raw_model_cor, na.rm = TRUE)),
  zscore = c(mean(z_model_cor, na.rm = TRUE), median(z_model_cor, na.rm = TRUE))
)
write.csv(profile_paradox, file.path(report_dir, "tables", "profile_paradox.csv"), row.names = FALSE)

# -----------------------------------------------------------------------------
# Expert arena comparison (AI Bench more)
# -----------------------------------------------------------------------------

arena <- load_arena_matrix(file.path(source_root, "AI Bench more", "LLM_arena", "arena_model_matrix.csv"))
arena_pc1 <- pc1_summary(arena)
arena_cor <- cor(arena, use = "pairwise.complete.obs")
arena_omega <- omega(arena_cor, nfactors = 3, n.obs = nrow(arena), plot = FALSE)

arena_summary <- data.frame(
  dataset = "Arena Expert 5k",
  models = nrow(arena),
  categories = ncol(arena),
  pc1_variance = arena_pc1$pc1_variance,
  omega_h = arena_omega$omega_h,
  omega_total = arena_omega$omega.tot
)
write.csv(arena_summary, file.path(report_dir, "tables", "arena_summary.csv"), row.names = FALSE)

arena_best_models <- read.csv(file.path(source_root, "AI Bench more", "LLM_arena", "comprehensive_model_comparison.csv"))
arena_best_models <- arena_best_models[arena_best_models$BIC != "NA", ]
arena_best_models$BIC <- as.numeric(arena_best_models$BIC)
arena_best_models <- arena_best_models[order(arena_best_models$BIC), c("Model", "Description", "BIC", "CFI", "TLI", "RMSEA", "SRMR")]
write.csv(head(arena_best_models, 10), file.path(report_dir, "tables", "arena_best_models.csv"), row.names = FALSE)

# -----------------------------------------------------------------------------
# Human comparison: classic psychometric datasets
# -----------------------------------------------------------------------------

data(HolzingerSwineford1939)
data(Thurstone)
data(Bechtoldt)

fit_classic_models <- function() {
  out <- list()

  hs_uni <- "g =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9"
  hs_corr <- "
    Visual =~ x1 + x2 + x3
    Textual =~ x4 + x5 + x6
    Speed =~ x7 + x8 + x9
  "
  hs_hier <- "
    Visual =~ x1 + x2 + x3
    Textual =~ x4 + x5 + x6
    Speed =~ x7 + x8 + x9
    g =~ Visual + Textual + Speed
  "
  hs_bi <- "
    g =~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
    Visual =~ x1 + x2 + x3
    Textual =~ x4 + x5 + x6
    Speed =~ x7 + x8 + x9
    g ~~ 0*Visual
    g ~~ 0*Textual
    g ~~ 0*Speed
    Visual ~~ 0*Textual
    Visual ~~ 0*Speed
    Textual ~~ 0*Speed
  "

  out[[length(out) + 1]] <- fit_summary(cfa(hs_uni, data = HolzingerSwineford1939, std.lv = TRUE), "Unidimensional", "HolzingerSwineford1939")
  out[[length(out) + 1]] <- fit_summary(cfa(hs_corr, data = HolzingerSwineford1939, std.lv = TRUE), "Correlated", "HolzingerSwineford1939")
  out[[length(out) + 1]] <- fit_summary(cfa(hs_hier, data = HolzingerSwineford1939, std.lv = TRUE), "Hierarchical", "HolzingerSwineford1939")
  out[[length(out) + 1]] <- fit_summary(cfa(hs_bi, data = HolzingerSwineford1939, std.lv = TRUE), "Bifactor", "HolzingerSwineford1939")

  th_names <- colnames(Thurstone)
  th_uni <- paste0("g =~ ", paste(th_names, collapse = " + "))
  th_corr <- paste0(
    "Verbal =~ ", paste(th_names[1:3], collapse = " + "), "\n",
    "Fluency =~ ", paste(th_names[4:6], collapse = " + "), "\n",
    "Reasoning =~ ", paste(th_names[7:9], collapse = " + ")
  )
  th_hier <- paste0(th_corr, "\n", "g =~ Verbal + Fluency + Reasoning")
  th_bi <- paste0(
    "g =~ ", paste(th_names, collapse = " + "), "\n",
    "Verbal =~ ", paste(th_names[1:3], collapse = " + "), "\n",
    "Fluency =~ ", paste(th_names[4:6], collapse = " + "), "\n",
    "Reasoning =~ ", paste(th_names[7:9], collapse = " + "), "\n",
    "g ~~ 0*Verbal\n",
    "g ~~ 0*Fluency\n",
    "g ~~ 0*Reasoning\n",
    "Verbal ~~ 0*Fluency\n",
    "Verbal ~~ 0*Reasoning\n",
    "Fluency ~~ 0*Reasoning"
  )

  out[[length(out) + 1]] <- fit_summary(cfa(th_uni, sample.cov = Thurstone, sample.nobs = 213, std.lv = TRUE), "Unidimensional", "Thurstone")
  out[[length(out) + 1]] <- fit_summary(cfa(th_corr, sample.cov = Thurstone, sample.nobs = 213, std.lv = TRUE), "Correlated", "Thurstone")
  out[[length(out) + 1]] <- fit_summary(cfa(th_hier, sample.cov = Thurstone, sample.nobs = 213, std.lv = TRUE), "Hierarchical", "Thurstone")
  out[[length(out) + 1]] <- fit_summary(cfa(th_bi, sample.cov = Thurstone, sample.nobs = 213, std.lv = TRUE), "Bifactor", "Thurstone")

  be_names <- colnames(Bechtoldt.1)
  be_uni <- paste0("g =~ ", paste(be_names, collapse = " + "))
  be_corr <- paste0(
    "V =~ ", paste(be_names[3:5], collapse = " + "), "\n",
    "F =~ ", paste(be_names[6:8], collapse = " + "), "\n",
    "R =~ ", paste(be_names[15:17], collapse = " + ")
  )
  be_hier <- paste0(be_corr, "\n", "g =~ V + F + R")

  out[[length(out) + 1]] <- fit_summary(cfa(be_uni, sample.cov = Bechtoldt.1, sample.nobs = 212, std.lv = TRUE), "Unidimensional", "Bechtoldt")
  out[[length(out) + 1]] <- fit_summary(cfa(be_corr, sample.cov = Bechtoldt.1, sample.nobs = 212, std.lv = TRUE), "Correlated", "Bechtoldt")
  out[[length(out) + 1]] <- fit_summary(cfa(be_hier, sample.cov = Bechtoldt.1, sample.nobs = 212, std.lv = TRUE), "Hierarchical", "Bechtoldt")

  do.call(rbind, out)
}

classic_fit_summary <- fit_classic_models()
write.csv(classic_fit_summary, file.path(report_dir, "tables", "classic_fit_summary.csv"), row.names = FALSE)

thur_model <- paste0(
  "V =~ ", paste(colnames(Thurstone)[1:3], collapse = " + "), "\n",
  "F =~ ", paste(colnames(Thurstone)[4:6], collapse = " + "), "\n",
  "R =~ ", paste(colnames(Thurstone)[7:9], collapse = " + "), "\n",
  "g =~ V + F + R"
)
bech_model <- paste0(
  "V =~ ", paste(colnames(Bechtoldt.1)[3:5], collapse = " + "), "\n",
  "F =~ ", paste(colnames(Bechtoldt.1)[6:8], collapse = " + "), "\n",
  "R =~ ", paste(colnames(Bechtoldt.1)[15:17], collapse = " + "), "\n",
  "g =~ V + F + R"
)

thur_fit <- cfa(thur_model, sample.cov = Thurstone, sample.nobs = 213, std.lv = TRUE)
bech_fit <- cfa(bech_model, sample.cov = Bechtoldt.1, sample.nobs = 212, std.lv = TRUE)

thur_item_g <- extract_item_g_loadings(standardizedSolution(thur_fit), c("V", "F", "R"))
bech_item_g <- extract_item_g_loadings(standardizedSolution(bech_fit), c("V", "F", "R"))
names(thur_item_g) <- paste0("Item_", seq_along(thur_item_g))
names(bech_item_g) <- paste0("Item_", seq_along(bech_item_g))

g_congruence <- data.frame(
  measure = c("Tucker congruence", "Pearson correlation"),
  value = c(
    tucker_congruence(thur_item_g, bech_item_g),
    cor(thur_item_g[names(bech_item_g)], bech_item_g, use = "pairwise.complete.obs")
  )
)
write.csv(g_congruence, file.path(report_dir, "tables", "classic_g_congruence.csv"), row.names = FALSE)

congruence_plot_df <- data.frame(
  item = c("Sentences", "Vocabulary", "Completion", "First Letters", "4-Letter Words", "Suffixes", "Letter Series", "Pedigrees", "Letter Grouping"),
  thurstone = as.numeric(thur_item_g),
  bechtoldt = as.numeric(bech_item_g)
)

congruence_plot <- ggplot(congruence_plot_df, aes(x = thurstone, y = bechtoldt, label = item)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray60") +
  geom_point(color = "#9e2a2b", size = 3) +
  geom_text(nudge_y = 0.02, size = 3.2) +
  labs(
    title = "Classic Human g-Loading Congruence",
    subtitle = sprintf("Thurstone vs Bechtoldt, Tucker congruence = %.3f", g_congruence$value[1]),
    x = "Thurstone item-level g loading",
    y = "Bechtoldt item-level g loading"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))
ggsave(
  filename = file.path(report_dir, "figures", "classic_g_congruence.png"),
  plot = congruence_plot,
  width = 8,
  height = 6,
  dpi = 300
)

# -----------------------------------------------------------------------------
# Existing heavy analyses to preserve in the synthesis
# -----------------------------------------------------------------------------

icar_existing <- read.csv(file.path(source_root, "HumanData", "icar_exhaustive_fit.csv"))
write.csv(icar_existing, file.path(report_dir, "tables", "icar_exhaustive_fit_referenced.csv"), row.names = FALSE)

existing_llm_cfa <- read.csv(file.path(source_root, "HumanData", "llm_cfa_comparison.csv"))
write.csv(existing_llm_cfa, file.path(report_dir, "tables", "existing_llm_cfa_comparison.csv"), row.names = FALSE)

# -----------------------------------------------------------------------------
# Cross-dataset comparison figure
# -----------------------------------------------------------------------------

comparison_bars <- data.frame(
  dataset = c(
    "Latest standard benchmarks",
    "AA web benchmarks (raw)",
    "AA web benchmarks (z-score)",
    "Arena Expert 5k"
  ),
  pc1_variance = c(
    latest_pc1$pc1_variance,
    baseline_pc1$pc1_variance,
    pc1_summary(web_z)$pc1_variance,
    arena_pc1$pc1_variance
  )
)
write.csv(comparison_bars, file.path(report_dir, "tables", "cross_dataset_pc1.csv"), row.names = FALSE)

comparison_plot <- ggplot(comparison_bars, aes(x = reorder(dataset, pc1_variance), y = pc1_variance, fill = dataset)) +
  geom_col(width = 0.72, show.legend = FALSE) +
  coord_flip() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Generality Is Strong on Standard Benchmarks, Weaker on Expert Tasks",
    x = NULL,
    y = "First principal component variance"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid.major.y = element_blank()
  )
ggsave(
  filename = file.path(report_dir, "figures", "cross_dataset_pc1.png"),
  plot = comparison_plot,
  width = 8.5,
  height = 5.5,
  dpi = 300
)

# -----------------------------------------------------------------------------
# Write a compact machine-readable summary for the paper
# -----------------------------------------------------------------------------

summary_lines <- c(
  sprintf("Latest standard benchmark PC1 variance: %.3f", latest_pc1$pc1_variance),
  sprintf("AA web raw PC1 variance: %.3f", baseline_pc1$pc1_variance),
  sprintf("AA web z-score PC1 variance: %.3f", pc1_summary(web_z)$pc1_variance),
  sprintf("Arena expert PC1 variance: %.3f", arena_pc1$pc1_variance),
  sprintf("Arena omega hierarchical: %.3f", arena_omega$omega_h),
  sprintf("Classic g congruence (Tucker): %.3f", g_congruence$value[1]),
  sprintf("Classic g correlation: %.3f", g_congruence$value[2])
)
writeLines(summary_lines, file.path(report_dir, "notes", "key_numbers.txt"))

cat("Report analysis build complete.\n")
