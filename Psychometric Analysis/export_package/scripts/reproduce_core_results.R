root_dir <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
out_fig <- file.path(root_dir, "reproduced", "figures")
out_tbl <- file.path(root_dir, "reproduced", "tables")
dir.create(out_fig, recursive = TRUE, showWarnings = FALSE)
dir.create(out_tbl, recursive = TRUE, showWarnings = FALSE)

evals <- read.csv(file.path(root_dir, "data", "latest", "latest_benchmark_matrix_raw.csv"), check.names = FALSE)
rownames(evals) <- make.unique(as.character(evals$model))
evals$model <- NULL
for (nm in names(evals)) evals[[nm]] <- as.numeric(evals[[nm]])

keep_rows <- rowMeans(is.na(evals)) < 0.5
evals <- evals[keep_rows, , drop = FALSE]
keep_cols <- colMeans(is.na(evals)) < 0.3
evals <- evals[, keep_cols, drop = FALSE]
keep_cols <- vapply(evals, function(x) sd(x, na.rm = TRUE) > 0, logical(1))
evals <- evals[, keep_cols, drop = FALSE]

cor_mat <- cor(evals, use = "pairwise.complete.obs")
write.csv(cor_mat, file.path(out_tbl, "benchmark_correlation_matrix.csv"))

pair_df <- as.data.frame(as.table(cor_mat), stringsAsFactors = FALSE)
colnames(pair_df) <- c("benchmark_a", "benchmark_b", "correlation")
pair_df <- pair_df[pair_df$benchmark_a < pair_df$benchmark_b, ]
pair_df <- pair_df[order(pair_df$correlation, decreasing = TRUE), ]
write.csv(pair_df, file.path(out_tbl, "benchmark_correlation_pairs.csv"), row.names = FALSE)

eig <- eigen(cor_mat)
pc1_var <- eig$values[1] / sum(eig$values)
loadings <- eig$vectors[, 1] * sqrt(eig$values[1])
names(loadings) <- colnames(cor_mat)
if (sum(loadings) < 0) loadings <- -loadings
load_df <- data.frame(
  benchmark = names(loadings),
  loading = as.numeric(loadings),
  row.names = NULL
)
load_df <- load_df[order(load_df$loading, decreasing = TRUE), ]
write.csv(load_df, file.path(out_tbl, "pc1_loadings.csv"), row.names = FALSE)

holdout <- data.frame()
for (target in names(evals)) {
  others <- evals[, setdiff(names(evals), target), drop = FALSE]
  cc <- complete.cases(cbind(others, evals[[target]]))
  x <- scale(others[cc, , drop = FALSE])
  y <- scale(evals[cc, target])[, 1]
  pca <- prcomp(x, center = FALSE, scale. = FALSE)
  g <- pca$x[, 1]
  if (cor(g, y) < 0) g <- -g
  fit <- lm(y ~ g)
  holdout <- rbind(
    holdout,
    data.frame(
      benchmark = target,
      n = sum(cc),
      r = cor(g, y),
      r2 = summary(fit)$r.squared
    )
  )
}
holdout <- holdout[order(holdout$r2, decreasing = TRUE), ]
write.csv(holdout, file.path(out_tbl, "holdout_g_prediction.csv"), row.names = FALSE)

png(file.path(out_fig, "benchmark_correlation_heatmap.png"), width = 1200, height = 1000)
par(mar = c(9, 9, 4, 2))
image(
  1:ncol(cor_mat),
  1:nrow(cor_mat),
  t(cor_mat[nrow(cor_mat):1, ]),
  col = colorRampPalette(c("#2f4858", "white", "#bc4749"))(100),
  axes = FALSE,
  main = "Benchmark Correlation Matrix"
)
axis(1, at = 1:ncol(cor_mat), labels = colnames(cor_mat), las = 2, cex.axis = 0.7)
axis(2, at = 1:nrow(cor_mat), labels = rev(rownames(cor_mat)), las = 2, cex.axis = 0.7)
box()
dev.off()

png(file.path(out_fig, "holdout_g_prediction.png"), width = 1200, height = 900)
par(mar = c(10, 10, 4, 2))
barplot(
  holdout$r2,
  names.arg = holdout$benchmark,
  horiz = TRUE,
  las = 1,
  col = "#386641",
  xlim = c(0, 1),
  main = sprintf(
    "Held-Out Benchmark Prediction\nMean R2 = %.3f | Median R2 = %.3f",
    mean(holdout$r2),
    median(holdout$r2)
  ),
  xlab = "R2"
)
dev.off()

cat("Reproduced core outputs in ./reproduced\n")
