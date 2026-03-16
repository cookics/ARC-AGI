.libPaths(c(file.path(getwd(), "Rlib"), .libPaths()))

suppressPackageStartupMessages(library(jsonlite))

raw <- fromJSON("C:/Users/cooki/Desktop/AI Bench/AI Bench more/filtered_evals_floats.json")
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
evals <- evals[, keep_cols, drop = FALSE]

results <- data.frame()

for (target in names(evals)) {
  others <- evals[, setdiff(names(evals), target), drop = FALSE]
  cc <- complete.cases(cbind(others, evals[[target]]))
  x <- scale(others[cc, , drop = FALSE])
  y <- scale(evals[cc, target])[, 1]
  pca <- prcomp(x, center = FALSE, scale. = FALSE)
  g <- pca$x[, 1]
  if (cor(g, y) < 0) {
    g <- -g
  }
  fit <- lm(y ~ g)
  results <- rbind(
    results,
    data.frame(
      benchmark = target,
      n = sum(cc),
      r = cor(g, y),
      r2 = summary(fit)$r.squared
    )
  )
}

results <- results[order(results$r2, decreasing = TRUE), ]
write.csv(results, "tables/holdout_g_prediction.csv", row.names = FALSE)
print(results)
cat("\nMean R2 =", mean(results$r2), "\n")
cat("Median R2 =", median(results$r2), "\n")
