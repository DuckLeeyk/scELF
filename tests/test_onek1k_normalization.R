# Run from the repository root: Rscript tests/test_onek1k_normalization.R
# Test numerical/grouping helpers without loading private data or fitting SCT.
suppressPackageStartupMessages(library(Matrix))
expressions <- parse(file = "analysis/onek1k/preprocessing/normalize_counts.R")
helper_names <- c("as_dgC", "force_exact_target_depth_gc", "make_processing_plan")
for (expr in expressions) {
  if (is.call(expr) && identical(expr[[1]], as.name("<-")) && is.symbol(expr[[2]]) &&
      as.character(expr[[2]]) %in% helper_names) eval(expr)
}

# Nonnegative integer totals, stable support, and idempotence at target depth.
raw <- Matrix(matrix(c(1, 2, 0, 0, 1, 4, 3, 0, 1, 2, 2, 2), nrow = 3), sparse = TRUE)
out <- force_exact_target_depth_gc(raw, 7)
stopifnot(all(colSums(out) == 7), all(out@x >= 0), all(out@x == round(out@x)))
stopifnot(all(as.matrix(out)[as.matrix(raw) == 0] == 0))
stopifnot(identical(as.matrix(out), as.matrix(force_exact_target_depth_gc(out, 7))))
failed <- try(force_exact_target_depth_gc(Matrix(0, 3, 1, sparse = TRUE), 7), silent = TRUE)
stopifnot(inherits(failed, "try-error"))

# Large pools fit alone; small pools merge; undersized cell types use fallback.
CELLTYPE_SPLIT_THRESHOLD <- 10L
MIN_GROUP_CELLS_FOR_SCT <- 3L
plan <- make_processing_plan(c(rep("A", 13), rep("B", 2)),
                            c(rep("P1", 5), rep("P2", 5), "P3", rep("P4", 2), "P1", "P2"))
stopifnot(nrow(plan$plan_df) == 4, sum(plan$plan_df$run_sct) == 3)
stopifnot(identical(sort(unlist(plan$idx_list, use.names = FALSE)), seq_len(15L)))
stopifnot(plan$plan_df$n_cells[plan$plan_df$split_mode == "celltype_then_merged_small_pools"] == 3)
cat("OneK1K normalization helper checks passed.\n")
