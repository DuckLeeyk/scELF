#!/usr/bin/env Rscript

options(
  stringsAsFactors = FALSE,
  warn = 1,
  future.globals.maxSize = 30 * 1024^3
)

# The R argparse package provides the same command-line style as the Python scripts.
# Only generic preprocessing defaults are retained; every data/environment path
# is supplied by the caller. Numerical helpers below retain the source algorithm.
suppressPackageStartupMessages(library(argparse))
parser <- ArgumentParser(description = "Normalize OneK1K counts using sctransform v1 and optional exact-depth integer postprocessing.")
parser$add_argument("--input", required = TRUE, help = "Input h5ad with raw UMI counts in X.")
parser$add_argument("--output", required = TRUE, help = "Output corrected-count h5ad; must differ from input.")
parser$add_argument("--summary", required = TRUE, help = "Per-group processing summary TSV.")
parser$add_argument("--python", default = NULL, help = "Optional Python executable for reticulate with anndata/numpy/scipy.")
parser$add_argument("--celltype-column", default = "predicted.celltype.l2")
parser$add_argument("--pool-column", default = "pool_number")
parser$add_argument("--cell-types", nargs = "+", default = NULL, help = "Optional raw cell-type labels; output contains selected cells only.")
parser$add_argument("--target-depth", type = "integer", default = 1261L)
parser$add_argument("--celltype-split-threshold", type = "integer", default = 10000L)
parser$add_argument("--min-group-cells", type = "integer", default = 200L)
parser$add_argument("--gene-min-cells", type = "integer", default = 5L)
parser$add_argument("--bin-size", type = "integer", default = 256L)
parser$add_argument("--verbosity", type = "integer", default = 1L)
parser$add_argument("--seed", type = "integer", default = 42L)
parser$add_argument("--no-exact-depth", action = "store_true", help = "Skip largest-remainder exact-depth postprocessing.")
parser$add_argument("--allow-error-fallback", action = "store_true", help = "Explicitly allow raw-count fallback on SCT fitting errors.")
parser$add_argument("--allow-identical-output", action = "store_true")
args <- parser$parse_args()
set.seed(args$seed)
input_h5ad <- args$input
output_h5ad <- args$output
summary_tsv <- args$summary
OUTPUT_X_MODE <- "corrected_umi"
VST_FLAVOR <- "v1"
VST_BIN_SIZE <- args$bin_size
VST_VERBOSITY <- args$verbosity
GENE_MIN_CELLS_DEFAULT <- args$gene_min_cells
CELLTYPE_SPLIT_THRESHOLD <- args$celltype_split_threshold
MIN_GROUP_CELLS_FOR_SCT <- args$min_group_cells
ALLOW_ERROR_FALLBACK <- args$allow_error_fallback
STOP_IF_OUTPUT_IDENTICAL <- !args$allow_identical_output
WRITE_SUMMARY_TSV <- TRUE
TARGET_DEPTH <- args$target_depth
FORCE_EXACT_TARGET_DEPTH <- !args$no_exact_depth
if (!is.null(args$python)) {
  python_exe <- if (file.exists(args$python)) {
    normalizePath(args$python, mustWork = TRUE, winslash = "/")
  } else {
    unname(Sys.which(args$python))
  }
  if (!nzchar(python_exe)) stop("Cannot resolve --python executable: ", args$python)
  Sys.setenv(RETICULATE_PYTHON = python_exe)
}
if (any(c(TARGET_DEPTH, CELLTYPE_SPLIT_THRESHOLD, MIN_GROUP_CELLS_FOR_SCT,
          GENE_MIN_CELLS_DEFAULT, VST_BIN_SIZE) < 1L)) stop("Count/depth thresholds must be positive.")
if (normalizePath(input_h5ad, mustWork = FALSE) == normalizePath(output_h5ad, mustWork = FALSE)) {
  stop("Input and output must be different files.")
}
dir.create(dirname(output_h5ad), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(summary_tsv), recursive = TRUE, showWarnings = FALSE)

## =========================================================
## Packages
## =========================================================
suppressPackageStartupMessages({
  library(Matrix)
  library(sctransform)
  library(reticulate)
  library(future)
})

future::plan("sequential")

## =========================================================
## Checks
## =========================================================
if (!file.exists(input_h5ad)) {
  stop("Input file does not exist: ", input_h5ad)
}

if (!is.na(TARGET_DEPTH)) {
  if (!is.numeric(TARGET_DEPTH) || length(TARGET_DEPTH) != 1L || !is.finite(TARGET_DEPTH)) {
    stop("TARGET_DEPTH must be NA or one finite numeric scalar.")
  }
  if (TARGET_DEPTH <= 0) {
    stop("TARGET_DEPTH must be > 0 when provided.")
  }
  TARGET_DEPTH <- as.integer(round(TARGET_DEPTH))
}

if (FORCE_EXACT_TARGET_DEPTH && is.na(TARGET_DEPTH)) {
  stop("FORCE_EXACT_TARGET_DEPTH = TRUE requires a numeric TARGET_DEPTH (cannot be NA).")
}

if (file.exists(output_h5ad)) {
  stop("Output file already exists: ", output_h5ad)
}

if (WRITE_SUMMARY_TSV && file.exists(summary_tsv)) {
  stop("Summary file already exists: ", summary_tsv)
}

## =========================================================
## Logging helpers
## =========================================================
msg <- function(...) {
  cat(format(Sys.time(), "%F %T"), " | ", ..., "\n", sep = "")
}

py_gc <- function() {
  gc()
  try(reticulate::py_run_string("import gc; gc.collect()"), silent = TRUE)
}

close_adata_file <- function(adata_obj) {
  try({
    if (!is.null(adata_obj) && !is.null(adata_obj$file)) {
      adata_obj$file$close()
    }
  }, silent = TRUE)
}

msg(
  "TARGET_DEPTH = ",
  if (is.na(TARGET_DEPTH)) "NA(default median depth in correct_counts)" else as.character(TARGET_DEPTH),
  " | FORCE_EXACT_TARGET_DEPTH = ",
  FORCE_EXACT_TARGET_DEPTH
)

## =========================================================
## Matrix helpers
## =========================================================
as_dgC <- function(x) {
  if (!inherits(x, "Matrix")) {
    x <- Matrix::Matrix(x, sparse = TRUE)
  }
  x <- as(x, "CsparseMatrix")
  x <- as(x, "dgCMatrix")
  x
}

restore_full_corrected <- function(corrected, umi) {
  corrected <- as_dgC(corrected)

  if (is.null(rownames(corrected))) {
    if (nrow(corrected) == nrow(umi)) {
      rownames(corrected) <- rownames(umi)
    } else {
      stop("corrected matrix has no rownames and nrow != nrow(umi)")
    }
  }

  if (is.null(colnames(corrected))) {
    if (ncol(corrected) == ncol(umi)) {
      colnames(corrected) <- colnames(umi)
    } else {
      stop("corrected matrix has no colnames and ncol != ncol(umi)")
    }
  }

  if (!setequal(colnames(corrected), colnames(umi))) {
    stop("Column names of corrected matrix do not match umi")
  }
  corrected <- corrected[, colnames(umi), drop = FALSE]

  # Restore raw UMI counts for genes omitted from the corrected matrix.
  if (!setequal(rownames(corrected), rownames(umi))) {
    miss_genes <- setdiff(rownames(umi), rownames(corrected))
    if (length(miss_genes) > 0L) {
      raw_rest <- umi[miss_genes, , drop = FALSE]
      raw_rest <- as_dgC(raw_rest)
      corrected <- rbind(corrected, raw_rest)
    }
  }

  corrected <- corrected[rownames(umi), colnames(umi), drop = FALSE]
  corrected <- as_dgC(corrected)
  corrected
}

# ---------------------------------------------------------
# Enforce an exact target depth on a genes-by-cells sparse matrix:
# each cell (column) must sum to target_depth.
#
# Procedure:
# 1) Scale each cell's counts proportionally to target_depth.
# 2) Take the floor of the scaled counts.
# 3) Allocate the remaining counts using the largest remainder method.
#
# The result has:
# - Nonnegative integer counts.
# - Column sums exactly equal to target_depth.
# - No new nonzero gene entries; counts stay within the original support.
# ---------------------------------------------------------
force_exact_target_depth_gc <- function(mat_gc, target_depth) {
  mat_gc <- as_dgC(mat_gc)

  if (!is.numeric(target_depth) || length(target_depth) != 1L ||
      !is.finite(target_depth) || target_depth <= 0) {
    stop("target_depth must be one finite positive number.")
  }

  target_depth <- as.integer(round(target_depth))

  nc <- ncol(mat_gc)
  nr <- nrow(mat_gc)

  p_old <- mat_gc@p
  i_old <- mat_gc@i
  x_old <- mat_gc@x

  new_i_list <- vector("list", nc)
  new_x_list <- vector("list", nc)

  for (j in seq_len(nc)) {
    start <- p_old[j] + 1L
    end   <- p_old[j + 1L]

    if (start > end) {
      stop(
        "Cell ", j,
        " has zero total counts after correction/fallback, cannot force target depth = ",
        target_depth
      )
    }

    idx0 <- i_old[start:end]       # 0-based row indices for dgCMatrix slot
    vals <- x_old[start:end]

    total <- sum(vals)

    if (!is.finite(total) || total <= 0) {
      stop(
        "Cell ", j,
        " has non-finite or non-positive total counts (", total,
        "), cannot force target depth."
      )
    }

    # Keep integer counts unchanged when their total already equals the target.
    if (abs(total - target_depth) < 1e-8 &&
        all(abs(vals - round(vals)) < 1e-8) &&
        all(vals >= 0)) {
      vals_int <- as.integer(round(vals))
      keep <- vals_int > 0L
      new_i_list[[j]] <- idx0[keep]
      new_x_list[[j]] <- as.numeric(vals_int[keep])
      next
    }

    scaled <- vals * (target_depth / total)
    base   <- floor(scaled)
    rem_n  <- as.integer(target_depth - sum(base))

    if (rem_n > 0L) {
      frac <- scaled - base
      ord  <- order(frac, decreasing = TRUE)
      base[ord[seq_len(rem_n)]] <- base[ord[seq_len(rem_n)]] + 1
    } else if (rem_n < 0L) {
      # Rare floating-point overshoot: subtract from the smallest fractions first.
      frac <- scaled - base
      ord  <- order(frac, decreasing = FALSE)
      need <- -rem_n
      hit  <- 0L
      for (k in ord) {
        if (base[k] > 0) {
          base[k] <- base[k] - 1
          hit <- hit + 1L
          if (hit == need) break
        }
      }
      if (sum(base) != target_depth) {
        stop("Internal rounding adjustment failed while forcing exact target depth.")
      }
    }

    vals_int <- as.integer(base)

    if (sum(vals_int) != target_depth) {
      stop(
        "Internal error: forced counts in cell ", j,
        " do not sum to target_depth."
      )
    }

    keep <- vals_int > 0L
    new_i_list[[j]] <- idx0[keep]
    new_x_list[[j]] <- as.numeric(vals_int[keep])
  }

  lens <- vapply(new_x_list, length, integer(1))
  p_new <- c(0L, cumsum(lens))

  i_new <- if (sum(lens) > 0L) {
    as.integer(unlist(new_i_list, use.names = FALSE))
  } else {
    integer(0)
  }

  x_new <- if (sum(lens) > 0L) {
    as.numeric(unlist(new_x_list, use.names = FALSE))
  } else {
    numeric(0)
  }

  out <- methods::new(
    "dgCMatrix",
    i = i_new,
    p = as.integer(p_new),
    x = x_new,
    Dim = c(as.integer(nr), as.integer(nc)),
    Dimnames = dimnames(mat_gc)
  )

  cs <- Matrix::colSums(out)
  if (!all(cs == target_depth)) {
    bad <- which(cs != target_depth)[1]
    stop(
      "Exact target-depth enforcement failed. Example bad cell: ",
      bad, ", colSum = ", cs[bad], ", target = ", target_depth
    )
  }

  out
}

## =========================================================
## Python imports + helpers
## =========================================================
anndata <- import("anndata", convert = FALSE)
np      <- import("numpy", convert = FALSE)
sp      <- import("scipy.sparse", convert = FALSE)

reticulate::py_run_string("
import numpy as np
import scipy.sparse as sp

def get_X_rows(adata, idx):
    return adata.X[idx, :]

def as_csr_float32(x):
    return sp.csr_matrix(x, dtype=np.float32)

def as_dense_float32(x):
    return np.asarray(x, dtype=np.float32)
")

py <- reticulate::py

## =========================================================
## Read metadata in backed mode
## =========================================================
msg("Opening input h5ad in backed='r' mode ...")
adata_in <- anndata$read_h5ad(input_h5ad, backed = "r")

selected_obs <- NULL
if (!is.null(args$cell_types)) {
  initial_obs <- py_to_r(adata_in$obs)
  if (!(args$celltype_column %in% colnames(initial_obs))) stop("Cell-type column is missing.")
  available <- unique(as.character(initial_obs[[args$celltype_column]]))
  if (any(!args$cell_types %in% available)) stop("Unknown --cell-types values.")
  keep <- as.character(initial_obs[[args$celltype_column]]) %in% args$cell_types
  selected_obs <- which(keep) - 1L
  adata_in <- adata_in$`__getitem__`(r_to_py(as.integer(selected_obs)))
}
shape <- as.integer(py_to_r(adata_in$shape))
n_cells <- shape[1]
n_genes <- shape[2]

msg("n_cells = ", n_cells, "; n_genes = ", n_genes)

obs_df <- py_to_r(adata_in$obs)

required_obs_cols <- c(args$pool_column, args$celltype_column)
missing_cols <- setdiff(required_obs_cols, colnames(obs_df))
if (length(missing_cols) > 0) {
  close_adata_file(adata_in)
  stop("Missing required obs columns: ", paste(missing_cols, collapse = ", "))
}

celltype_vec <- as.character(obs_df[[args$celltype_column]])
pool_vec     <- as.character(obs_df[[args$pool_column]])

if (length(celltype_vec) != n_cells || length(pool_vec) != n_cells) {
  close_adata_file(adata_in)
  stop("obs columns length mismatch with adata shape.")
}

celltype_vec[is.na(celltype_vec)] <- "__NA_celltype__"
pool_vec[is.na(pool_vec)]         <- "__NA_pool__"

obs_names <- as.character(py_to_r(adata_in$obs_names$to_list()))
var_names <- as.character(py_to_r(adata_in$var_names$to_list()))

if (length(obs_names) != n_cells) {
  close_adata_file(adata_in)
  stop("obs_names length mismatch: got ", length(obs_names), ", expected ", n_cells)
}

if (length(var_names) != n_genes) {
  close_adata_file(adata_in)
  stop("var_names length mismatch: got ", length(var_names), ", expected ", n_genes)
}

if (anyNA(obs_names) || any(obs_names == "")) {
  close_adata_file(adata_in)
  stop("obs_names contains NA or empty strings.")
}

if (anyNA(var_names) || any(var_names == "")) {
  close_adata_file(adata_in)
  stop("var_names contains NA or empty strings.")
}

if (anyDuplicated(obs_names)) {
  msg("Warning: duplicated obs_names detected.")
}

if (anyDuplicated(var_names)) {
  msg("Warning: duplicated var_names detected.")
}

## =========================================================
## Build processing plan
## =========================================================
make_processing_plan <- function(celltype_vec, pool_vec) {
  celltypes <- unique(celltype_vec)

  plan_rows <- list()
  idx_list <- list()
  k <- 1L

  for (ct in celltypes) {
    idx_ct <- which(celltype_vec == ct)
    n_ct <- length(idx_ct)

    if (n_ct > CELLTYPE_SPLIT_THRESHOLD) {
      # Split large cell types into sequencing pools.
      split_by_pool <- split(idx_ct, pool_vec[idx_ct])
      pool_sizes <- vapply(split_by_pool, length, integer(1))

      big_pool_names <- names(split_by_pool)[pool_sizes >= MIN_GROUP_CELLS_FOR_SCT]
      sml_pool_names <- names(split_by_pool)[pool_sizes <  MIN_GROUP_CELLS_FOR_SCT]

      # Process sufficiently large pools separately.
      for (pn in big_pool_names) {
        idx_now <- split_by_pool[[pn]]

        plan_rows[[k]] <- data.frame(
          group_id = paste0("group_", k),
          celltype = ct,
          pool_label = pn,
          split_mode = "celltype_then_pool",
          n_cells = length(idx_now),
          run_sct = TRUE,
          fallback_reason = "",
          stringsAsFactors = FALSE
        )
        idx_list[[k]] <- idx_now
        k <- k + 1L
      }

      # Merge small pools before checking the minimum group size.
      if (length(sml_pool_names) > 0L) {
        idx_small <- sort(unlist(split_by_pool[sml_pool_names], use.names = FALSE))
        n_small <- length(idx_small)

        if (n_small >= MIN_GROUP_CELLS_FOR_SCT) {
          plan_rows[[k]] <- data.frame(
            group_id = paste0("group_", k),
            celltype = ct,
            pool_label = "__MERGED_SMALL_POOLS__",
            split_mode = "celltype_then_merged_small_pools",
            n_cells = n_small,
            run_sct = TRUE,
            fallback_reason = "",
            stringsAsFactors = FALSE
          )
        } else {
          plan_rows[[k]] <- data.frame(
            group_id = paste0("group_", k),
            celltype = ct,
            pool_label = "__MERGED_SMALL_POOLS__",
            split_mode = "celltype_then_merged_small_pools",
            n_cells = n_small,
            run_sct = FALSE,
            fallback_reason = paste0(
              "merged small pools still have too few cells (< ",
              MIN_GROUP_CELLS_FOR_SCT,
              "); keep raw counts before optional exact target-depth postprocessing"
            ),
            stringsAsFactors = FALSE
          )
        }

        idx_list[[k]] <- idx_small
        k <- k + 1L
      }

    } else {
      # Process smaller cell types as a single group.
      run_this <- n_ct >= MIN_GROUP_CELLS_FOR_SCT
      reason <- if (run_this) "" else paste0(
        "cell type has too few cells (< ",
        MIN_GROUP_CELLS_FOR_SCT,
        "); keep raw counts before optional exact target-depth postprocessing"
      )

      plan_rows[[k]] <- data.frame(
        group_id = paste0("group_", k),
        celltype = ct,
        pool_label = "__ALL_POOLS__",
        split_mode = "celltype_only",
        n_cells = n_ct,
        run_sct = run_this,
        fallback_reason = reason,
        stringsAsFactors = FALSE
      )
      idx_list[[k]] <- idx_ct
      k <- k + 1L
    }
  }

  plan_df <- do.call(rbind, plan_rows)

  # Verify that the plan covers every cell exactly once.
  combined_idx <- sort(unlist(idx_list, use.names = FALSE))
  if (!identical(combined_idx, seq_len(length(celltype_vec)))) {
    stop("Processing plan does not cover all cells exactly once.")
  }

  list(plan_df = plan_df, idx_list = idx_list)
}

plan_obj <- make_processing_plan(celltype_vec, pool_vec)
plan_df  <- plan_obj$plan_df
idx_list <- plan_obj$idx_list

msg("Planned groups: ", nrow(plan_df))
msg("Groups to run SCT: ", sum(plan_df$run_sct), " / ", nrow(plan_df))
msg("Groups to keep raw counts before optional exact-depth postprocessing: ", sum(!plan_df$run_sct), " / ", nrow(plan_df))

## =========================================================
## Core worker
## =========================================================
process_one_group <- function(plan_row, idx_cell_1based) {
  group_id   <- plan_row$group_id[[1]]
  ct_name    <- plan_row$celltype[[1]]
  pool_label <- plan_row$pool_label[[1]]
  split_mode <- plan_row$split_mode[[1]]
  run_sct    <- isTRUE(plan_row$run_sct[[1]])
  fb_reason  <- plan_row$fallback_reason[[1]]

  msg(
    "Processing ", group_id,
    " | celltype = ", ct_name,
    " | pool_label = ", pool_label,
    " | split_mode = ", split_mode,
    " | n_cells = ", length(idx_cell_1based),
    " | run_sct = ", run_sct
  )

  # Extract this group's cells-by-genes submatrix from the backed AnnData object.
  x_py <- py$get_X_rows(adata_in, as.integer(idx_cell_1based - 1L))
  x_r  <- py_to_r(x_py)
  rm(x_py)
  py_gc()

  # Convert to a sparse Matrix representation.
  if (!inherits(x_r, "Matrix")) {
    x_r <- Matrix::Matrix(x_r, sparse = TRUE)
  } else if (!inherits(x_r, "sparseMatrix")) {
    x_r <- Matrix::Matrix(x_r, sparse = TRUE)
  }

  # cells x genes -> genes x cells
  umi <- Matrix::t(x_r)
  umi <- as_dgC(umi)
  rm(x_r)
  py_gc()

  rownames(umi) <- var_names
  colnames(umi) <- obs_names[idx_cell_1based]

  # Check that UMI counts are integers.
  if (length(umi@x) > 0L) {
    delta <- max(abs(umi@x - round(umi@x)))
    if (delta > 1e-6) {
      stop(
        "Non-integer-like values found in adata.X for group ", group_id,
        ". sctransform expects raw UMI counts."
      )
    }
    umi@x <- round(umi@x)
  }

  raw_gc <- umi
  raw_ct_sparse <- Matrix::t(raw_gc)
  raw_ct_sparse <- as_dgC(raw_ct_sparse)
  rownames(raw_ct_sparse) <- obs_names[idx_cell_1based]
  colnames(raw_ct_sparse) <- var_names

  out_gc <- raw_gc
  used_fallback <- FALSE
  fallback_reason <- ""
  error_message <- ""
  used_target_depth_in_sct <- FALSE
  exact_target_applied <- FALSE

  # -------------------------------------------------------
  # SCT branch
  # -------------------------------------------------------
  if (!run_sct) {
    used_fallback <- TRUE
    fallback_reason <- fb_reason
    msg("Skipping SCT for ", group_id, " | reason: ", fb_reason)
  } else {
    fit_result <- tryCatch({
      umi_counts <- Matrix::colSums(raw_gc)

      if (any(!is.finite(umi_counts))) {
        stop("Non-finite UMI library sizes found in group ", group_id)
      }
      if (any(umi_counts <= 0)) {
        stop(
          "Some cells have zero or negative total UMI counts in group ",
          group_id,
          ". sctransform cannot model these cells safely."
        )
      }

      # Use umi/log_umi only; do not regress out percent.mt or pool_number here.
      cell_attr <- data.frame(
        umi       = as.numeric(umi_counts),
        log_umi   = log10(as.numeric(umi_counts)),
        row.names = obs_names[idx_cell_1based],
        stringsAsFactors = FALSE,
        check.names = FALSE
      )

      min_cells_use <- min(GENE_MIN_CELLS_DEFAULT, ncol(raw_gc))
      if (min_cells_use < 1L) min_cells_use <- 1L

      vst_out <- sctransform::vst(
        umi               = raw_gc,
        cell_attr         = cell_attr,
        latent_var        = "log_umi",
        batch_var         = NULL,
        bin_size          = VST_BIN_SIZE,
        min_cells         = min_cells_use,
        residual_type     = "none",
        return_cell_attr  = TRUE,
        return_gene_attr  = FALSE,
        vst.flavor        = VST_FLAVOR,
        verbosity         = VST_VERBOSITY
      )

      cell_attr_for_correct <- if (!is.null(vst_out$cell_attr)) {
        vst_out$cell_attr
      } else {
        cell_attr
      }

      cell_attr_for_correct <- cell_attr_for_correct[colnames(raw_gc), , drop = FALSE]

      corrected <- sctransform::correct_counts(
        x            = vst_out,
        umi          = raw_gc,
        cell_attr    = cell_attr_for_correct,
        scale_factor = if (is.na(TARGET_DEPTH)) NA_real_ else as.numeric(TARGET_DEPTH),
        verbosity    = VST_VERBOSITY
      )

      corrected <- restore_full_corrected(corrected, raw_gc)

      rm(vst_out, cell_attr, cell_attr_for_correct, umi_counts)
      py_gc()

      list(
        out_gc = corrected,
        used_fallback = FALSE,
        fallback_reason = "",
        error_message = "",
        used_target_depth_in_sct = !is.na(TARGET_DEPTH)
      )
    }, error = function(e) {
      err_msg <- conditionMessage(e)

      if (!ALLOW_ERROR_FALLBACK) {
        stop(
          "SCT failed for ", group_id,
          " | celltype = ", ct_name,
          " | pool_label = ", pool_label,
          " | error: ", err_msg
        )
      }

      warning(
        "SCT failed for ", group_id,
        ". Keeping raw counts for this group before optional exact target-depth postprocessing. Error: ",
        err_msg
      )

      list(
        out_gc = raw_gc,
        used_fallback = TRUE,
        fallback_reason = "SCT runtime error; keep raw counts before optional exact target-depth postprocessing",
        error_message = err_msg,
        used_target_depth_in_sct = FALSE
      )
    })

    out_gc <- fit_result$out_gc
    used_fallback <- isTRUE(fit_result$used_fallback)
    fallback_reason <- fit_result$fallback_reason
    error_message <- fit_result$error_message
    used_target_depth_in_sct <- isTRUE(fit_result$used_target_depth_in_sct)

    rm(fit_result)
    py_gc()
  }

  # -------------------------------------------------------
  # Optional exact postprocessing:
  # force every cell to have exactly TARGET_DEPTH total counts
  # -------------------------------------------------------
  if (FORCE_EXACT_TARGET_DEPTH) {
    msg(
      "Applying exact target-depth postprocessing for ", group_id,
      " | target_depth = ", TARGET_DEPTH
    )
    out_gc <- force_exact_target_depth_gc(out_gc, TARGET_DEPTH)
    exact_target_applied <- TRUE
  }

  # genes x cells -> cells x genes
  out_ct <- Matrix::t(out_gc)
  out_ct <- as_dgC(out_ct)

  if (nrow(out_ct) != length(idx_cell_1based) || ncol(out_ct) != length(var_names)) {
    stop(
      "output matrix dimension mismatch for group ", group_id,
      ": got ", nrow(out_ct), " x ", ncol(out_ct),
      ", expected ", length(idx_cell_1based), " x ", length(var_names)
    )
  }

  rownames(out_ct) <- obs_names[idx_cell_1based]
  colnames(out_ct) <- var_names

  diff_ct <- out_ct - raw_ct_sparse
  changed_entries <- Matrix::nnzero(diff_ct)
  max_abs_diff <- if (changed_entries > 0L) max(abs(diff_ct@x)) else 0

  final_ncount <- Matrix::rowSums(out_ct)
  final_ncount_min <- min(final_ncount)
  final_ncount_max <- max(final_ncount)
  all_final_ncount_equal_target <- if (is.na(TARGET_DEPTH)) {
    NA
  } else {
    all(final_ncount == TARGET_DEPTH)
  }

  rm(diff_ct, out_gc, raw_gc, raw_ct_sparse, umi)
  py_gc()

  msg(
    "Finished ", group_id,
    " | fallback = ", used_fallback,
    " | changed_entries = ", changed_entries,
    " | max_abs_diff = ", signif(max_abs_diff, 6),
    " | final_ncount_min = ", final_ncount_min,
    " | final_ncount_max = ", final_ncount_max
  )

  list(
    mat = out_ct,
    used_fallback = used_fallback,
    fallback_reason = fallback_reason,
    changed_entries = changed_entries,
    max_abs_diff = max_abs_diff,
    error_message = error_message,
    used_target_depth_in_sct = used_target_depth_in_sct,
    exact_target_applied = exact_target_applied,
    target_depth = if (is.na(TARGET_DEPTH)) NA_real_ else as.numeric(TARGET_DEPTH),
    final_ncount_min = as.numeric(final_ncount_min),
    final_ncount_max = as.numeric(final_ncount_max),
    all_final_ncount_equal_target = all_final_ncount_equal_target
  )
}

## =========================================================
## Loop over groups
## =========================================================
out_chunks <- vector("list", nrow(plan_df))
summary_list <- vector("list", nrow(plan_df))

for (i in seq_len(nrow(plan_df))) {
  res <- process_one_group(
    plan_row = plan_df[i, , drop = FALSE],
    idx_cell_1based = idx_list[[i]]
  )

  out_chunks[[i]] <- res$mat

  summary_list[[i]] <- data.frame(
    group_id = plan_df$group_id[i],
    celltype = plan_df$celltype[i],
    pool_label = plan_df$pool_label[i],
    split_mode = plan_df$split_mode[i],
    n_cells = plan_df$n_cells[i],
    planned_run_sct = plan_df$run_sct[i],
    used_fallback = isTRUE(res$used_fallback),
    fallback_reason = if (nzchar(res$fallback_reason)) res$fallback_reason else "",
    used_target_depth_in_sct = isTRUE(res$used_target_depth_in_sct),
    exact_target_applied = isTRUE(res$exact_target_applied),
    target_depth = if (is.na(res$target_depth)) NA_real_ else as.numeric(res$target_depth),
    final_ncount_min = as.numeric(res$final_ncount_min),
    final_ncount_max = as.numeric(res$final_ncount_max),
    all_final_ncount_equal_target = if (is.na(res$all_final_ncount_equal_target)) NA else isTRUE(res$all_final_ncount_equal_target),
    changed_entries = as.numeric(res$changed_entries),
    max_abs_diff = as.numeric(res$max_abs_diff),
    error_message = if (nzchar(res$error_message)) res$error_message else "",
    stringsAsFactors = FALSE
  )

  rm(res)
  py_gc()
}

summary_df <- do.call(rbind, summary_list)

msg("Groups processed: ", nrow(summary_df))
msg("Fallback groups: ", sum(summary_df$used_fallback), " / ", nrow(summary_df))
msg("Groups with any change: ", sum(summary_df$changed_entries > 0), " / ", nrow(summary_df))
msg("Total changed entries across all groups: ", format(sum(summary_df$changed_entries), scientific = FALSE))

if (!is.na(TARGET_DEPTH)) {
  msg(
    "Groups with all cells exactly at target depth: ",
    sum(isTRUE(summary_df$all_final_ncount_equal_target) | (!is.na(summary_df$all_final_ncount_equal_target) & summary_df$all_final_ncount_equal_target)),
    " / ",
    nrow(summary_df)
  )
}

if (WRITE_SUMMARY_TSV) {
  write.table(
    summary_df,
    file = summary_tsv,
    sep = "\t",
    quote = TRUE,
    row.names = FALSE,
    col.names = TRUE
  )
  msg("Summary written to: ", summary_tsv)
}

if (all(summary_df$used_fallback) && !FORCE_EXACT_TARGET_DEPTH) {
  close_adata_file(adata_in)
  stop("All groups fell back to raw counts and no exact target-depth postprocessing was applied. Output would be identical to input.")
}

if (STOP_IF_OUTPUT_IDENTICAL && all(summary_df$changed_entries == 0)) {
  close_adata_file(adata_in)
  stop("All processed groups produced zero changes relative to raw counts. Refusing to write an output identical to input.")
}

## =========================================================
## Combine chunks and restore original cell order
## =========================================================
msg("Combining chunks ...")

combined_idx <- unlist(idx_list, use.names = FALSE)
out_mat_cat <- do.call(rbind, out_chunks)

rm(out_chunks)
py_gc()

ord <- order(combined_idx)
out_mat <- out_mat_cat[ord, , drop = FALSE]

rm(out_mat_cat, combined_idx, ord, idx_list)
py_gc()

if (nrow(out_mat) != n_cells || ncol(out_mat) != n_genes) {
  close_adata_file(adata_in)
  stop(
    "Output matrix dimension mismatch: got ",
    nrow(out_mat), " x ", ncol(out_mat),
    ", expected ", n_cells, " x ", n_genes
  )
}

msg("Final matrix assembled: ", nrow(out_mat), " x ", ncol(out_mat))

if (!is.na(TARGET_DEPTH) && FORCE_EXACT_TARGET_DEPTH) {
  final_ncount <- Matrix::rowSums(out_mat)
  if (!all(final_ncount == TARGET_DEPTH)) {
    bad <- which(final_ncount != TARGET_DEPTH)[1]
    close_adata_file(adata_in)
    stop(
      "Final matrix failed exact target-depth check. Example bad cell index: ",
      bad,
      " | rowSum = ", final_ncount[bad],
      " | target = ", TARGET_DEPTH
    )
  }
  msg("Verified: all cells have exact nCount_RNA = ", TARGET_DEPTH)
}

## =========================================================
## Close input backed file
## =========================================================
msg("Closing input backed file ...")
close_adata_file(adata_in)
rm(adata_in)
py_gc()

## =========================================================
## Write output h5ad
## =========================================================
msg("Writing output by full read/write ...")

adata_full <- anndata$read_h5ad(input_h5ad)
if (!is.null(selected_obs)) {
  adata_full <- adata_full$`__getitem__`(r_to_py(as.integer(selected_obs)))$copy()
}
# Preserve raw-depth metadata: it is the log-library-size covariate used by
# the source analysis. X contains corrected counts with the requested depth.

out_mat <- as_dgC(out_mat)
adata_full$X <- py$as_csr_float32(r_to_py(out_mat))

adata_full$write_h5ad(
  output_h5ad,
  compression = "gzip",
  convert_strings_to_categoricals = FALSE
)

rm(adata_full, out_mat)
py_gc()

msg("Done.")
msg("Output written to: ", output_h5ad)
