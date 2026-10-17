# Reference values from R packages, called by generate_fixtures.py: Rscript reference.R <indir> <out.json>
# Reads id_<name>.csv (samples x features) and rank_<name>.csv; writes one JSON object.
suppressMessages({
  library(jsonlite)
  library(intrinsicDimension)
  library(PCAtools)
  library(scran)
  library(ider)
})
args <- commandArgs(trailingOnly = TRUE)
indir <- args[1]
read_mat <- function(f) as.matrix(read.csv(f, header = FALSE))
out <- list(versions = list(
  R = as.character(getRversion()),
  intrinsicDimension = as.character(packageVersion("intrinsicDimension")),
  PCAtools = as.character(packageVersion("PCAtools")),
  scran = as.character(packageVersion("scran")),
  ider = as.character(packageVersion("ider"))
))

out$intrinsicDimension <- list()
out$ider <- list()
out$danco <- list()
for (f in list.files(indir, pattern = "^id_.*\\.csv$", full.names = TRUE)) {
  X <- read_mat(f)
  name <- sub("^id_(.*)\\.csv$", "\\1", basename(f))
  out$intrinsicDimension[[name]] <- list(
    pca_FO = pcaLocalDimEst(X, ver = "FO")$dim.est,
    pca_fan = pcaLocalDimEst(X, ver = "fan")$dim.est,
    pca_maxgap = pcaLocalDimEst(X, ver = "maxgap")$dim.est,
    mle_k20 = maxLikGlobalDimEst(X, k = 20)$dim.est,
    mle_k20_unbiased = maxLikGlobalDimEst(X, k = 20, unbiased = TRUE)$dim.est,
    mle_k5 = maxLikGlobalDimEst(X, k = 5)$dim.est,
    mle_k20_dnoise = maxLikGlobalDimEst(X, k = 20, n = 20, sigma = 0.1, dnoise = "dnoiseGaussH")$dim.est,
    mle_k20_K10_global = maxLikGlobalDimEst(X, k = 20, K = 10, neighborhood.based = FALSE)$dim.est,
    mle_k20_mean = maxLikGlobalDimEst(X, k = 20, neighborhood.aggregation = "mean")$dim.est,
    # first 30 rows: below 100 points skdim's ESS uses all combinations, like R (no subsampling)
    ess_a_n30 = essLocalDimEst(X[1:30, ], ver = "a")$dim.est,
    ess_b_n30 = essLocalDimEst(X[1:30, ], ver = "b")$dim.est,
    mind_mli_k20_D10 = dancoDimEst(X, k = 20, D = 10, ver = "MIND_MLi")$dim.est,
    mind_mlk_k20_D10 = dancoDimEst(X, k = 20, D = 10, ver = "MIND_MLk")$dim.est
  )
  # DANCo calibration simulates data, so R's calibration is exported and fed to skdim
  D <- ncol(X)
  set.seed(0)
  dn <- dancoDimEst(X, k = 10, D = D, ver = "DANCo")
  out$danco[[name]] <- list(
    k = 10, D = D, N = nrow(X),
    dim_est = dn$dim.est,
    kl_divergence = dn$kl.divergence,
    nocal = as.list(intrinsicDimension:::dancoDimEstNoCalibration(X, 10, D)),
    calibration = lapply(dn$calibration.data$calibration.data, as.list)
  )
  out$ider[[name]] <- list(
    corint_k10_k20 = corint(X, k1 = 10, k2 = 20),
    mada_local_k20 = as.numeric(mada(X, k = 20, local = TRUE))
  )
}

out$PCAtools <- list()
out$scran <- list()
for (f in list.files(indir, pattern = "^rank_.*\\.csv$", full.names = TRUE)) {
  X <- read_mat(f)  # samples x features
  name <- sub("^rank_(.*)\\.csv$", "\\1", basename(f))
  n <- nrow(X); p <- ncol(X)
  # covariance-scale eigenvalues of the centred data
  ev <- svd(scale(X, center = TRUE, scale = FALSE), nu = 0, nv = 0)$d^2 / (n - 1)
  dims <- c(p, n)  # PCAtools convention: features x samples
  set.seed(0)
  pa <- PCAtools::parallelPCA(t(X), max.rank = min(50, n - 1, p), niters = 100, threshold = 0.05)
  out$PCAtools[[name]] <- list(
    MP_noise1 = as.integer(PCAtools::chooseMarchenkoPastur(.dim = dims, var.explained = ev, noise = 1)),
    GD_noise1 = as.integer(PCAtools::chooseGavishDonoho(.dim = dims, var.explained = ev, noise = 1)),
    parallel_niters100_thr0.05 = as.integer(pa$n)
  )
  out$scran[[name]] <- list(
    denoised_noise1 = as.integer(scran::denoisePCANumber(ev, var.tech = p * 1, var.total = sum(apply(X, 2, var))))
  )
}
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), args[2])
