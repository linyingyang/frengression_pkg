# Original binary ATE data-generating function from
# xwshen51/frengression/data_causl/data_causl.R at commit 40d0ee43f4f72e6dc7b8c21c0f22b436c0cc7c0d.
# This wrapper calls the same function and parameter values as paper_exp/binary.ipynb.
# Install the R package causl from https://github.com/rje42/causl before running.
if (!requireNamespace("causl", quietly = TRUE)) {
  stop("R package 'causl' is required; install it with remotes::install_github('rje42/causl')")
}
library(causl)

data.causl <- function(n=10000, nI=3, nX=1, nO=1, nS=1, ate=2, beta_cov=0, strength_instr=3, strength_conf=1, strength_outcome=0.2, binary_intervention=TRUE){
  
  forms <- list(list(), A ~ 1, Y ~ A, ~ 1)
  
  if(binary_intervention){
    fam <- list(rep(1, nI + nX + nO + nS), 5, 1, 1)
  } else {
    fam <- list(rep(1, nI + nX + nO + nS), 1, 1, 1)
  }
  
  pars <- list()
  
  # Specify the formula and parameters for each covariate type
  ## Instrumental variables (I)
  if (nI > 0) {
    for (i in seq_len(nI)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("I", i, " ~ 1")))
      pars[paste0("I", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }
  
  ## Confounders (X)
  if (nX > 0) {
    for (i in seq_len(nX)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("X", i, " ~ 1")))
      pars[paste0("X", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }
  
  ## Outcome variables (O)
  if (nO > 0) {
    for (i in seq_len(nO)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("O", i, " ~ 1")))
      pars[paste0("O", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }
  
  ## Spurious variables (S)
  if (nS > 0) {
    for (i in seq_len(nS)) {
      forms[[1]] <- c(forms[[1]], as.formula(paste0("S", i, " ~ 1")))
      pars[paste0("S", i)] <- list(list(beta = beta_cov, phi = 1))
    }
  }
  
  # Specify the formula for A given covariates
  ## Add I to the propensity score formula
  if (nI > 0) {
    for (i in seq_len(nI)) {
      forms[[2]] <- update.formula(forms[[2]], paste0("A ~ . + I", i))
    }
  }
  
  ## Add X to the propensity score formula
  if (nX > 0) {
    for (i in seq_len(nX)) {
      forms[[2]] <- update.formula(forms[[2]], paste0("A ~ . + X", i))
    }
  }
  
  # Parameters for copula
  parY <- list()
  parY_names <- c()

  if (nX > 0) {
    parY <- c(parY, rep(list(list(beta = strength_conf)), nX))
    parY_names <- c(parY_names, paste0("X", seq_len(nX)))
  }
  if (nO > 0) {
    parY <- c(parY, rep(list(list(beta = strength_outcome)), nO))
    parY_names <- c(parY_names, paste0("O", seq_len(nO)))
  }
  if (nI > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nI))
    parY_names <- c(parY_names, paste0("I", seq_len(nI)))
  }
  if (nS > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nS))
    parY_names <- c(parY_names, paste0("S", seq_len(nS)))
  }

  names(parY) <- parY_names
  pars$cop <- list(Y = parY)

  
  # Set parameters for A
  pars$A$beta <- c(0, rep(strength_instr, nI), rep(strength_conf, nX))
  if (!binary_intervention) {
    pars$A$phi <- 1
  }
  
  # Set parameters for Y
  pars$Y$beta <- c(0, ate)
  pars$Y$phi <- 1
  
  # Generate data
  df <- rfrugalParam(n = n, formulas = forms, pars = pars, family = fam)
  p <- nX + nI + nO + nS
  
  # Flatten the A column
  df$A <- as.vector(df$A)

  # Propensity score
  if (binary_intervention) {
    if (nI + nX == 1) {
      df$propen <- plogis(c(rep(strength_instr, nI), rep(strength_conf, nX)) * df[, 1])
    } else {
      df$propen <- plogis(rowSums(c(rep(strength_instr, nI), rep(strength_conf, nX)) * df[, c(1:(nI + nX))]))
    }
    colnames(df) <- c(paste("X", 1:p, sep = ""), 'A', 'y', 'propen')
  } else {
    colnames(df) <- c(paste("X", 1:p, sep = ""), 'A', 'y')
  }
  
  # # Remove nested attributes
  # attributes(df) <- NULL
  
  return(df)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) stop("Usage: Rscript causl_binary.R <output.csv> <n> <seed> <strength_instr>")
outfile <- args[[1]]
n <- as.integer(args[[2]])
seed <- as.integer(args[[3]])
strength_instr <- as.numeric(args[[4]])
if (is.na(n) || n < 2L || is.na(seed) || is.na(strength_instr)) stop("Invalid n, seed, or strength_instr")
set.seed(seed)
dat <- data.causl(n=n, nI=5, nX=5, nO=0, nS=0, ate=2,
                  beta_cov=0, strength_instr=strength_instr,
                  strength_conf=1, strength_outcome=0,
                  binary_intervention=TRUE)
write.csv(dat[c(paste0("X", 1:10), "A", "y")], outfile, row.names=FALSE)

