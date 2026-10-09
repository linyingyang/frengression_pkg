# Joint-outcome extension of the paper's binary causl experiment.
# Run: Rscript causl_joint_outcomes.R output.csv n seed strength_instr [observational|randomized]
# The causal law is bivariate Gaussian with means (2a, -a), unit variances,
# and correlation 0.4 under each intervention do(A=a).

if (!requireNamespace("causl", quietly = TRUE)) {
  stop("Install the R package causl: remotes::install_github('rje42/causl')")
}
library(causl)

args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(4L, 5L))) {
  stop("Usage: Rscript causl_joint_outcomes.R output.csv n seed strength_instr [observational|randomized]")
}
outfile <- args[[1]]
n <- as.integer(args[[2]])
seed <- as.integer(args[[3]])
strength_instr <- as.numeric(args[[4]])
mode <- if (length(args) == 5L) args[[5]] else "observational"
if (!(mode %in% c("observational", "randomized"))) stop("Unknown mode")
if (is.na(n) || n < 2L || is.na(seed) || !is.finite(strength_instr)) {
  stop("Invalid n, seed, or strength_instr")
}

cov_names <- c(paste0("I", seq_len(5)), paste0("X", seq_len(5)))
cov_forms <- lapply(cov_names, function(v) as.formula(paste0(v, " ~ 1")))
forms <- list(cov_forms,
              as.formula(paste("A ~", paste(cov_names, collapse = " + "))),
              list(Y1 ~ A, Y2 ~ A), ~ 1)
fam <- list(rep(1, 10), 5, c(1, 1), 1)
pars <- setNames(lapply(cov_names, function(v) list(beta = 0, phi = 1)), cov_names)
pars$A <- list(beta = c(0, rep(strength_instr, 5), rep(1, 5)))
# Randomization is used solely for a separate check of the specified
# interventional moments; it is never used to train frengression.
if (mode == "randomized") pars$A$beta[] <- 0
pars$Y1 <- list(beta = c(0, 2), phi = 1)
pars$Y2 <- list(beta = c(0, -1), phi = 1)

# Gaussian pair copulas use rho = 2*plogis(beta) - 1 = tanh(beta/2).
# Preserve the original Y1 copula links: beta=1 for all five confounders.
# The Y1 innovation has coefficient (1-r^2)^(5/2); the Y2-Y1 conditional
# pair copula uses that innovation, making the marginal Y1-Y2 rho equal 0.4.
r <- 2 * plogis(1) - 1
q <- 0.40 / (1 - r^2)^(5/2)
stopifnot(q < 1)
cop_y1 <- setNames(lapply(cov_names, function(v) list(beta = 0)), cov_names)
cop_y2 <- setNames(lapply(cov_names, function(v) list(beta = 0)), cov_names)
for (v in paste0("X", seq_len(5))) cop_y1[[v]] <- list(beta = 1)
cop_y2$Y1 <- list(beta = 2 * atanh(q))
pars$cop <- list(Y1 = cop_y1, Y2 = cop_y2)

set.seed(seed)
dat <- rfrugalParam(n = n, formulas = forms, family = fam, pars = pars,
                    method = "inversion")
dat$A <- as.vector(dat$A)
write.csv(dat[c(cov_names, "A", "Y1", "Y2")], outfile, row.names = FALSE)
