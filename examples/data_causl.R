## Import packages
library(glmnet)
library(zeallot) #enable %<-%
library(mvtnorm)
library(causl)
library(survivl)
library(npcausal)

logit <- function(p){
  return(log(p/(1-p)))
}


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


# an example of simulating data from survivl msm
data.longitudinl <- function(n=1000, T=10, random_seed = 42, C_coeff = 0){
  forms <- list(C ~ 1,
                Z ~ X_l1+Z_l1+C,
                X ~ Z_l0+C,
                Y ~ X_l0+X_l1+X_l2+C,
                ~ 1)
  fams <- list(1, 1, 5, 1, 1) # note outcome is Gaussian
  pars <- list(C=list(beta=0,phi=1),
              Z=list(beta=c(-1/2,1/2,1/2,0.25),phi=0.5),
              # X=list(beta=c(0,0.5,0.25),phi=1),
              X = list(beta = c(0,1/2,1/10), phi=1),
              Y=list(beta=c(0, 2, 1,0.5, C_coeff),phi=1),
              cop=list(beta=0.5))

  set.seed(random_seed)
  dat <- msm_samp(n=n, T=T, formulas = forms, family = fams, pars = pars, control=list(surv=FALSE))
  dat$C=as.numeric(dat$C)
  return(dat)
}


data.survivl <- function(n=1000, T=10, random_seed = 42,  C_coeff = 0, setting = 1){
  if(setting == 1){
    ## Setting 1
    formulas <- list(C ~ 1,
                  Z ~ X_l1 + C,
                  X ~ Z_l0 + C,
                  Y ~ X_l0 + C,
                  cop ~ 1)
    family <- list(5,1,5,3,1)
    link <- list("logit", "identity", "logit", "inverse")
    pars <- list(C = list(beta=0),
                Z = list(beta = c(-1/2,1/2,0.25), phi=0.5),
                X = list(beta = c(0,1/2,1/10)),
                Y = list(beta = c(0.3,0.2,C_coeff), phi=1),
                cop = list(beta=0.8472979))  # gives correlation 0.4
  }else if(setting == 2){
    ## Setting 2
    formulas <- list(C ~ 1,
                Z ~ Z_l1 + X_l1,
                X ~ X_l1 + Z_l0,
                Y ~ C + X_l0,
                list(Y = list(Z ~ C)))
    family <- list(3, 1, 5, 3, c(1))
    link <- list("log", "identity", "logit", "inverse")
    pars <- list(C = list(beta=0, phi=1/2),
                Z = list(beta=c(0,0.7,0.2), phi=1),
                X = list(beta=c(-0.5,0.25,0.5)), 
                # Y = list(beta=c(1,-1/10,1/5), phi=1),
                Y = list(beta=c(0.5,1/5,1/5), phi=1),
                cop = list(Y=list(Z=list(beta=c(0.5, 0.0))))
                )
  }else if(setting == 3){
  ## Setting 3
    formulas <- list(C ~ 1,
                  Z ~ X_l1 + C,
                  X ~ Z_l0 + C,
                  Y ~ X_l0 + C,
                  cop ~ 1)
    family <- list(5,1,1,3,1)
    link <- list("logit", "identity", "identity", "inverse")
    pars <- list(C = list(beta=0),
                Z = list(beta = c(-1/2,1/2,0.25), phi=0.5),
                X = list(beta = c(1.2,0.1,2),phi=0.1),
                Y = list(beta = c(0.1,0.3,0.1), phi=1),
                # Y = list(beta = c(0.05), phi=1),
                cop = list(beta=0.8472979))  # gives correlation 0.4
  }else if(setting == 4){
    formulas <- list(C ~ 1,
                Z ~ X_l1 + C,
                X ~ Z_l0 + C,
                Y ~ X_l0 + C,
                cop ~ 1)
    family <- list(5,1,5,3,1)
    link <- list("logit", "identity", "logit", "inverse")
    pars <- list(C = list(beta=0),
                Z = list(beta = c(-1/2,1/2,0.25), phi=0.5),
                X = list(beta = c(0,1/2,1/10)),
                Y = list(beta = c(0.15,0.1,C_coeff), phi=1),
                cop = list(beta=0.8472979))  # gives correlation 0.4
  
  }


  set.seed(random_seed)
  dat <- msm_samp(n=n, T=T, formulas=formulas, family=family, pars=pars, link=link,control=list(surv=TRUE))
  dat$C=as.numeric(dat$C)
  return(dat)
}

# an example of simulating data from causl
data.causl.non_linear <- function(n=1000, nI=1, nX=3, nO=2, nS=1, ate=2, beta_cov=0, strength_instr=1, strength_conf=1, strength_outcome=1){
  O_terms <- paste0("O", 1:nO)
  O_sum <- paste(O_terms, collapse = " + ")

  X_terms <- paste0("X",1:nX)
  X_sum <- paste(X_terms, collapse = " + ")
  formula_str <- paste0(
    "Y ~ A + I(10*sin(pi*", O_sum, "))",
    " + I(20*(", O_sum, ")^2)",
    " + I(((", X_sum, ")*cos(pi*(", O_sum, ")))^2)"
  )


  # Convert the string to an R formula
  po_formula <- as.formula(formula_str)

  forms <- list(list(), A ~ 1, po_formula, ~ 1)

  fam <- list(rep(1, nI + nX + nO + nS), 5, 1, 1)

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
    parY <- c(parY, rep(list(list(beta = 0)), nX))
    parY_names <- c(parY_names, paste0("X", seq_len(nX)))
  }

  if (nO > 0) {
    parY <- c(parY, rep(list(list(beta = 0)), nO))
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

  # Set parameters for Y
  pars$Y$beta <- c(0, ate, 1,1,1)
  pars$Y$phi <- 1

  # Generate data
  df <- rfrugalParam(n = n, formulas = forms, pars = pars, family = fam)
  p <- nX + nI + nO + nS

  # Propensity score
  if (nI  == 1) {
    df$propen <- plogis(c(rep(strength_instr, nI)))
  } else {
    df$propen <- plogis(rowSums(c(rep(strength_instr, nI))))
  }
  colnames(df) <- c(paste("X", 1:p, sep = ""), 'A', 'y', 'propen')
  df$A=as.numeric(df$A)
  df$y=as.numeric(df$y)
  
  # # Remove nested attributes
  # attributes(df) <- NULL
  
  return(df)
}