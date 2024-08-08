# Clear the workspace
rm(list = ls())

library(cmdstanr)
library(MASS)  # For mvrnorm function
library(Matrix)  # For nearPD function
library(matrixcalc)  # For is.positive.definite function

set.seed(123)

# Generate toy data
N <- 500  # Number of observations
K <- 7    # Number of outcomes
P <- 2    # Number of predictors including intercept
J_field <- 5  # Number of fields
J_journal <- 10  # Number of journals

year_centered <- rnorm(N)
field <- sample(1:J_field, N, replace = TRUE)
journal <- sample(1:J_journal, N, replace = TRUE)

# Generate predictors matrix
X <- model.matrix(~ year_centered)

# True parameters
beta <- matrix(rnorm(P * K, 0, 1), P, K)
alpha <- rnorm(K, 0, 1)
sigma_field <- runif(K, 0.5, 1)
sigma_journal <- runif(K, 0.5, 1)
# Use a random correlation matrix for Omega
Omega <- matrix(runif(K * K, -1, 1), K, K)
Omega <- (Omega + t(Omega)) / 2  # Symmetrize
diag(Omega) <- 1  # Set diagonals to 1

# Check and make Omega positive definite
if (!is.positive.definite(Omega)) {
  Omega <- as.matrix(nearPD(Omega)$mat)
}

# Generate random effects
u_field <- matrix(rnorm(J_field * K, 0, sigma_field), J_field, K)
u_journal <- matrix(rnorm(J_journal * K, 0, sigma_journal), J_journal, K)

# Generate outcomes with correlated residuals
eta <- matrix(0, N, K)
for (n in 1:N) {
  for (k in 1:K) {
    eta[n, k] <- alpha[k] + X[n, ] %*% beta[, k] + u_field[field[n], k] + u_journal[journal[n], k]
  }
}

# Generate binary outcomes with correlated residuals
y <- matrix(0, N, K)
for (n in 1:N) {
  residuals <- mvrnorm(1, mu = rep(0, K), Sigma = Omega)  # Correlated residuals
  eta[n, ] <- eta[n, ] + residuals
  probs <- pnorm(eta[n, ])  # Use pnorm for probit model
  y[n, ] <- rbinom(K, 1, probs)
}

# Prepare the data list for Stan
data_list <- list(
  N = N,
  K = K,
  P = P,
  y = y,
  X = X,
  J_field = J_field,
  field = field,
  J_journal = J_journal,
  journal = journal
)


library(cmdstanr)

# Define the Stan model file path
stan_model_file <- "multivariate_probit.stan"  

# Compile the model
#mod <- cmdstan_model(stan_model_file,compile=F)
#mod$format(canonicalize = list("deprecations"))

mod <- cmdstan_model(stan_model_file)

# Fit the model
fit <- mod$sample(
  data = data_list,
  iter_sampling = 1000,
  iter_warmup = 500,
  chains = 3,
  parallel_chains = 3,
  seed = 666 
)

# Print the summary of the fit
print(fit)

#Create a stanfit object
stanfit <- rstan::read_stan_csv(fit$output_files())
#Visualize output
rstan::plot(stanfit, pars="beta")
rstan::plot(stanfit, plotfun = "trace", pars = "beta", inc_warmup = TRUE)

# Load necessary libraries
library(bayesplot)
library(posterior)
library(tidyr)
library(dplyr)
library(ggplot2)

# Extract posterior samples
posterior_samples <- as_draws_matrix(fit)

# Generate a range of X values for prediction
X_new <- seq(min(year_centered), max(year_centered), length.out = 100)
X_new_df <- model.matrix(~ X_new)

# Get the dimensions
num_draws <- dim(posterior_samples)[1]
K <- length(grep("alpha", colnames(posterior_samples)))
P <- dim(X_new_df)[2]

# Initialize array for predicted probabilities
pred_probs <- array(0, dim = c(length(X_new), K, num_draws))

# Extract parameters from posterior samples and calculate predicted probabilities
for (i in 1:num_draws) {
  beta_samples <- posterior_samples[i, grep("beta", colnames(posterior_samples))]
  beta <- matrix(beta_samples, nrow = P, ncol = K)
  alpha <- posterior_samples[i, grep("alpha", colnames(posterior_samples))]
  
  for (n in 1:length(X_new)) {
    for (k in 1:K) {
      eta <- alpha[k] + X_new_df[n, ] %*% beta[, k]
      pred_probs[n, k, i] <- pnorm(eta) # pnorm is the CDF of the standard normal distribution
    }
  }
}

# Calculate the mean predicted probabilities across posterior samples
mean_pred_probs <- apply(pred_probs, c(1, 2), mean)

# Convert to a data frame for plotting
pred_probs_df <- as.data.frame(mean_pred_probs)
pred_probs_df$X_new <- X_new

# Melt the data frame for ggplot
pred_probs_melted <- pred_probs_df %>%
  pivot_longer(cols = -X_new, names_to = "Outcome", values_to = "Probability") %>%
  mutate(Outcome = factor(Outcome, levels = paste0("V", 1:K), labels = paste("Outcome", 1:K)))

# Plot the predicted probabilities
ggplot(pred_probs_melted, aes(x = X_new, y = Probability, color = Outcome)) +
  geom_line() +
  labs(title = "Predicted Probabilities of Selection", x = "X (year_centered)", y = "Probability", color = "Outcome Type")

#Quick visualization of the residual correlations

# Install and load necessary packages
library(corrplot)

# Extract posterior samples
posterior_samples <- as_draws_matrix(fit)

# Get the dimensions
num_draws <- dim(posterior_samples)[1]
K <- length(grep("alpha", colnames(posterior_samples)))

# Extract Omega correlation matrices
Omega_samples <- posterior_samples[, grep("Omega", colnames(posterior_samples))]

# Convert Omega samples to a 3D array (draws, K, K)
Omega_array <- array(0, dim = c(num_draws, K, K))
for (i in 1:num_draws) {
  Omega_array[i, , ] <- matrix(Omega_samples[i, ], nrow = K, ncol = K)
}

# Compute the average correlation matrix
avg_Omega <- apply(Omega_array, c(2, 3), mean)

# Plot the heatmap of the average correlation matrix
corrplot(avg_Omega, method = "color", type = "upper", 
         tl.col = "black", tl.srt = 45,
         title = "Average Correlation Matrix of Outcome Residuals",
         addCoef.col = "black", number.cex = 0.7)

