data {
  int<lower=1> N; // Number of observations
  int<lower=1> K; // Number of outcomes
  int<lower=1> P; // Number of predictors
  array[N, K] int y; // Outcome matrix
  matrix[N, P] X; // Predictor matrix
  int<lower=1> J_field; // Number of fields
  array[N] int field; // Field indicator
  int<lower=1> J_journal; // Number of journals
  array[N] int journal; // Journal indicator
}
parameters {
  matrix[P, K] beta; // Regression coefficients
  vector[K] alpha; // Intercepts
  matrix[J_field, K] u_field; // Random effects for fields
  matrix[J_journal, K] u_journal; // Random effects for journals
  corr_matrix[K] Omega; // Correlation matrix for residuals
  vector<lower=0>[K] sigma_field; // Std dev for field random effects
  vector<lower=0>[K] sigma_journal; // Std dev for journal random effects
  matrix[N, K] z; // Latent variables for probit
}
transformed parameters {
  matrix[N, K] eta;
  matrix[N, K] mu;
  for (n in 1 : N) {
    for (k in 1 : K) {
      eta[n, k] = alpha[k] + X[n,  : ] * beta[ : , k]
                  + u_field[field[n], k] * sigma_field[k]
                  + u_journal[journal[n], k] * sigma_journal[k];
    }
  }
  mu = eta; // Linear predictor
}
model {
  // Priors
  to_vector(beta) ~ normal(0, 1);
  alpha ~ normal(0, 1);
  sigma_field ~ normal(0, 1);
  sigma_journal ~ normal(0, 1);
  Omega ~ lkj_corr(2);
  
  // Linear predictor
  for (n in 1 : N) {
    z[n] ~ multi_normal(mu[n], Omega); // Latent variable
    for (k in 1 : K) {
      target += bernoulli_logit_lpmf(y[n, k] | z[n, k]);
    }
  }
}