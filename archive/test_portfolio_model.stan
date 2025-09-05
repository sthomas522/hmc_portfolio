
        data {
            int<lower=1> T;  // number of time periods
            int<lower=1> N;  // number of assets
            matrix[T, N] returns;
        }
        
        parameters {
            vector[N] mu;  // expected returns
            corr_matrix[N] Omega;  // correlation matrix
            vector<lower=0>[N] sigma;  // volatilities
        }
        
        transformed parameters {
            cov_matrix[N] Sigma;
            Sigma = quad_form_diag(Omega, sigma);
        }
        
        model {
            // Priors
            mu ~ normal(0.08/252, 0.05/252);  // Reasonable prior for daily returns
            sigma ~ exponential(50);  // Prior for volatilities
            Omega ~ lkj_corr(2);  // Prior for correlation matrix
            
            // Likelihood
            for (t in 1:T) {
                returns[t] ~ multi_normal(mu, Sigma);
            }
        }
        
        generated quantities {
            vector[N] annual_mu;
            vector[N] annual_sigma;
            matrix[N, N] annual_corr;
            
            annual_mu = mu * 252;
            annual_sigma = sigma * sqrt(252);
            annual_corr = Omega;
        }
        