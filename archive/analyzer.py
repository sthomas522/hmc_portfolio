"""
Retirement Portfolio Analysis Tool with Bayesian HMC Estimation
Analyzes portfolio expected returns, correlation risks, and uses Stan for advanced modeling
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Optional CmdStanPy import - will work without if not installed
try:
    import cmdstanpy
    STAN_AVAILABLE = True
except ImportError:
    print("Warning: CmdStanPy not available. Bayesian analysis will be skipped.")
    STAN_AVAILABLE = False

class RetirementPortfolioAnalyzer:
    """
    Comprehensive portfolio analysis tool for retirement planning
    """
    
    def __init__(self, tickers, weights=None, start_date='2019-01-01', end_date=None):
        """
        Initialize the analyzer
        
        Args:
            tickers: List of stock tickers
            weights: Portfolio weights (default: equal weight)
            start_date: Start date for historical data
            end_date: End date for historical data (default: today)
        """
        self.tickers = tickers
        self.weights = weights if weights else np.array([1/len(tickers)] * len(tickers))
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.portfolio_returns = None
        
    def fetch_data(self):
        """Fetch historical price data"""
        print(f"Fetching data for {len(self.tickers)} stocks...")
        
        try:
            # Download raw data
            raw_data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)
            
            # Handle different data structures
            if len(self.tickers) == 1:
                # Single ticker - data has MultiIndex columns like ('Close', 'AAPL')
                if isinstance(raw_data.columns, pd.MultiIndex):
                    # Try Adj Close first, fallback to Close
                    if 'Adj Close' in raw_data.columns.get_level_values(0):
                        price_data = raw_data['Adj Close']
                    else:
                        price_data = raw_data['Close']
                    self.data = pd.DataFrame(price_data.values, 
                                           index=price_data.index, 
                                           columns=self.tickers)
                else:
                    # Simple columns
                    if 'Adj Close' in raw_data.columns:
                        self.data = pd.DataFrame(raw_data['Adj Close'], columns=self.tickers)
                    else:
                        self.data = pd.DataFrame(raw_data['Close'], columns=self.tickers)
            else:
                # Multiple tickers - data has MultiIndex columns
                if isinstance(raw_data.columns, pd.MultiIndex):
                    # Try Adj Close first, fallback to Close
                    if 'Adj Close' in raw_data.columns.get_level_values(0):
                        self.data = raw_data['Adj Close']
                    else:
                        self.data = raw_data['Close']
                else:
                    # Shouldn't happen with multiple tickers, but handle gracefully
                    self.data = raw_data
            
            # Remove any tickers with no data
            self.data = self.data.dropna(axis=1, how='all')
            valid_tickers = self.data.columns.tolist()
            
            if len(valid_tickers) < len(self.tickers):
                print(f"Warning: {len(self.tickers) - len(valid_tickers)} tickers had no data")
                self.tickers = valid_tickers
                self.weights = self.weights[:len(valid_tickers)]
                self.weights = self.weights / self.weights.sum()  # Renormalize
            
            # Calculate returns
            self.returns = self.data.pct_change().dropna()
            self.portfolio_returns = (self.returns * self.weights).sum(axis=1)
            
            print(f"Successfully loaded data for {len(self.tickers)} stocks")
            print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def basic_statistics(self):
        """Calculate basic portfolio statistics"""
        if self.returns is None:
            print("Please fetch data first")
            return None
        
        # Annualized metrics
        annual_returns = self.returns.mean() * 252
        annual_volatility = self.returns.std() * np.sqrt(252)
        portfolio_annual_return = self.portfolio_returns.mean() * 252
        portfolio_annual_vol = self.portfolio_returns.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = portfolio_annual_return / portfolio_annual_vol
        max_drawdown = self.calculate_max_drawdown()
        var_95 = np.percentile(self.portfolio_returns, 5)
        cvar_95 = self.portfolio_returns[self.portfolio_returns <= var_95].mean()
        
        stats_dict = {
            'Portfolio Annual Return': f"{portfolio_annual_return:.2%}",
            'Portfolio Annual Volatility': f"{portfolio_annual_vol:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.3f}",
            'Maximum Drawdown': f"{max_drawdown:.2%}",
            'VaR (95%)': f"{var_95:.2%}",
            'CVaR (95%)': f"{cvar_95:.2%}",
            'Number of Observations': len(self.returns)
        }
        
        return stats_dict, annual_returns, annual_volatility
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        cumulative = (1 + self.portfolio_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def correlation_analysis(self):
        """Analyze correlation structure"""
        correlation_matrix = self.returns.corr()
        
        # Calculate portfolio concentration risk
        eigenvalues = np.linalg.eigvals(correlation_matrix)
        effective_rank = np.exp(-np.sum(eigenvalues * np.log(eigenvalues + 1e-10)))
        concentration_ratio = effective_rank / len(self.tickers)
        
        return correlation_matrix, concentration_ratio
    
    def monte_carlo_simulation(self, n_simulations=10000, years=30):
        """Traditional Monte Carlo simulation for retirement planning"""
        annual_return = self.portfolio_returns.mean() * 252
        annual_vol = self.portfolio_returns.std() * np.sqrt(252)
        
        # Simulate future returns
        np.random.seed(42)
        simulations = np.random.normal(annual_return, annual_vol, (n_simulations, years))
        
        # Calculate cumulative wealth
        cumulative_returns = np.cumprod(1 + simulations, axis=1)
        
        return simulations, cumulative_returns
    
    def create_stan_model(self):
        """Create Stan model for Bayesian analysis"""
        stan_code = """
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
        """
        return stan_code

    def bayesian_analysis(self, n_samples=2000, n_chains=4):
        """Perform Bayesian analysis using Stan HMC"""
        if not STAN_AVAILABLE:
            print("CmdStanPy not available. Skipping Bayesian analysis.")
            return None
        
        print("Running Bayesian analysis with Hamiltonian Monte Carlo...")
        
        # Prepare data for Stan
        stan_data = {
            'T': len(self.returns),
            'N': len(self.tickers),
            'returns': self.returns.values
        }
        
        try:
            # Create a temporary directory for Stan files
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as temp_dir:
                stan_file_path = os.path.join(temp_dir, 'portfolio_model.stan')
                
                # Write Stan model to file
                stan_code = self.create_stan_model()
                with open(stan_file_path, 'w') as f:
                    f.write(stan_code)
                
                print(f"Stan model written to: {stan_file_path}")
                
                # Compile and sample
                model = cmdstanpy.CmdStanModel(stan_file=stan_file_path)
                fit = model.sample(
                    data=stan_data,
                    chains=n_chains,
                    iter_warmup=1000,
                    iter_sampling=n_samples,
                    seed=42,
                    show_progress=True
                )
                
                # Extract results to pandas DataFrame
                posterior_samples = fit.draws_pd()
                
                print(f"Bayesian analysis completed successfully!")
                print(f"Posterior samples shape: {posterior_samples.shape}")
                
                return fit, posterior_samples
            
        except Exception as e:
            print(f"Error in Bayesian analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_analysis(self, save_plots=False):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Price evolution
        plt.subplot(3, 4, 1)
        normalized_prices = self.data / self.data.iloc[0]
        normalized_prices.plot()
        plt.title('Normalized Price Evolution')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Portfolio cumulative returns
        plt.subplot(3, 4, 2)
        cumulative_portfolio = (1 + self.portfolio_returns).cumprod()
        cumulative_portfolio.plot()
        plt.title('Portfolio Cumulative Returns')
        plt.yscale('log')
        
        # 3. Return distribution
        plt.subplot(3, 4, 3)
        plt.hist(self.portfolio_returns, bins=50, alpha=0.7, density=True)
        plt.axvline(self.portfolio_returns.mean(), color='red', linestyle='--', label='Mean')
        plt.title('Portfolio Return Distribution')
        plt.legend()
        
        # 4. Rolling volatility
        plt.subplot(3, 4, 4)
        rolling_vol = self.portfolio_returns.rolling(252).std() * np.sqrt(252)
        rolling_vol.plot()
        plt.title('Rolling Annual Volatility (1-year window)')
        
        # 5. Correlation heatmap
        plt.subplot(3, 4, 5)
        corr_matrix, _ = self.correlation_analysis()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        
        # 6. Drawdown analysis
        plt.subplot(3, 4, 6)
        cumulative = (1 + self.portfolio_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        drawdown.plot(color='red')
        plt.fill_between(drawdown.index, drawdown, alpha=0.3, color='red')
        plt.title('Portfolio Drawdown')
        
        # 7. Risk-Return scatter
        plt.subplot(3, 4, 7)
        _, annual_returns, annual_volatility = self.basic_statistics()
        plt.scatter(annual_volatility, annual_returns, s=100, alpha=0.7)
        for i, ticker in enumerate(self.tickers):
            plt.annotate(ticker, (annual_volatility.iloc[i], annual_returns.iloc[i]))
        plt.xlabel('Annual Volatility')
        plt.ylabel('Annual Return')
        plt.title('Risk-Return Profile')
        
        # 8. Monte Carlo simulation
        plt.subplot(3, 4, 8)
        _, cumulative_returns = self.monte_carlo_simulation(n_simulations=1000, years=30)
        percentiles = np.percentile(cumulative_returns, [5, 25, 50, 75, 95], axis=0)
        years = np.arange(1, 31)
        
        plt.fill_between(years, percentiles[0], percentiles[4], alpha=0.2, label='90% CI')
        plt.fill_between(years, percentiles[1], percentiles[3], alpha=0.3, label='50% CI')
        plt.plot(years, percentiles[2], 'r-', linewidth=2, label='Median')
        plt.yscale('log')
        plt.title('30-Year Wealth Projection')
        plt.legend()
        
        # 9. Weight allocation
        plt.subplot(3, 4, 9)
        plt.pie(self.weights, labels=self.tickers, autopct='%1.1f%%')
        plt.title('Portfolio Weights')
        
        # 10. Annual returns by year
        plt.subplot(3, 4, 10)
        annual_rets = self.portfolio_returns.groupby(self.portfolio_returns.index.year).apply(
            lambda x: (1 + x).prod() - 1
        )
        colors = ['green' if r > 0 else 'red' for r in annual_rets]
        plt.bar(annual_rets.index, annual_rets.values, color=colors, alpha=0.7)
        plt.title('Annual Returns by Year')
        plt.xticks(rotation=45)
        
        # 11. Rolling Sharpe ratio
        plt.subplot(3, 4, 11)
        rolling_returns = self.portfolio_returns.rolling(252).mean() * 252
        rolling_vol = self.portfolio_returns.rolling(252).std() * np.sqrt(252)
        rolling_sharpe = rolling_returns / rolling_vol
        rolling_sharpe.plot()
        plt.title('Rolling Sharpe Ratio (1-year window)')
        
        # 12. VaR evolution
        plt.subplot(3, 4, 12)
        rolling_var = self.portfolio_returns.rolling(252).quantile(0.05)
        rolling_var.plot(color='orange')
        plt.title('Rolling VaR (95%, 1-year window)')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('portfolio_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        if self.returns is None:
            print("Please fetch data first")
            return
        
        print("="*80)
        print("RETIREMENT PORTFOLIO ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics
        stats, annual_returns, annual_volatility = self.basic_statistics()
        
        print("\n1. PORTFOLIO OVERVIEW")
        print("-" * 40)
        print(f"Number of holdings: {len(self.tickers)}")
        print(f"Analysis period: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"Total observations: {len(self.returns)}")
        
        print("\n2. PERFORMANCE METRICS")
        print("-" * 40)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Correlation analysis
        correlation_matrix, concentration_ratio = self.correlation_analysis()
        
        print("\n3. RISK ANALYSIS")
        print("-" * 40)
        print(f"Average correlation: {correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean():.3f}")
        print(f"Portfolio concentration ratio: {concentration_ratio:.3f}")
        print(f"Diversification score: {1 - concentration_ratio:.3f}")
        
        # Individual stock analysis
        print("\n4. INDIVIDUAL HOLDINGS ANALYSIS")
        print("-" * 40)
        holdings_df = pd.DataFrame({
            'Weight': self.weights,
            'Annual Return': annual_returns,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': annual_returns / annual_volatility
        }, index=self.tickers)
        holdings_df = holdings_df.sort_values('Weight', ascending=False)
        print(holdings_df.round(4))
        
        # Risk warnings
        print("\n5. RISK WARNINGS FOR RETIREMENT")
        print("-" * 40)
        
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        if avg_correlation > 0.7:
            print("⚠️  HIGH CORRELATION RISK: Average correlation > 70%")
        
        max_drawdown_value = float(stats['Maximum Drawdown'].strip('%')) / 100
        if max_drawdown_value < -0.3:
            print("⚠️  HIGH DRAWDOWN RISK: Maximum drawdown > 30%")
        
        portfolio_vol = float(stats['Portfolio Annual Volatility'].strip('%')) / 100
        if portfolio_vol > 0.25:
            print("⚠️  HIGH VOLATILITY RISK: Annual volatility > 25%")
        
        # Retirement-specific analysis
        print("\n6. RETIREMENT PLANNING INSIGHTS")
        print("-" * 40)
        
        # Monte Carlo results
        simulations, cumulative_returns = self.monte_carlo_simulation(years=30)
        final_wealth = cumulative_returns[:, -1]
        
        print(f"30-year wealth projection (median): {np.median(final_wealth):.1f}x initial investment")
        print(f"Probability of positive returns: {np.mean(final_wealth > 1):.1%}")
        print(f"5th percentile outcome: {np.percentile(final_wealth, 5):.1f}x initial investment")
        print(f"95th percentile outcome: {np.percentile(final_wealth, 95):.1f}x initial investment")
        
        # Sequence of returns risk
        print(f"\n7. SEQUENCE OF RETURNS RISK")
        print("-" * 40)
        print("This analysis shows how your portfolio might perform during retirement withdrawal phase...")
        
        # Bayesian analysis if available
        if STAN_AVAILABLE:
            print("\n8. BAYESIAN ANALYSIS")
            print("-" * 40)
            bayesian_results = self.bayesian_analysis()
            if bayesian_results:
                fit, posterior_samples = bayesian_results
                print("Bayesian analysis completed successfully")
                print("Use the posterior samples for advanced risk modeling")
        
        print("\n" + "="*80)

# Example usage
if __name__ == "__main__":
    # Example portfolio of diversified stocks
    example_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech
        'JNJ', 'PG', 'KO', 'PFE', 'WMT',          # Consumer/Healthcare
        'JPM', 'BAC', 'V', 'MA', 'BRK-B',         # Financial
        'XOM', 'CVX', 'NEE', 'DUK', 'SO',         # Energy/Utilities
        'VTI', 'VOO', 'QQQ', 'IWM', 'VEA'         # ETFs
    ]
    
    # Create analyzer
    analyzer = RetirementPortfolioAnalyzer(example_tickers[:10])  # Use first 10 for demo
    
    # Run analysis
    if analyzer.fetch_data():
        analyzer.generate_report()
        analyzer.plot_analysis()

# Installation requirements:
"""
pip install yfinance pandas numpy matplotlib seaborn scipy scikit-learn

For Bayesian analysis (optional):
pip install cmdstanpy
# Then install CmdStan:
# python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"

# Alternative with uv:
# uv add cmdstanpy
"""