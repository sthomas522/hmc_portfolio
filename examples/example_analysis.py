"""
Example portfolio analysis using HMC Portfolio Analyzer
"""

from hmc_portfolio import RetirementPortfolioAnalyzer
import matplotlib.pyplot as plt

def main():
    """Run example portfolio analysis"""
    
    # Example diversified retirement portfolio
    tickers = [
        # Large Cap Growth
        'AAPL', 'MSFT', 'GOOGL', 'AMZN',
        
        # Large Cap Value  
        'JNJ', 'PG', 'KO', 'WMT',
        
        # Financial
        'JPM', 'BAC', 'V',
        
        # International & Bonds
        'VEA', 'VTI', 'BND', 'VXUS'
    ]
    
    # Custom weights (optional - remove for equal weight)
    weights = [
        # Growth stocks (40%)
        0.12, 0.12, 0.08, 0.08,
        
        # Value stocks (25%) 
        0.07, 0.06, 0.06, 0.06,
        
        # Financial (15%)
        0.08, 0.04, 0.03,
        
        # International & Bonds (20%)
        0.08, 0.07, 0.03, 0.02
    ]
    
    print("🔍 Analyzing retirement portfolio...")
    print(f"Holdings: {len(tickers)} securities")
    print(f"Tickers: {', '.join(tickers)}")
    
    # Create analyzer
    analyzer = RetirementPortfolioAnalyzer(
        tickers=tickers,
        weights=weights,
        start_date='2020-01-01'  # 5 years of data
    )
    
    # Fetch data
    print("\n📊 Fetching market data...")
    if not analyzer.fetch_data():
        print("❌ Failed to fetch data")
        return
    
    # Generate comprehensive report
    print("\n📋 Generating analysis report...")
    analyzer.generate_report()
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    analyzer.plot_analysis(save_plots=True)
    
    # Bayesian analysis (if Stan is available)
    print("\n🔬 Running Bayesian analysis...")
    try:
        bayesian_results = analyzer.bayesian_analysis(n_samples=1000, n_chains=2)
        if bayesian_results:
            print("✅ Bayesian analysis completed successfully")
            fit, posterior_samples = bayesian_results
            
            # Print some Bayesian insights
            print("\n🎯 Bayesian Insights:")
            print(f"Posterior samples: {len(posterior_samples)}")
            
            # Example: uncertainty in expected returns
            mu_cols = [col for col in posterior_samples.columns if 'annual_mu' in col]
            if mu_cols:
                for i, col in enumerate(mu_cols[:5]):  # First 5 stocks
                    mean_return = posterior_samples[col].mean()
                    std_return = posterior_samples[col].std()
                    print(f"{analyzer.tickers[i]}: {mean_return:.1%} ± {std_return:.1%}")
        else:
            print("⚠️  Bayesian analysis failed or Stan not available")
            
    except Exception as e:
        print(f"⚠️  Bayesian analysis error: {e}")
    
    print("\n✅ Analysis complete!")
    print("Check 'portfolio_analysis.png' for visualizations")

if __name__ == "__main__":
    main()