# Test script: test_factor_attribution.py
import sys
sys.path.append('src')

from analysis.portfolio_analyzer import PortfolioCorrelationAnalyzer
import pandas as pd
import numpy as np

# Create some test data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=252, freq='D')

# Create correlated returns (simulate tech portfolio)
market_factor = np.random.normal(0, 0.015, 252)
tech_factor = np.random.normal(0, 0.01, 252)
noise = np.random.normal(0, 0.008, (252, 5))

# Simulate returns with factor structure
returns_data = pd.DataFrame({
    'AAPL': 0.8 * market_factor + 0.6 * tech_factor + noise[:, 0],
    'MSFT': 0.7 * market_factor + 0.7 * tech_factor + noise[:, 1], 
    'GOOGL': 0.6 * market_factor + 0.8 * tech_factor + noise[:, 2],
    'AMZN': 0.9 * market_factor + 0.4 * tech_factor + noise[:, 3],
    'NVDA': 0.5 * market_factor + 0.9 * tech_factor + noise[:, 4]
}, index=dates)

print("Testing factor attribution with synthetic data...")

# Test the analyzer
analyzer = PortfolioCorrelationAnalyzer()
results = analyzer.analyze_portfolio_returns(
    returns_data, 
    include_factor_attribution=True
)

print(f"Analysis successful: {results.get('analysis_successful', False)}")

if results.get('factor_attribution'):
    factor_data = results['factor_attribution']
    print(f"Factor attribution successful: {factor_data.get('attribution_successful', False)}")
    print(f"Number of factors analyzed: {factor_data.get('num_factors_analyzed', 0)}")
    
    if factor_data.get('factor_interpretations'):
        print("\nFactor Interpretations:")
        for i, factor in enumerate(factor_data['factor_interpretations']):
            print(f"  Factor {i+1}: {factor['factor_name']} ({factor['factor_type']})")
            print(f"    Confidence: {factor['confidence']:.2f}")
            print(f"    Variance explained: {factor['variance_explained']:.2%}")
            print(f"    Description: {factor['description']}")
else:
    print("No factor attribution results found")
    print("Available keys:", list(results.keys()))