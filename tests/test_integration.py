#!/usr/bin/env python
"""
Simple test with synthetic data to verify the cleaned system works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from analysis.portfolio_analyzer import PortfolioCorrelationAnalyzer

def create_synthetic_data():
    """Create synthetic portfolio returns data for testing"""
    np.random.seed(42)
    
    # Create a two-factor model
    T = 252  # 1 year of daily data
    N = 10   # 10 assets
    
    # Generate factors
    factor1 = np.random.normal(0, 0.02, T)
    factor2 = np.random.normal(0, 0.015, T)
    
    # Create asset returns with factor loadings
    returns_data = []
    asset_names = [f'Asset_{i+1}' for i in range(N)]
    
    for i in range(N):
        # Varying factor loadings
        loading1 = 0.6 + 0.3 * (i / N)
        loading2 = 0.4 - 0.2 * (i / N)
        idiosyncratic = 0.3 * np.random.normal(0, 0.01, T)
        
        asset_returns = loading1 * factor1 + loading2 * factor2 + idiosyncratic
        returns_data.append(asset_returns)
    
    # Create DataFrame
    returns_df = pd.DataFrame(
        np.column_stack(returns_data), 
        columns=asset_names,
        index=pd.date_range('2023-01-01', periods=T, freq='D')
    )
    
    # Create weights
    weights = pd.Series([0.1] * N, index=asset_names)
    
    return returns_df, weights

def test_synthetic_portfolio():
    """Test with synthetic portfolio data"""
    print("Testing Synthetic Portfolio Analysis")
    print("=" * 40)
    
    try:
        # Create synthetic data
        returns_df, weights = create_synthetic_data()
        print(f"Created synthetic data: {len(returns_df.columns)} assets, {len(returns_df)} observations")
        
        # Run analysis
        analyzer = PortfolioCorrelationAnalyzer()
        results = analyzer.analyze_portfolio_returns(returns_df, weights)
        
        if results['analysis_successful']:
            print("\n✅ Analysis successful!")
            
            # Print key metrics
            print(f"Effective Rank: {results['effective_rank']:.1f}")
            print(f"Diversification Loss: {results['diversification_loss']:.1%}")
            print(f"Largest Factor Weight: {results['largest_eigenvalue_weight']:.1%}")
            
            # MP Analysis
            if results.get('mp_fitting_successful'):
                print(f"MP Noise Fraction: {results['noise_fraction']:.1%}")
                print(f"Signal Factors: {results['num_signal_factors']}")
                print(f"MP Quality: {results.get('mp_validation', {}).get('overall_credible', 'Unknown')}")
            else:
                print("MP Analysis: Failed")
            
            # Quality metrics
            quality = results['quality_metrics']
            print(f"Analysis Quality: {quality['quality_level']} (Score: {quality['overall_score']:.2f})")
            
            # Generate reports
            print("\n" + "="*50)
            print("SUMMARY REPORT:")
            print(analyzer.generate_report(results, "summary"))
            
            print("\n" + "="*50)
            print("ADVISOR TALKING POINTS:")
            advisor_report = analyzer.generate_report(results, "advisor")
            print(advisor_report)
            
            return True
            
        else:
            print(f"❌ Analysis failed: {results.get('error')}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_modules():
    """Test individual modules work correctly"""
    print("\nTesting Individual Modules")
    print("=" * 30)
    
    try:
        # Test MP analyzer
        from analysis.mp_analysis import MarchenkoPosturAnalyzer
        mp = MarchenkoPosturAnalyzer()
        eigenvals = np.array([3.0, 1.2, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        mp_result = mp.analyze(eigenvals, N=10, T=252)
        print(f"✅ MP Analysis: {mp_result['mp_fitting_successful']}")
        
        # Test correlation functions
        from analysis.correlation_analysis import calculate_effective_rank_metrics
        eff_result = calculate_effective_rank_metrics(eigenvals, 10)
        print(f"✅ Effective Rank: {eff_result['effective_rank']:.1f}")
        
        # Test data service
        from data.data_service import DataService
        ds = DataService()
        print("✅ Data Service: Imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Module test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("SIMPLE INTEGRATION TEST")
    print("=" * 30)
    
    # Test individual modules first
    modules_ok = test_individual_modules()
    
    if modules_ok:
        # Test full integration
        integration_ok = test_synthetic_portfolio()
        
        if integration_ok:
            print("\n🎉 ALL TESTS PASSED!")
            print("Your cleaned portfolio analysis system is working correctly!")
        else:
            print("\n❌ Integration test failed")
    else:
        print("\n❌ Module tests failed")
    
    print("\n" + "=" * 50)
    print("Test complete!")