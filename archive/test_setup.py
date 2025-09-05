"""
Test the fixed portfolio analyzer
Save this as test_fixed_analyzer.py in your project root
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hmc_portfolio import RetirementPortfolioAnalyzer

def test_fixed_analyzer():
    """Test the fixed analyzer with multiple tickers"""
    print("🔍 Testing fixed portfolio analyzer...")
    
    # Test with multiple tickers
    tickers = ['AAPL', 'MSFT', 'SPY']
    
    analyzer = RetirementPortfolioAnalyzer(
        tickers=tickers,
        start_date='2023-01-01',
        end_date='2024-01-01'
    )
    
    # Test data fetching
    print("📊 Testing data fetch...")
    if not analyzer.fetch_data():
        print("❌ Data fetch failed")
        return False
    
    print(f"✅ Data fetched successfully!")
    print(f"   - Data shape: {analyzer.data.shape}")
    print(f"   - Returns shape: {analyzer.returns.shape}")
    print(f"   - Portfolio returns: {len(analyzer.portfolio_returns)} observations")
    print(f"   - Tickers: {analyzer.tickers}")
    
    # Test basic statistics
    print("\n📈 Testing basic statistics...")
    try:
        stats, annual_returns, annual_volatility = analyzer.basic_statistics()
        print(f"✅ Portfolio annual return: {stats['Portfolio Annual Return']}")
        print(f"✅ Portfolio volatility: {stats['Portfolio Annual Volatility']}")
        print(f"✅ Sharpe ratio: {stats['Sharpe Ratio']}")
        print(f"✅ Max drawdown: {stats['Maximum Drawdown']}")
    except Exception as e:
        print(f"❌ Basic statistics failed: {e}")
        return False
    
    # Test correlation analysis
    print("\n🔗 Testing correlation analysis...")
    try:
        corr_matrix, concentration_ratio = analyzer.correlation_analysis()
        print(f"✅ Portfolio concentration ratio: {concentration_ratio:.3f}")
        print(f"✅ Correlation matrix shape: {corr_matrix.shape}")
    except Exception as e:
        print(f"❌ Correlation analysis failed: {e}")
        return False
    
    print("\n🎉 Multi-ticker test PASSED!")
    return True

def test_single_ticker():
    """Test with single ticker"""
    print("\n" + "="*50)
    print("🔍 Testing single ticker...")
    
    try:
        analyzer = RetirementPortfolioAnalyzer(['AAPL'], start_date='2023-01-01', end_date='2024-01-01')
        
        if not analyzer.fetch_data():
            print("❌ Single ticker fetch failed")
            return False
        
        print(f"✅ Single ticker data shape: {analyzer.data.shape}")
        print(f"✅ Single ticker columns: {analyzer.data.columns.tolist()}")
        
        stats, _, _ = analyzer.basic_statistics()
        print(f"✅ Single ticker return: {stats['Portfolio Annual Return']}")
        print(f"✅ Single ticker volatility: {stats['Portfolio Annual Volatility']}")
        
        print("🎉 Single ticker test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Single ticker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_report_generation():
    """Test report generation"""
    print("\n" + "="*50)
    print("🔍 Testing report generation...")
    
    try:
        analyzer = RetirementPortfolioAnalyzer(
            ['AAPL', 'MSFT', 'JNJ'], 
            start_date='2023-01-01', 
            end_date='2024-01-01'
        )
        
        if not analyzer.fetch_data():
            print("❌ Data fetch for report failed")
            return False
        
        print("📋 Generating report...")
        analyzer.generate_report()
        print("✅ Report generation PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Portfolio Analyzer Comprehensive Test")
    print("=" * 60)
    
    tests = [
        ("Multi-ticker Analysis", test_fixed_analyzer),
        ("Single Ticker Analysis", test_single_ticker),
        ("Report Generation", test_report_generation)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} ERROR: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY:")
    print("="*60)
    
    passed = 0
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Your setup is working correctly.")
        print("\n🚀 Next steps:")
        print("   - Run: python examples/example_analysis.py")
        print("   - Install CmdStan for Bayesian analysis:")
        print("     uv add --optional bayesian cmdstanpy")
        print("     python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()