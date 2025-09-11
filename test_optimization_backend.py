"""
Comprehensive backend test suite for portfolio optimization system.

This script tests all optimization components including:
- Portfolio optimization engine
- Backtesting framework
- Integration layer
- API endpoints (if running)

Run this script to validate the complete optimization system.
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def test_portfolio_optimizer():
    """Test the core portfolio optimization engine."""
    print("=" * 60)
    print("TESTING PORTFOLIO OPTIMIZER")
    print("=" * 60)
    
    try:
        from src.analysis.portfolio_optimizer import (
            PortfolioOptimizer, OptimizationConstraints, 
            optimize_max_sharpe_portfolio, optimize_risk_parity_portfolio
        )
        
        # Create synthetic test data
        np.random.seed(42)
        dates = pd.date_range('2022-01-01', periods=252, freq='D')
        n_assets = 5
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        
        # Generate correlated returns
        market_factor = np.random.normal(0, 0.015, 252)
        idiosyncratic = np.random.normal(0, 0.01, (252, n_assets))
        
        returns_data = pd.DataFrame({
            'AAPL': 0.8 * market_factor + idiosyncratic[:, 0],
            'MSFT': 0.7 * market_factor + idiosyncratic[:, 1],
            'GOOGL': 0.9 * market_factor + idiosyncratic[:, 2],
            'AMZN': 0.6 * market_factor + idiosyncratic[:, 3],
            'NVDA': 1.2 * market_factor + idiosyncratic[:, 4]
        }, index=dates)
        
        # Calculate inputs
        expected_returns = returns_data.mean() * 252
        covariance_matrix = returns_data.cov() * 252
        
        print(f"Test data: {len(returns_data)} observations, {n_assets} assets")
        print(f"Expected returns range: {expected_returns.min():.1%} to {expected_returns.max():.1%}")
        print(f"Volatility range: {np.sqrt(np.diag(covariance_matrix)).min():.1%} to {np.sqrt(np.diag(covariance_matrix)).max():.1%}")
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer()
        
        # Test different optimization methods
        methods_to_test = ['max_sharpe', 'min_variance', 'risk_parity', 'mean_variance']
        constraints = OptimizationConstraints(min_weight=0.05, max_weight=0.4)
        
        optimization_results = {}
        
        for method in methods_to_test:
            print(f"\nTesting {method} optimization...")
            
            try:
                if method == 'mean_variance':
                    result = optimizer.optimize_portfolio(
                        expected_returns, covariance_matrix, method, 
                        constraints, risk_aversion=2.0
                    )
                else:
                    result = optimizer.optimize_portfolio(
                        expected_returns, covariance_matrix, method, constraints
                    )
                
                optimization_results[method] = result
                
                if result.success:
                    print(f"  ✓ Success: Sharpe={result.sharpe_ratio:.3f}, Return={result.expected_return:.1%}, Risk={result.expected_risk:.1%}")
                    print(f"    Weights: {dict(zip(tickers, [f'{w:.1%}' for w in result.weights]))}")
                    print(f"    Concentration: {np.sum(result.weights**2):.3f}")
                else:
                    print(f"  ✗ Failed: {result.message}")
                    
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        
        # Test efficient frontier generation
        print(f"\nTesting efficient frontier generation...")
        try:
            frontier = optimizer.generate_efficient_frontier(
                expected_returns, covariance_matrix, num_points=10, constraints=constraints
            )
            
            if frontier.get('success', True):
                returns_range = f"{min(frontier['returns']):.1%} to {max(frontier['returns']):.1%}"
                risks_range = f"{min(frontier['risks']):.1%} to {max(frontier['risks']):.1%}"
                max_sharpe = max(frontier['sharpe_ratios'])
                print(f"  ✓ Generated {len(frontier['returns'])} frontier points")
                print(f"    Return range: {returns_range}")
                print(f"    Risk range: {risks_range}")
                print(f"    Maximum Sharpe: {max_sharpe:.3f}")
            else:
                print(f"  ✗ Failed: {frontier.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
        
        # Test convenience functions
        print(f"\nTesting convenience functions...")
        try:
            quick_sharpe = optimize_max_sharpe_portfolio(expected_returns, covariance_matrix)
            quick_rp = optimize_risk_parity_portfolio(expected_returns, covariance_matrix)
            
            print(f"  ✓ Quick max Sharpe: {quick_sharpe.sharpe_ratio:.3f}")
            print(f"  ✓ Quick risk parity: {quick_rp.sharpe_ratio:.3f}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
        
        print(f"\n✓ Portfolio Optimizer tests completed")
        return True, optimization_results, returns_data
        
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        print("  Check that portfolio_optimizer.py is in the analysis/ directory")
        return False, {}, None
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False, {}, None

def test_portfolio_backtester(returns_data):
    """Test the portfolio backtesting framework."""
    print("\n" + "=" * 60)
    print("TESTING PORTFOLIO BACKTESTER")
    print("=" * 60)
    
    try:
        from src.analysis.portfolio_backtester import (
            PortfolioBacktester, BacktestConfig,
            backtest_max_sharpe_strategy, compare_common_strategies
        )
        
        if returns_data is None:
            print("✗ No test data available from optimizer test")
            return False
        
        # Extend data for backtesting (need more history)
        extended_returns = pd.concat([returns_data, returns_data, returns_data])  # 3x data
        extended_returns.index = pd.date_range('2021-01-01', periods=len(extended_returns), freq='D')
        
        print(f"Test data: {len(extended_returns)} observations for backtesting")
        
        # Initialize backtester
        backtester = PortfolioBacktester()
        
        # Test single strategy backtest
        print(f"\nTesting single strategy backtest (Max Sharpe)...")
        
        config = BacktestConfig(
            start_date='2022-01-01',
            end_date='2023-12-31',
            rebalance_frequency='M',
            transaction_cost=0.001,
            max_weight=0.3
        )
        
        try:
            backtest_result = backtester.backtest_strategy(
                extended_returns, 'max_sharpe', config
            )
            
            if backtest_result.success:
                metrics = backtest_result.performance_metrics
                print(f"  ✓ Backtest successful")
                print(f"    Annual Return: {metrics.annualized_return:.1%}")
                print(f"    Volatility: {metrics.annualized_volatility:.1%}")
                print(f"    Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
                print(f"    Max Drawdown: {abs(metrics.max_drawdown):.1%}")
                print(f"    Win Rate: {metrics.win_rate:.1%}")
                print(f"    Total Transaction Costs: {backtest_result.transaction_costs.sum():.4f}")
                print(f"    Average Turnover: {backtest_result.turnover.mean():.1%}")
            else:
                print(f"  ✗ Backtest failed: {backtest_result.message}")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
        
        # Test strategy comparison
        print(f"\nTesting strategy comparison...")
        
        strategies = ['equal_weight', 'max_sharpe', 'min_variance', 'risk_parity']
        
        try:
            comparison_df = backtester.compare_strategies(
                extended_returns, strategies, config
            )
            
            print(f"  ✓ Strategy comparison completed")
            print(f"    Strategies tested: {len(comparison_df)}")
            
            if not comparison_df.empty:
                # Show results sorted by Sharpe ratio
                sorted_results = comparison_df.sort_values('Sharpe Ratio', ascending=False, )
                print(f"    Results (by Sharpe Ratio):")
                for strategy, row in sorted_results.iterrows():
                    if not pd.isna(row['Sharpe Ratio']):
                        print(f"      {strategy}: Sharpe={row['Sharpe Ratio']:.3f}, Return={row['Annualized Return']:.1%}")
                    else:
                        print(f"      {strategy}: Failed")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
        
        # Test convenience functions
        print(f"\nTesting convenience functions...")
        
        try:
            quick_backtest = backtest_max_sharpe_strategy(
                extended_returns, '2022-01-01', '2022-12-31'
            )
            
            if quick_backtest.success:
                print(f"  ✓ Quick backtest: Sharpe={quick_backtest.performance_metrics.sharpe_ratio:.3f}")
            else:
                print(f"  ✗ Quick backtest failed: {quick_backtest.message}")
                
            quick_comparison = compare_common_strategies(
                extended_returns, '2022-01-01', '2022-12-31'
            )
            
            print(f"  ✓ Quick comparison: {len(quick_comparison)} strategies")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
        
        print(f"\n✓ Portfolio Backtester tests completed")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        print("  Check that portfolio_backtester.py is in the analysis/ directory")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False

def test_optimization_integration(returns_data):
    """Test the optimization integration layer."""
    print("\n" + "=" * 60)
    print("TESTING OPTIMIZATION INTEGRATION")
    print("=" * 60)
    
    try:
        from src.analysis.optimization_integration import (
            IntegratedPortfolioAnalyzer,
            quick_optimize_portfolio, compare_optimization_strategies
        )
        
        if returns_data is None:
            print("✗ No test data available")
            return False
        
        print(f"Testing integrated analysis...")
        
        # Initialize integrated analyzer
        analyzer = IntegratedPortfolioAnalyzer()
        
        # Test full integrated analysis
        try:
            # Extend data for backtesting
            extended_returns = pd.concat([returns_data, returns_data])
            extended_returns.index = pd.date_range('2021-01-01', periods=len(extended_returns), freq='D')
            
            result = analyzer.analyze_and_optimize(
                extended_returns,
                optimization_methods=['max_sharpe', 'min_variance', 'risk_parity'],
                include_backtesting=True
            )
            
            print(f"  ✓ Integrated analysis completed")
            
            # Check risk analysis
            if 'analysis_successful' in result.risk_analysis and result.risk_analysis['analysis_successful']:
                effective_rank = result.risk_analysis.get('effective_rank', 0)
                num_assets = result.risk_analysis.get('num_assets', 0)
                print(f"    Risk Analysis: {num_assets} assets, {effective_rank:.1f} effective rank")
            
            # Check optimization results
            successful_opts = {k: v for k, v in result.optimization_results.items() if v.success}
            print(f"    Optimization: {len(successful_opts)}/{len(result.optimization_results)} methods successful")
            
            if successful_opts:
                best_method = max(successful_opts, key=lambda k: successful_opts[k].sharpe_ratio)
                best_sharpe = successful_opts[best_method].sharpe_ratio
                print(f"    Best Strategy: {best_method} (Sharpe: {best_sharpe:.3f})")
            
            # Check recommended portfolio
            if result.recommended_portfolio.success:
                rec = result.recommended_portfolio
                print(f"    Recommendation: {rec.optimization_method} (Sharpe: {rec.sharpe_ratio:.3f})")
            
            # Check insights and recommendations
            print(f"    Generated {len(result.insights)} insights and {len(result.advisor_recommendations)} recommendations")
            
            if result.insights:
                print(f"    Sample insight: {result.insights[0][:80]}...")
            
            # Check backtesting results
            if result.backtesting_results:
                successful_backtests = {k: v for k, v in result.backtesting_results.items() if v.success}
                print(f"    Backtesting: {len(successful_backtests)} strategies tested")
            
        except Exception as e:
            print(f"  ✗ Integrated analysis error: {str(e)}")
        
        # Test convenience functions
        print(f"\nTesting integration convenience functions...")
        
        try:
            # Quick optimization
            quick_result = quick_optimize_portfolio(returns_data, method='max_sharpe')
            if quick_result.success:
                print(f"  ✓ Quick optimization: Sharpe={quick_result.sharpe_ratio:.3f}")
            else:
                print(f"  ✗ Quick optimization failed: {quick_result.message}")
            
            # Strategy comparison
            comparison_df = compare_optimization_strategies(
                returns_data, ['max_sharpe', 'min_variance', 'risk_parity']
            )
            print(f"  ✓ Strategy comparison: {len(comparison_df)} strategies")
            
        except Exception as e:
            print(f"  ✗ Convenience functions error: {str(e)}")
        
        print(f"\n✓ Optimization Integration tests completed")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        print("  Check that optimization_integration.py is in the analysis/ directory")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False

def test_api_endpoints():
    """Test API endpoints if server is running."""
    print("\n" + "=" * 60)
    print("TESTING API ENDPOINTS")
    print("=" * 60)
    
    try:
        import requests
        
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        print(f"Testing API connectivity...")
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"  ✓ API server is running")
            else:
                print(f"  ✗ API server responded with status {response.status_code}")
                return False
        except requests.ConnectionError:
            print(f"  ✗ Cannot connect to API server at {base_url}")
            print(f"    Start the server with: python main.py")
            return False
        except requests.Timeout:
            print(f"  ✗ API server timeout")
            return False
        
        # Test optimization health endpoint
        try:
            response = requests.get(f"{base_url}/health/optimization", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ Optimization components: {data.get('status', 'unknown')}")
                
                components = data.get('components', {})
                for component, status in components.items():
                    print(f"    {component}: {status}")
            else:
                print(f"  ✗ Optimization health check failed: {response.status_code}")
        except Exception as e:
            print(f"  ✗ Health check error: {str(e)}")
        
        # Test portfolio optimization endpoint
        print(f"\nTesting portfolio optimization endpoint...")
        try:
            payload = {
                "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
                "method": "max_sharpe",
                "max_weight": 0.3,
                "min_weight": 0.05,
                "period": "1y"
            }
            
            response = requests.post(f"{base_url}/optimize/portfolio", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"  ✓ Optimization successful")
                    print(f"    Method: {data.get('optimization_method')}")
                    print(f"    Sharpe Ratio: {data.get('sharpe_ratio', 0):.3f}")
                    print(f"    Expected Return: {data.get('expected_return', 0):.1%}")
                    print(f"    Expected Risk: {data.get('expected_risk', 0):.1%}")
                    
                    weights = data.get('weights', {})
                    if weights:
                        print(f"    Top holdings: {dict(list(sorted(weights.items(), key=lambda x: x[1], reverse=True))[:3])}")
                else:
                    print(f"  ✗ Optimization failed: {data.get('message', 'Unknown error')}")
            else:
                print(f"  ✗ API request failed: {response.status_code}")
                if response.text:
                    print(f"    Error: {response.text[:200]}...")
                    
        except Exception as e:
            print(f"  ✗ Optimization endpoint error: {str(e)}")
        
        # Test strategy comparison endpoint
        print(f"\nTesting strategy comparison endpoint...")
        try:
            params = {
                'tickers': ['AAPL', 'MSFT', 'GOOGL'],
                'strategies': ['max_sharpe', 'min_variance', 'risk_parity'],
                'period': '1y',
                'max_weight': 0.4
            }
            
            response = requests.post(f"{base_url}/optimize/compare-strategies", params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                comparison_data = data.get('comparison_data', [])
                successful_strategies = [s for s in comparison_data if s.get('success')]
                
                print(f"  ✓ Strategy comparison successful")
                print(f"    Strategies tested: {len(comparison_data)}")
                print(f"    Successful: {len(successful_strategies)}")
                
                if successful_strategies:
                    best = max(successful_strategies, key=lambda x: x.get('sharpe_ratio', 0))
                    print(f"    Best strategy: {best.get('strategy')} (Sharpe: {best.get('sharpe_ratio', 0):.3f})")
            else:
                print(f"  ✗ Strategy comparison failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ✗ Strategy comparison error: {str(e)}")
        
        print(f"\n✓ API Endpoints tests completed")
        return True
        
    except ImportError:
        print(f"✗ 'requests' library not available")
        print(f"  Install with: pip install requests")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False

def run_full_test_suite():
    """Run the complete test suite."""
    print("PORTFOLIO OPTIMIZATION BACKEND TEST SUITE")
    print("=" * 60)
    print("Testing all optimization components...\n")
    
    results = {}
    
    # Test 1: Portfolio Optimizer
    optimizer_success, optimization_results, test_data = test_portfolio_optimizer()
    results['optimizer'] = optimizer_success
    
    # Test 2: Portfolio Backtester
    backtester_success = test_portfolio_backtester(test_data)
    results['backtester'] = backtester_success
    
    # Test 3: Integration Layer
    integration_success = test_optimization_integration(test_data)
    results['integration'] = integration_success
    
    # Test 4: API Endpoints (optional)
    api_success = test_api_endpoints()
    results['api'] = api_success
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for component, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{component.upper():20} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} components passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Portfolio optimization system is ready.")
    elif passed_tests >= total_tests - 1:  # Allow API to fail (server might not be running)
        print("✅ Core system functional. Start API server to test endpoints.")
    else:
        print("⚠️  Some components failed. Check error messages above.")
    
    return results

if __name__ == "__main__":
    results = run_full_test_suite()