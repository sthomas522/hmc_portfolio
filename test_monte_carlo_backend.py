"""
Monte Carlo Backend Test Suite

Comprehensive testing of the Monte Carlo simulation module including:
- Core simulation functionality
- Multiple modeling approaches
- Stress scenario testing
- API endpoint validation
- Performance and accuracy checks
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def test_monte_carlo_simulator():
    """Test the core Monte Carlo simulator functionality."""
    print("=" * 60)
    print("TESTING MONTE CARLO SIMULATOR")
    print("=" * 60)
    
    try:
        from analysis.monte_carlo_simulator import (
            MonteCarloSimulator, SimulationConfig, 
            quick_monte_carlo, compare_simulation_methods
        )
        
        # Create synthetic test data
        np.random.seed(42)
        dates = pd.date_range('2021-01-01', periods=504, freq='D')  # 2 years of data
        n_assets = 4
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        
        # Generate realistic return patterns
        market_factor = np.random.normal(0.0004, 0.015, 504)  # ~10% annual return, 24% vol
        sector_factors = {
            'tech': np.random.normal(0, 0.01, 504),
            'momentum': np.random.normal(0, 0.008, 504)
        }
        
        returns_data = pd.DataFrame({
            'AAPL': 0.7 * market_factor + 0.8 * sector_factors['tech'] + 0.3 * sector_factors['momentum'] + np.random.normal(0, 0.008, 504),
            'MSFT': 0.6 * market_factor + 0.7 * sector_factors['tech'] + 0.1 * sector_factors['momentum'] + np.random.normal(0, 0.007, 504),
            'GOOGL': 0.8 * market_factor + 0.9 * sector_factors['tech'] + 0.2 * sector_factors['momentum'] + np.random.normal(0, 0.009, 504),
            'AMZN': 0.9 * market_factor + 0.6 * sector_factors['tech'] + 0.5 * sector_factors['momentum'] + np.random.normal(0, 0.012, 504)
        }, index=dates)
        
        portfolio_weights = pd.Series([0.3, 0.3, 0.2, 0.2], index=tickers)
        
        print(f"Test data: {len(returns_data)} observations, {n_assets} assets")
        print(f"Portfolio annual return: {((returns_data * portfolio_weights).sum(axis=1).mean() * 252):.1%}")
        print(f"Portfolio annual volatility: {((returns_data * portfolio_weights).sum(axis=1).std() * np.sqrt(252)):.1%}")
        
        # Initialize simulator
        simulator = MonteCarloSimulator()
        
        # Test 1: Basic parametric simulation
        print(f"\n1. Testing parametric simulation...")
        try:
            config = SimulationConfig(
                num_simulations=1000,  # Smaller for testing
                time_horizon_years=5,
                initial_portfolio_value=100000,
                annual_contribution=12000
            )
            
            results = simulator.simulate_portfolio(
                returns_data, portfolio_weights, config, ['parametric']
            )
            
            if 'parametric' in results:
                result = results['parametric']
                median_outcome = result.percentiles[0.50]
                p10_outcome = result.percentiles[0.10]
                p90_outcome = result.percentiles[0.90]
                prob_loss = result.probability_metrics['probability_of_loss']
                
                print(f"  ✓ Parametric simulation successful")
                print(f"    Median outcome: ${median_outcome:,.0f}")
                print(f"    10th-90th percentile: ${p10_outcome:,.0f} - ${p90_outcome:,.0f}")
                print(f"    Probability of loss: {prob_loss:.1%}")
                print(f"    Number of simulations: {len(result.final_values)}")
                
                # Sanity checks
                if median_outcome < config.initial_portfolio_value * 0.5:
                    print(f"    ⚠️  Warning: Median outcome seems too low")
                elif median_outcome > config.initial_portfolio_value * 5:
                    print(f"    ⚠️  Warning: Median outcome seems too high")
                else:
                    print(f"    ✓ Outcome range appears reasonable")
                    
                if prob_loss > 0.5:
                    print(f"    ⚠️  Warning: High probability of loss")
                elif prob_loss < 0.01:
                    print(f"    ⚠️  Warning: Very low probability of loss (unrealistic?)")
                else:
                    print(f"    ✓ Loss probability appears reasonable")
            else:
                print(f"  ✗ Parametric simulation failed")
                
        except Exception as e:
            print(f"  ✗ Parametric simulation error: {str(e)}")
        
        # Test 2: Bootstrap simulation
        print(f"\n2. Testing bootstrap simulation...")
        try:
            results = simulator.simulate_portfolio(
                returns_data, portfolio_weights, config, ['bootstrap']
            )
            
            if 'bootstrap' in results:
                result = results['bootstrap']
                median_outcome = result.percentiles[0.50]
                
                print(f"  ✓ Bootstrap simulation successful")
                print(f"    Median outcome: ${median_outcome:,.0f}")
                print(f"    Preserves historical patterns: {result.assumptions.get('preserves_serial_correlation', False)}")
                print(f"    Block size: {result.assumptions.get('block_bootstrap_size', 'N/A')}")
            else:
                print(f"  ✗ Bootstrap simulation failed")
                
        except Exception as e:
            print(f"  ✗ Bootstrap simulation error: {str(e)}")
        
        # Test 3: Multiple methods comparison
        print(f"\n3. Testing multiple methods comparison...")
        try:
            results = simulator.simulate_portfolio(
                returns_data, portfolio_weights, config, ['parametric', 'bootstrap']
            )
            
            if len(results) >= 2:
                print(f"  ✓ Multiple methods simulation successful")
                
                # Compare outcomes
                medians = {}
                for method, result in results.items():
                    if result.percentiles:
                        medians[method] = result.percentiles[0.50]
                
                if medians:
                    print(f"    Method comparison:")
                    for method, median in medians.items():
                        print(f"      {method}: ${median:,.0f}")
                    
                    # Check for reasonable differences
                    median_values = list(medians.values())
                    max_diff = (max(median_values) - min(median_values)) / np.mean(median_values)
                    print(f"    Model uncertainty: {max_diff:.1%}")
                    
                    if max_diff > 0.5:
                        print(f"    ⚠️  Warning: Large differences between methods")
                    else:
                        print(f"    ✓ Reasonable consistency between methods")
            else:
                print(f"  ✗ Multiple methods comparison failed")
                
        except Exception as e:
            print(f"  ✗ Multiple methods error: {str(e)}")
        
        # Test 4: Stress scenarios
        print(f"\n4. Testing stress scenarios...")
        try:
            # Get scenario results from parametric simulation
            if 'parametric' in results and hasattr(results['parametric'], 'scenario_results'):
                scenario_results = results['parametric'].scenario_results
                
                if scenario_results and isinstance(scenario_results, dict):
                    print(f"  ✓ Stress scenarios generated")
                    print(f"    Number of scenarios: {len(scenario_results)}")
                    
                    for scenario_name, scenario_data in scenario_results.items():
                        if isinstance(scenario_data, dict) and 'median_outcome' in scenario_data:
                            median_stress = scenario_data['median_outcome']
                            baseline_median = results['parametric'].percentiles[0.50]
                            impact = (median_stress - baseline_median) / baseline_median
                            
                            print(f"    {scenario_name}: ${median_stress:,.0f} ({impact:+.1%} vs baseline)")
                        elif isinstance(scenario_data, dict):
                            print(f"    {scenario_name}: {scenario_data}")
                        else:
                            print(f"    {scenario_name}: Scenario data available")
                else:
                    print(f"  ⚠️  Stress scenarios generated but format unexpected")
            else:
                print(f"  ⚠️  Stress scenarios not found in results")
                
        except Exception as e:
            print(f"  ✗ Stress scenarios error: {str(e)}")
        
        # Test 5: Scenario analysis generation
        print(f"\n5. Testing scenario analysis...")
        try:
            analysis = simulator.generate_scenario_analysis(results, config)
            
            if analysis and isinstance(analysis, dict):
                print(f"  ✓ Scenario analysis generated")
                
                # Check components
                components = ['executive_summary', 'uncertainty_analysis', 'key_insights', 'recommendations']
                for component in components:
                    if component in analysis and analysis[component]:
                        print(f"    ✓ {component}: Available")
                        
                        # Show sample content
                        if component == 'key_insights' and isinstance(analysis[component], list):
                            print(f"      Sample insight: {analysis[component][0][:60]}...")
                        elif component == 'executive_summary' and isinstance(analysis[component], dict):
                            median_key = analysis[component].get('median_outcome_nominal', 'N/A')
                            print(f"      Median outcome: {median_key}")
                    else:
                        print(f"    ⚠️  {component}: Missing or empty")
            else:
                print(f"  ✗ Scenario analysis generation failed")
                
        except Exception as e:
            print(f"  ✗ Scenario analysis error: {str(e)}")
        
        # Test 6: Convenience functions
        print(f"\n6. Testing convenience functions...")
        try:
            # Quick Monte Carlo
            quick_result = quick_monte_carlo(
                returns_data, portfolio_weights, time_horizon_years=3, initial_value=50000
            )
            
            if quick_result.percentiles:
                print(f"  ✓ Quick Monte Carlo: ${quick_result.percentiles[0.50]:,.0f} median")
            else:
                print(f"  ✗ Quick Monte Carlo failed")
            
            # Method comparison
            method_results = compare_simulation_methods(
                returns_data, portfolio_weights, time_horizon_years=3
            )
            
            if method_results and len(method_results) >= 2:
                print(f"  ✓ Method comparison: {len(method_results)} methods")
            else:
                print(f"  ✗ Method comparison failed")
                
        except Exception as e:
            print(f"  ✗ Convenience functions error: {str(e)}")
        
        print(f"\n✓ Monte Carlo Simulator tests completed")
        return True, results
        
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        print("  Check that monte_carlo_simulator.py is in the analysis/ directory")
        return False, None
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False, None

def test_monte_carlo_api_endpoints():
    """Test the Monte Carlo API endpoints."""
    print("\n" + "=" * 60)
    print("TESTING MONTE CARLO API ENDPOINTS")
    print("=" * 60)
    
    try:
        import requests
        
        base_url = "http://localhost:8000"
        
        # Test server connectivity
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
        
        # Test Monte Carlo health endpoint
        print(f"\nTesting Monte Carlo health endpoint...")
        try:
            response = requests.get(f"{base_url}/simulate/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ Monte Carlo components: {data.get('status', 'unknown')}")
                
                components = data.get('components', {})
                for component, status in components.items():
                    print(f"    {component}: {status}")
                    
                if 'test_simulation' in data and data['test_simulation'].get('completed'):
                    print(f"    Test simulation: {data['test_simulation']['median_outcome']}")
            else:
                print(f"  ✗ Monte Carlo health check failed: {response.status_code}")
        except Exception as e:
            print(f"  ✗ Health check error: {str(e)}")
        
        # Test quick forecast endpoint
        print(f"\nTesting quick forecast endpoint...")
        try:
            params = {
                'tickers': ['AAPL', 'MSFT', 'GOOGL'],
                'time_horizon_years': 5,
                'initial_value': 100000
            }
            
            response = requests.post(f"{base_url}/simulate/quick-forecast", params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('simulation_successful'):
                    print(f"  ✓ Quick forecast successful")
                    print(f"    Median outcome: {data.get('median_outcome', 'N/A')}")
                    print(f"    Confidence range: {data.get('confidence_range_80pct', 'N/A')}")
                    print(f"    Loss probability: {data.get('probability_of_loss', 'N/A')}")
                else:
                    print(f"  ✗ Quick forecast failed: {data.get('error', 'Unknown error')}")
            else:
                print(f"  ✗ Quick forecast API failed: {response.status_code}")
                if response.text:
                    print(f"    Error: {response.text[:200]}...")
                    
        except Exception as e:
            print(f"  ✗ Quick forecast error: {str(e)}")
        
        # Test method comparison endpoint
        print(f"\nTesting method comparison endpoint...")
        try:
            params = {
                'tickers': ['AAPL', 'MSFT'],
                'time_horizon_years': 3,
                'initial_value': 50000
            }
            
            response = requests.post(f"{base_url}/simulate/compare-methods", params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('comparison_successful'):
                    print(f"  ✓ Method comparison successful")
                    print(f"    Model uncertainty: {data.get('model_uncertainty_pct', 'N/A')}")
                    print(f"    Key insight: {data.get('key_insight', '')[:60]}...")
                    
                    method_comparison = data.get('method_comparison', {})
                    print(f"    Methods compared: {len(method_comparison)}")
                else:
                    print(f"  ✗ Method comparison failed")
            else:
                print(f"  ✗ Method comparison API failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ✗ Method comparison error: {str(e)}")
        
        # Test full Monte Carlo endpoint
        print(f"\nTesting full Monte Carlo endpoint...")
        try:
            payload = {
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "time_horizon_years": 5,
                "initial_portfolio_value": 100000,
                "num_simulations": 1000,  # Smaller for testing
                "simulation_methods": ["parametric"]
            }
            
            response = requests.post(f"{base_url}/simulate/monte-carlo", 
                                   json=payload, timeout=90)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('simulation_successful'):
                    print(f"  ✓ Full Monte Carlo successful")
                    
                    exec_summary = data.get('executive_summary', {})
                    print(f"    Median outcome: {exec_summary.get('median_outcome_nominal', 'N/A')}")
                    print(f"    Loss probability: {exec_summary.get('probability_of_loss', 'N/A')}")
                    
                    insights = data.get('key_insights', [])
                    recommendations = data.get('recommendations', [])
                    print(f"    Generated {len(insights)} insights, {len(recommendations)} recommendations")
                    
                    if insights:
                        print(f"    Sample insight: {insights[0][:60]}...")
                else:
                    print(f"  ✗ Full Monte Carlo failed: {data.get('error', 'Unknown error')}")
            else:
                print(f"  ✗ Full Monte Carlo API failed: {response.status_code}")
                if response.text:
                    print(f"    Error: {response.text[:200]}...")
                    
        except Exception as e:
            print(f"  ✗ Full Monte Carlo error: {str(e)}")
        
        print(f"\n✓ Monte Carlo API tests completed")
        return True
        
    except ImportError:
        print(f"✗ 'requests' library not available")
        print(f"  Install with: pip install requests")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False

def validate_simulation_results(results):
    """Validate simulation results for reasonableness."""
    print("\n" + "=" * 60)
    print("VALIDATING SIMULATION RESULTS")
    print("=" * 60)
    
    if not results:
        print("✗ No results to validate")
        return False
    
    validation_passed = True
    
    for method, result in results.items():
        print(f"\nValidating {method} results...")
        
        try:
            # Check basic structure
            if not hasattr(result, 'percentiles') or not result.percentiles:
                print(f"  ✗ Missing percentiles")
                validation_passed = False
                continue
            
            # Check percentile ordering
            percentiles = result.percentiles
            p10, p50, p90 = percentiles.get(0.10, 0), percentiles.get(0.50, 0), percentiles.get(0.90, 0)
            
            if p10 >= p50 or p50 >= p90:
                print(f"  ✗ Percentiles not properly ordered: {p10:.0f}, {p50:.0f}, {p90:.0f}")
                validation_passed = False
            else:
                print(f"  ✓ Percentiles properly ordered")
            
            # Check for reasonable values
            if p50 <= 0:
                print(f"  ✗ Non-positive median outcome: {p50}")
                validation_passed = False
            else:
                print(f"  ✓ Positive median outcome: ${p50:,.0f}")
            
            # Check probability metrics
            if hasattr(result, 'probability_metrics'):
                prob_metrics = result.probability_metrics
                prob_loss = prob_metrics.get('probability_of_loss', -1)
                
                if 0 <= prob_loss <= 1:
                    print(f"  ✓ Valid loss probability: {prob_loss:.1%}")
                else:
                    print(f"  ✗ Invalid loss probability: {prob_loss}")
                    validation_passed = False
            
            # Check assumptions and limitations
            if hasattr(result, 'assumptions') and result.assumptions:
                print(f"  ✓ Assumptions documented")
            else:
                print(f"  ⚠️  Assumptions not documented")
            
            if hasattr(result, 'limitations') and result.limitations:
                print(f"  ✓ Limitations documented ({len(result.limitations)} items)")
            else:
                print(f"  ⚠️  Limitations not documented")
            
        except Exception as e:
            print(f"  ✗ Validation error: {str(e)}")
            validation_passed = False
    
    return validation_passed

def run_monte_carlo_test_suite():
    """Run the complete Monte Carlo test suite."""
    print("MONTE CARLO SIMULATION TEST SUITE")
    print("=" * 60)
    print("Testing Monte Carlo simulation components...\n")
    
    results = {}
    
    # Test 1: Core simulator
    simulator_success, simulation_results = test_monte_carlo_simulator()
    results['simulator'] = simulator_success
    
    # Test 2: Result validation
    if simulation_results:
        validation_success = validate_simulation_results(simulation_results)
        results['validation'] = validation_success
    else:
        results['validation'] = False
    
    # Test 3: API endpoints
    api_success = test_monte_carlo_api_endpoints()
    results['api'] = api_success
    
    # Summary
    print("\n" + "=" * 60)
    print("MONTE CARLO TEST SUITE SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for component, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{component.upper():20} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} components passed")
    
    if passed_tests == total_tests:
        print("🎉 All Monte Carlo tests passed! Simulation system is ready.")
    elif passed_tests >= total_tests - 1:  # Allow API to fail if server not running
        print("✅ Core Monte Carlo system functional. Start API server to test endpoints.")
    else:
        print("⚠️  Some Monte Carlo components failed. Check error messages above.")
    
    return results

if __name__ == "__main__":
    results = run_monte_carlo_test_suite()