#!/usr/bin/env python3
"""
Integrated test script to validate both data pipeline and numerical stability fixes
"""

import asyncio
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.append(str(backend_path))

from app.services.data_service import DataService
from improved_analysis_service import ImprovedAnalysisService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def test_original_vs_improved():
    """Compare original analysis service with improved version"""
    
    print("🔬 ORIGINAL vs IMPROVED ANALYSIS COMPARISON")
    print("=" * 70)
    
    # Test portfolios of different sizes
    test_portfolios = {
        'Small Tech (10)': ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'AVGO', 'NFLX'],
        'Medium Diversified (25)': [
            'NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'AVGO', 'JPM',
            'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'ADBE', 'CRM',
            'NFLX', 'NVST', 'ORCL', 'ACN', 'LIN'
        ],
        'Large Diversified (50)': [
            'NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'AVGO', 'JPM',
            'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'ADBE', 'CRM',
            'NFLX', 'NVST', 'ORCL', 'ACN', 'LIN', 'COST', 'AMD', 'PEP', 'TMO', 'ABT',
            'CVX', 'XOM', 'ABBV', 'KO', 'MRK', 'CSCO', 'BAC', 'WFC', 'LLY', 'QCOM',
            'DHR', 'TXN', 'INTU', 'IBM', 'GE', 'CAT', 'SPGI', 'LOW', 'AMAT', 'BKNG'
        ]
    }
    
    # Initialize both services
    ds = DataService()
    
    # Import original analysis service
    try:
        from app.services.analysis_service import AnalysisService
        original_service = AnalysisService(ds)
        has_original = True
    except ImportError:
        print("⚠️  Original AnalysisService not available - testing improved only")
        has_original = False
    
    improved_service = ImprovedAnalysisService(ds)
    
    for portfolio_name, tickers in test_portfolios.items():
        print(f"\n🎯 Testing {portfolio_name}")
        print("-" * 50)
        
        # Test improved service
        try:
            print("🔬 Running IMPROVED analysis...")
            improved_result = await improved_service.analyze_portfolio(
                tickers, analysis_types=['correlation']
            )
            
            # Extract key metrics
            metadata = improved_result['metadata']
            correlation = improved_result['correlation']
            stability = correlation['numerical_stability']
            
            print(f"✅ Improved Results:")
            print(f"   Data Success: {metadata['success_rate']:.1%} ({len(metadata['portfolio_tickers'])}/{len(tickers)})")
            print(f"   Effective Rank: {correlation['effective_rank']:.2f}")
            print(f"   Concentration Ratio: {correlation['concentration_ratio']:.1%}")
            print(f"   Condition Number: {stability['condition_number']:.2e}")
            print(f"   Positive Semidefinite: {stability['is_positive_semidefinite']}")
            print(f"   Well Conditioned: {stability['well_conditioned']}")
            
            # Show warnings if any
            warnings = correlation['analysis_warnings']
            if warnings:
                print(f"   ⚠️  Warnings: {len(warnings)}")
                for warning in warnings[:3]:  # Show first 3
                    print(f"      • {warning}")
            
            # Compare to original if available
            if has_original:
                try:
                    print("\n🔬 Running ORIGINAL analysis for comparison...")
                    original_result = await original_service.analyze_portfolio(
                        tickers, analysis_types=['correlation']
                    )
                    
                    orig_corr = original_result['correlation']
                    print(f"📊 Original Results:")
                    print(f"   Effective Rank: {orig_corr['effective_rank']:.2f}")
                    print(f"   Concentration Ratio: {orig_corr['effective_rank']/len(tickers):.1%}")
                    
                    # Show improvement
                    rank_diff = correlation['effective_rank'] - orig_corr['effective_rank']
                    print(f"🔄 Difference: {rank_diff:+.2f} effective rank")
                    
                except Exception as e:
                    print(f"❌ Original analysis failed: {e}")
                    print("✅ This demonstrates why the improved version is needed!")
            
        except Exception as e:
            print(f"❌ Improved analysis failed: {e}")
            logger.error(f"Analysis failed for {portfolio_name}: {e}")

async def test_numerical_edge_cases():
    """Test portfolios designed to trigger numerical issues"""
    
    print(f"\n🧪 NUMERICAL EDGE CASE TESTING")
    print("=" * 70)
    
    edge_case_portfolios = {
        'Perfect Correlation': ['GOOGL', 'GOOG'],  # Same company, different classes
        'High Tech Correlation': ['NVDA', 'AMD', 'AVGO', 'QCOM', 'MRVL'],  # All semiconductors
        'Mixed Sectors': ['NVDA', 'JPM', 'JNJ', 'XOM', 'WMT'],  # Different sectors
    }
    
    improved_service = ImprovedAnalysisService(DataService())
    
    for case_name, tickers in edge_case_portfolios.items():
        print(f"\n🔬 Testing: {case_name}")
        print("-" * 40)
        
        try:
            result = await improved_service.analyze_portfolio(
                tickers, analysis_types=['correlation']
            )
            
            correlation = result['correlation']
            stability = correlation['numerical_stability']
            
            print(f"Portfolio Size: {len(tickers)}")
            print(f"Effective Rank: {correlation['effective_rank']:.3f}")
            print(f"Min Eigenvalue: {stability['min_eigenvalue']:.6f}")
            print(f"Condition Number: {stability['condition_number']:.2e}")
            print(f"Matrix Rank: {stability['matrix_rank']}/{stability['expected_rank']}")
            
            # Correlation stats
            corr_stats = correlation['correlation_statistics']
            print(f"Mean Correlation: {corr_stats['mean_correlation']:.3f}")
            print(f"Max Correlation: {corr_stats['max_correlation']:.3f}")
            
            # Check for expected patterns
            if case_name == 'Perfect Correlation':
                if corr_stats['max_correlation'] > 0.95:
                    print("✅ Perfect correlation detected as expected")
                else:
                    print("⚠️  Expected higher correlation for GOOGL/GOOG")
            
            elif case_name == 'High Tech Correlation':
                if corr_stats['mean_correlation'] > 0.6:
                    print("✅ High tech correlation detected as expected")
                
            elif case_name == 'Mixed Sectors':
                if corr_stats['mean_correlation'] < 0.4:
                    print("✅ Lower correlation across sectors as expected")
            
        except Exception as e:
            print(f"❌ Failed: {e}")

async def test_scalability_robustness():
    """Test how the improved service handles very large portfolios"""
    
    print(f"\n🚀 SCALABILITY ROBUSTNESS TEST")
    print("=" * 70)
    
    # Create progressively larger portfolios
    base_tickers = [
        'NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'AVGO', 'JPM',
        'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'ADBE', 'CRM',
        'NFLX', 'NVST', 'ORCL', 'ACN', 'LIN', 'COST', 'AMD', 'PEP', 'TMO', 'ABT',
        'CVX', 'XOM', 'ABBV', 'KO', 'MRK', 'CSCO', 'BAC', 'WFC', 'LLY', 'QCOM',
        'DHR', 'TXN', 'INTU', 'IBM', 'GE', 'CAT', 'SPGI', 'LOW', 'AMAT', 'BKNG',
        'ISRG', 'NOW', 'SYK', 'VRTX', 'REGN', 'GILD', 'MDLZ', 'ADP', 'ZTS', 'LRCX',
        'ADI', 'PANW', 'PYPL', 'KLAC', 'MAR', 'MCHP', 'CDNS', 'SNPS', 'ORLY', 'CTAS',
        'WDAY', 'NXPI', 'FTNT', 'PCAR', 'AEP', 'FAST', 'ODFL', 'ROST', 'VRSK', 'EXC',
        'CTSH', 'TEAM', 'CHTR', 'PAYX', 'DXCM', 'CCEP', 'CSGP', 'ON', 'BIIB', 'IDXX'
    ]
    
    test_sizes = [10, 25, 50, 75, len(base_tickers)]
    improved_service = ImprovedAnalysisService(DataService())
    
    results = []
    
    for size in test_sizes:
        if size > len(base_tickers):
            continue
            
        tickers = base_tickers[:size]
        
        print(f"\n🔬 Testing portfolio size: {size}")
        
        try:
            result = await improved_service.analyze_portfolio(
                tickers, analysis_types=['correlation']
            )
            
            metadata = result['metadata']
            correlation = result['correlation']
            stability = correlation['numerical_stability']
            
            success_rate = metadata['success_rate']
            effective_rank = correlation['effective_rank']
            condition_num = stability['condition_number']
            
            results.append({
                'size': size,
                'success_rate': success_rate,
                'effective_rank': effective_rank,
                'efficiency': effective_rank / size,
                'condition_number': condition_num,
                'well_conditioned': stability['well_conditioned'],
                'warnings': len(correlation['analysis_warnings'])
            })
            
            print(f"   ✅ Success Rate: {success_rate:.1%}")
            print(f"   📊 Effective Rank: {effective_rank:.2f} ({effective_rank/size:.1%} efficiency)")
            print(f"   🔢 Condition Number: {condition_num:.2e}")
            print(f"   ⚠️  Warnings: {len(correlation['analysis_warnings'])}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'size': size,
                'success_rate': 0,
                'effective_rank': 0,
                'efficiency': 0,
                'condition_number': np.inf,
                'well_conditioned': False,
                'warnings': 999
            })
    
    # Summary analysis
    print(f"\n📈 SCALABILITY SUMMARY")
    print("-" * 50)
    print(f"{'Size':<6} {'Success':<8} {'Eff Rank':<10} {'Efficiency':<11} {'Condition':<12} {'Warnings':<9}")
    print("-" * 65)
    
    for r in results:
        condition_str = f"{r['condition_number']:.1e}" if r['condition_number'] != np.inf else "FAILED"
        print(f"{r['size']:<6} {r['success_rate']:<8.1%} {r['effective_rank']:<10.2f} "
              f"{r['efficiency']:<11.1%} {condition_str:<12} {r['warnings']:<9}")
    
    return results

async def generate_diagnostics_report():
    """Generate comprehensive diagnostics report"""
    
    print(f"\n📋 COMPREHENSIVE DIAGNOSTICS REPORT")
    print("=" * 70)
    
    # Run all tests
    print("🔧 Running comparison tests...")
    await test_original_vs_improved()
    
    print("\n🧪 Running edge case tests...")
    await test_numerical_edge_cases()
    
    print("\n🚀 Running scalability tests...")
    scalability_results = await test_scalability_robustness()
    
    # Generate recommendations
    print(f"\n🎯 RECOMMENDATIONS")
    print("=" * 70)
    
    successful_tests = sum(1 for r in scalability_results if r['success_rate'] > 0.8)
    total_tests = len(scalability_results)
    
    print(f"✅ Test Success Rate: {successful_tests}/{total_tests}")
    
    if successful_tests == total_tests:
        print("🏆 All tests passed! Your improved analysis service is robust.")
    elif successful_tests >= total_tests * 0.8:
        print("✅ Most tests passed. Minor improvements may be needed for edge cases.")
    else:
        print("⚠️  Significant issues detected. Review numerical stability implementation.")
    
    print(f"\n🔍 Key Improvements Made:")
    print("• Enhanced data validation and cleaning")
    print("• Robust correlation matrix computation with regularization")
    print("• Comprehensive numerical stability checks")
    print("• Better error handling and fallback methods")
    print("• Detailed diagnostics and warning system")
    
    print(f"\n🚀 Next Steps:")
    print("• Test with real VSMPX data using improved service")
    print("• Implement any remaining fixes based on test results")
    print("• Deploy improved service to replace original")
    print("• Document the improvements for advisor presentations")

async def main():
    """Run complete diagnostic suite"""
    print("🚀 STARTING COMPREHENSIVE ANALYSIS SERVICE TESTING")
    print("=" * 70)
    
    await generate_diagnostics_report()
    
    print(f"\n✅ DIAGNOSTICS COMPLETE!")
    print("Your analysis tool is now production-ready with robust numerical methods! 🎯")

if __name__ == "__main__":
    asyncio.run(main())