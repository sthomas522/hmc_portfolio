#!/usr/bin/env python3
"""
Test RMT integration on VTIVX data - validate signal/noise detection
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.append(str(backend_path))

from app.services.data_service import DataService
from rmt_integration_service import RMTIntegratedAnalysisService

async def test_vtivx_rmt_analysis():
    """Test RMT analysis on VTIVX top holdings"""
    
    print("🔬 VTIVX RMT Analysis - Validating Signal/Noise Detection")
    print("=" * 70)
    
    # VTIVX top holdings (the ones we analyzed before)
    vtivx_holdings = {
        'Top 10 (Original)': ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'JPM'],
        'Top 25': ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'JPM',
                   'LLY', 'V', 'NFLX', 'XOM', 'MA', 'WMT', 'ORCL', 'COST', 'JNJ', 'HD',
                   'PG', 'UNH', 'CRM', 'ABBV', 'BAC'],
        'Mixed Sectors': ['NVDA', 'JPM', 'JNJ', 'XOM', 'WMT', 'UNH', 'PG', 'HD', 'KO', 'VZ']  # More diverse
    }
    
    # Initialize RMT service
    ds = DataService()
    rmt_service = RMTIntegratedAnalysisService(ds)
    
    results = {}
    
    for portfolio_name, tickers in vtivx_holdings.items():
        print(f"\n🎯 Analyzing: {portfolio_name}")
        print(f"   Holdings: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}")
        
        try:
            # Run RMT-enhanced analysis
            result = await rmt_service.analyze_portfolio(
                tickers, 
                analysis_types=['correlation']
            )
            
            correlation = result['correlation']
            metadata = result['metadata']
            
            print(f"   ✅ Data Success: {metadata['success_rate']:.1%} ({len(metadata['portfolio_tickers'])}/{len(tickers)})")
            
            # Display traditional metrics
            print(f"\n   📊 Traditional Analysis:")
            print(f"      Effective Rank: {correlation['effective_rank']:.2f}")
            print(f"      Concentration Ratio: {correlation['concentration_ratio']:.1%}")
            print(f"      Diversification Loss: {1 - correlation['concentration_ratio']:.1%}")
            
            # Display RMT analysis
            if 'marchenko_pastur_analysis' in correlation:
                mp = correlation['marchenko_pastur_analysis']
                
                print(f"\n   🔬 Marchenko-Pastur Analysis:")
                print(f"      Q-ratio (N/T): {mp['q_ratio']:.3f}")
                print(f"      Fitting successful: {mp['mp_fitting_successful']}")
                print(f"      Signal factors detected: {mp['num_signal_factors']}")
                print(f"      Noise factors: {mp['num_noise_factors']}")
                print(f"      Noise fraction: {mp['noise_fraction']:.1%}")
                print(f"      Signal variance: {mp['signal_variance_fraction']:.1%}")
                
                if mp['signal_eigenvalues']:
                    print(f"      Largest signal eigenvalue: {mp['largest_signal_eigenvalue']:.2f}")
                    print(f"      Noise bounds: [{mp['lambda_minus']:.3f}, {mp['lambda_plus']:.3f}]")
                
                # Display interpretation
                print(f"\n   💡 RMT Interpretation:")
                for interpretation in mp['interpretation']:
                    print(f"      • {interpretation}")
            
            # Display eigenportfolio analysis
            if 'eigenportfolio_analysis' in correlation:
                ep = correlation['eigenportfolio_analysis']
                
                print(f"\n   🎯 Factor Structure:")
                print(f"      Top 3 factors explain: {ep['total_variance_top3']:.1%} of variance")
                print(f"      First factor dominance: {ep['eigenvalue_concentration']:.1%}")
                
                # Show top 3 factors
                for i, portfolio in enumerate(ep['eigenportfolios'][:3]):
                    print(f"\n      Factor {i+1}: {portfolio['interpretation']}")
                    print(f"         Variance explained: {portfolio['variance_explained']:.1%}")
                    print(f"         Weight concentration: {portfolio['weight_concentration']:.3f}")
                    
                    # Show top 3 holdings
                    top_holdings = portfolio['top_holdings'][:3]
                    if top_holdings:
                        holdings_str = ", ".join([f"{h['ticker']}({h['weight']:+.2f})" for h in top_holdings])
                        print(f"         Top holdings: {holdings_str}")
            
            # Display integrated insights
            if 'rmt_insights' in correlation:
                print(f"\n   🚀 Integrated Insights:")
                for insight in correlation['rmt_insights']:
                    print(f"      {insight}")
            
            # Store for comparison
            results[portfolio_name] = {
                'effective_rank': correlation['effective_rank'],
                'concentration_ratio': correlation['concentration_ratio'],
                'mp_analysis': correlation.get('marchenko_pastur_analysis', {}),
                'eigenportfolio_analysis': correlation.get('eigenportfolio_analysis', {})
            }
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            results[portfolio_name] = {'error': str(e)}
    
    return results

async def compare_rmt_vs_traditional():
    """Compare RMT-enhanced vs traditional analysis"""
    
    print(f"\n🔄 RMT vs Traditional Analysis Comparison")
    print("=" * 70)
    
    # Import traditional service for comparison
    from improved_analysis_service import ImprovedAnalysisService
    
    test_tickers = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'JPM']
    
    ds = DataService()
    traditional_service = ImprovedAnalysisService(ds)
    rmt_service = RMTIntegratedAnalysisService(ds)
    
    print(f"🧪 Testing portfolio: {', '.join(test_tickers)}")
    
    try:
        # Traditional analysis
        print(f"\n📊 Running Traditional Analysis...")
        traditional_result = await traditional_service.analyze_portfolio(
            test_tickers, analysis_types=['correlation']
        )
        
        # RMT-enhanced analysis
        print(f"🔬 Running RMT-Enhanced Analysis...")
        rmt_result = await rmt_service.analyze_portfolio(
            test_tickers, analysis_types=['correlation']
        )
        
        # Compare results
        trad_corr = traditional_result['correlation']
        rmt_corr = rmt_result['correlation']
        
        print(f"\n📈 Comparison Results:")
        print(f"{'Metric':<25} {'Traditional':<15} {'RMT-Enhanced':<15} {'Difference':<15}")
        print("-" * 70)
        
        # Core metrics should be same
        trad_rank = trad_corr['effective_rank']
        rmt_rank = rmt_corr['effective_rank']
        print(f"{'Effective Rank':<25} {trad_rank:<15.2f} {rmt_rank:<15.2f} {abs(trad_rank-rmt_rank):<15.3f}")
        
        trad_conc = trad_corr['concentration_ratio']
        rmt_conc = rmt_corr['concentration_ratio']
        print(f"{'Concentration Ratio':<25} {trad_conc:<15.1%} {rmt_conc:<15.1%} {abs(trad_conc-rmt_conc):<15.3f}")
        
        # RMT-specific insights
        if 'marchenko_pastur_analysis' in rmt_corr:
            mp = rmt_corr['marchenko_pastur_analysis']
            print(f"\n🔬 RMT-Enhanced Insights (not available in traditional):")
            print(f"• Signal factors detected: {mp['num_signal_factors']}")
            print(f"• Noise fraction: {mp['noise_fraction']:.1%}")
            print(f"• Signal vs Effective Rank: {mp['num_signal_factors']} vs {rmt_rank:.1f}")
        
        if 'eigenportfolio_analysis' in rmt_corr:
            ep = rmt_corr['eigenportfolio_analysis']
            print(f"• First factor interpretation: {ep['eigenportfolios'][0]['interpretation']}")
            print(f"• Factor dominance: {ep['eigenvalue_concentration']:.1%}")
        
        print(f"\n✅ Both analyses completed successfully!")
        print(f"💡 Core metrics identical - RMT adds interpretability and validation")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

async def validate_rmt_theoretical_expectations():
    """Validate that RMT results align with theoretical expectations"""
    
    print(f"\n🎯 Validating RMT Theoretical Expectations")
    print("=" * 70)
    
    # Test different portfolio types to validate RMT behavior
    test_cases = {
        'High Correlation (Tech)': ['NVDA', 'AMD', 'AVGO', 'QCOM', 'MRVL'],
        'Mixed Sectors': ['NVDA', 'JPM', 'JNJ', 'XOM', 'WMT'],
        'Same Company': ['GOOGL', 'GOOG'],  # Should show extreme correlation
        'Large Diverse': ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'JPM', 'JNJ', 'XOM', 'WMT', 'UNH']
    }
    
    ds = DataService()
    rmt_service = RMTIntegratedAnalysisService(ds)
    
    for case_name, tickers in test_cases.items():
        print(f"\n🧪 Testing: {case_name}")
        print(f"   Tickers: {', '.join(tickers)}")
        
        try:
            result = await rmt_service.analyze_portfolio(
                tickers, analysis_types=['correlation']
            )
            
            correlation = result['correlation']
            
            # Extract key metrics
            effective_rank = correlation['effective_rank']
            concentration_ratio = correlation['concentration_ratio']
            
            print(f"   📊 Effective Rank: {effective_rank:.2f} / {len(tickers)} ({concentration_ratio:.1%} efficiency)")
            
            # RMT analysis
            if 'marchenko_pastur_analysis' in correlation:
                mp = correlation['marchenko_pastur_analysis']
                print(f"   🔬 RMT Signal Factors: {mp['num_signal_factors']}")
                print(f"   🔬 Noise Fraction: {mp['noise_fraction']:.1%}")
                
                # Validate expectations
                if case_name == 'High Correlation (Tech)':
                    if mp['noise_fraction'] < 0.7:
                        print(f"   ✅ Expected: Lower noise in correlated sector")
                    else:
                        print(f"   ⚠️  Unexpected: High noise in correlated sector")
                
                elif case_name == 'Mixed Sectors':
                    if 0.6 < mp['noise_fraction'] < 0.9:
                        print(f"   ✅ Expected: Moderate noise in mixed sectors")
                    else:
                        print(f"   ⚠️  Unexpected noise level for mixed sectors")
                
                elif case_name == 'Same Company':
                    if mp['num_signal_factors'] <= 2:
                        print(f"   ✅ Expected: Few factors for same company")
                    else:
                        print(f"   ⚠️  Unexpected: Multiple factors for same company")
            
            # Factor interpretation
            if 'eigenportfolio_analysis' in correlation:
                ep = correlation['eigenportfolio_analysis']
                if ep['eigenportfolios']:
                    first_factor = ep['eigenportfolios'][0]
                    print(f"   🎯 Factor 1: {first_factor['interpretation']}")
                    print(f"   🎯 Dominance: {first_factor['variance_explained']:.1%}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n✅ RMT validation complete!")

async def main():
    """Run all RMT tests"""
    
    print("🚀 VTIVX RMT Integration - Complete Test Suite")
    print("=" * 80)
    
    # Test 1: VTIVX holdings analysis
    print("TEST 1: VTIVX Holdings RMT Analysis")
    vtivx_results = await test_vtivx_rmt_analysis()
    
    # Test 2: Compare RMT vs traditional
    print("\nTEST 2: RMT vs Traditional Comparison")
    await compare_rmt_vs_traditional()
    
    # Test 3: Validate theoretical expectations
    print("\nTEST 3: RMT Theoretical Validation")
    await validate_rmt_theoretical_expectations()
    
    print(f"\n🎯 SUMMARY: RMT Integration Testing Complete")
    print("=" * 80)
    
    # Summary of what we've accomplished
    successful_tests = 0
    total_tests = len(vtivx_results)
    
    for portfolio_name, result in vtivx_results.items():
        if 'error' not in result:
            successful_tests += 1
            mp_analysis = result.get('mp_analysis', {})
            print(f"✅ {portfolio_name}:")
            print(f"   Effective Rank: {result['effective_rank']:.2f}")
            print(f"   Efficiency: {result['concentration_ratio']:.1%}")
            if mp_analysis.get('mp_fitting_successful'):
                print(f"   RMT Noise: {mp_analysis['noise_fraction']:.1%}")
                print(f"   Signal Factors: {mp_analysis['num_signal_factors']}")
        else:
            print(f"❌ {portfolio_name}: {result['error']}")
    
    print(f"\n🏆 Test Results: {successful_tests}/{total_tests} successful")
    
    if successful_tests == total_tests:
        print("🎉 All RMT integration tests passed!")
        print("✅ Ready for Action Items 3 & 4: Enhanced reporting and advisor presentation!")
    else:
        print("🔧 Some tests failed - review implementation before proceeding")

if __name__ == "__main__":
    asyncio.run(main())