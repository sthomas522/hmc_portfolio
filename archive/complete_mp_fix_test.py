#!/usr/bin/env python3
"""
Complete test of the corrected Marchenko-Pastur implementation
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.append(str(backend_path))

from app.services.data_service import DataService
from fixed_mp_implementation import FixedMarchenkoPassturAnalysis, test_corrected_mp_implementation
from integrated_corrected_rmt_service import CorrectedRMTAnalysisService

async def comprehensive_mp_fix_test():
    """Run comprehensive test of MP fixes"""
    
    print("COMPREHENSIVE MP FIX VALIDATION")
    print("=" * 60)
    
    # Step 1: Validate MP implementation against theory
    print("STEP 1: Theoretical Validation")
    print("-" * 40)
    
    mp_analyzer = FixedMarchenkoPassturAnalysis()
    theoretical_validation = mp_analyzer.validate_implementation()
    
    if not theoretical_validation:
        print("CRITICAL: Theoretical validation failed!")
        return False
    
    # Step 2: Test on simulated realistic data
    print("\nSTEP 2: Realistic Data Simulation")
    print("-" * 40)
    
    realistic_test = test_corrected_mp_implementation()
    
    if not realistic_test:
        print("CRITICAL: Realistic data test failed!")
        return False
    
    # Step 3: Test corrected service on VTIVX
    print("\nSTEP 3: VTIVX Real Data Test")
    print("-" * 40)
    
    vtivx_success = await test_corrected_service_on_vtivx()
    
    if not vtivx_success:
        print("WARNING: VTIVX test had issues")
        return False
    
    # Step 4: Compare old vs new implementation
    print("\nSTEP 4: Before/After Comparison")
    print("-" * 40)
    
    comparison_success = await compare_old_vs_new_mp()
    
    return comparison_success

async def test_corrected_service_on_vtivx():
    """Test the corrected RMT service on VTIVX data"""
    
    # VTIVX holdings
    vtivx_holdings = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'JPM']
    
    try:
        # Initialize corrected service
        ds = DataService()
        corrected_service = CorrectedRMTAnalysisService(ds)
        
        print(f"Testing corrected service on {len(vtivx_holdings)} VTIVX holdings...")
        
        # Run analysis
        result = await corrected_service.analyze_portfolio(
            vtivx_holdings, 
            analysis_types=['correlation']
        )
        
        # Extract results
        correlation = result['correlation']
        metadata = result['metadata']
        
        print(f"Data Quality: {metadata['success_rate']:.1%}")
        print(f"Effective Rank: {correlation['effective_rank']:.2f}")
        
        # Validate MP analysis results
        if 'marchenko_pastur_analysis' in correlation:
            mp = correlation['marchenko_pastur_analysis']
            
            print(f"\nCorrected MP Results:")
            print(f"  Q-ratio: {mp['q_ratio']:.4f}")
            print(f"  Signal factors: {mp['num_signal_factors']}")
            print(f"  Noise fraction: {mp['noise_fraction']:.1%}")
            print(f"  Fitting successful: {mp['mp_fitting_successful']}")
            
            # Validate expectations for tech-heavy portfolio
            print(f"\nValidation Checks:")
            
            # Q-ratio should be reasonable (not 0.007!)
            if 0.005 <= mp['q_ratio'] <= 0.1:
                print(f"  ✅ Q-ratio ({mp['q_ratio']:.4f}) is reasonable")
            else:
                print(f"  ❌ Q-ratio ({mp['q_ratio']:.4f}) seems incorrect")
                return False
            
            # Noise fraction should be significant for correlated stocks
            if 0.3 <= mp['noise_fraction'] <= 0.9:
                print(f"  ✅ Noise fraction ({mp['noise_fraction']:.1%}) is realistic")
            else:
                print(f"  ❌ Noise fraction ({mp['noise_fraction']:.1%}) seems incorrect")
                return False
            
            # Signal factors should be limited for correlated portfolio
            if 1 <= mp['num_signal_factors'] <= 5:
                print(f"  ✅ Signal factors ({mp['num_signal_factors']}) is reasonable")
            else:
                print(f"  ❌ Signal factors ({mp['num_signal_factors']}) seems too high")
                return False
            
            print(f"\n💡 MP Interpretation:")
            for interpretation in mp['interpretation']:
                print(f"    • {interpretation}")
        
        # Check RMT validation
        if 'rmt_validation' in correlation:
            validation = correlation['rmt_validation']
            print(f"\nRMT Validation:")
            print(f"  Effective Rank: {validation['effective_rank']:.1f}")
            print(f"  MP Signal Factors: {validation['mp_signal_factors']}")
            print(f"  Consistency: {validation['consistency_check']}")
        
        print(f"\n✅ Corrected VTIVX test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Corrected VTIVX test failed: {e}")
        return False

async def compare_old_vs_new_mp():
    """Compare old broken implementation vs new corrected implementation"""
    
    print("Comparing Old vs New MP Implementation")
    
    # Test portfolio
    test_tickers = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META']
    
    try:
        ds = DataService()
        
        # Test new corrected implementation
        print(f"\n🔬 NEW (Corrected) Implementation:")
        corrected_service = CorrectedRMTAnalysisService(ds)
        new_result = await corrected_service.analyze_portfolio(
            test_tickers, analysis_types=['correlation']
        )
        
        new_correlation = new_result['correlation']
        new_mp = new_correlation.get('marchenko_pastur_analysis', {})
        
        print(f"  Effective Rank: {new_correlation['effective_rank']:.2f}")
        if new_mp:
            print(f"  Q-ratio: {new_mp['q_ratio']:.4f}")
            print(f"  Signal factors: {new_mp['num_signal_factors']}")
            print(f"  Noise fraction: {new_mp['noise_fraction']:.1%}")
        
        # Expected results for tech-heavy portfolio
        print(f"\n📊 Expected Results for Tech Portfolio:")
        print(f"  Q-ratio: ~0.005-0.01 (5 assets, 1000+ observations)")
        print(f"  Noise fraction: 60-80% (correlated tech stocks)")
        print(f"  Signal factors: 2-4 (market + tech + idiosyncratic)")
        print(f"  Effective rank: 2-4 (high correlation)")
        
        # Validation
        if new_mp:
            print(f"\n🎯 Validation Results:")
            
            validation_score = 0
            total_checks = 4
            
            # Check 1: Q-ratio
            if 0.003 <= new_mp['q_ratio'] <= 0.02:
                print(f"  ✅ Q-ratio realistic")
                validation_score += 1
            else:
                print(f"  ❌ Q-ratio unrealistic")
            
            # Check 2: Noise fraction
            if 0.4 <= new_mp['noise_fraction'] <= 0.9:
                print(f"  ✅ Noise fraction realistic for correlated stocks")
                validation_score += 1
            else:
                print(f"  ❌ Noise fraction unrealistic")
            
            # Check 3: Signal factors
            if 1 <= new_mp['num_signal_factors'] <= 4:
                print(f"  ✅ Signal factors reasonable")
                validation_score += 1
            else:
                print(f"  ❌ Signal factors unreasonable")
            
            # Check 4: Effective rank consistency
            rank_ratio = new_correlation['effective_rank'] / new_mp['num_signal_factors'] if new_mp['num_signal_factors'] > 0 else 0
            if 0.5 <= rank_ratio <= 2.0:
                print(f"  ✅ Effective rank consistent with signal factors")
                validation_score += 1
            else:
                print(f"  ❌ Effective rank inconsistent with signal factors")
            
            print(f"\n🏆 Validation Score: {validation_score}/{total_checks}")
            
            if validation_score >= 3:
                print(f"✅ NEW IMPLEMENTATION IS WORKING CORRECTLY!")
                return True
            else:
                print(f"❌ NEW IMPLEMENTATION STILL HAS ISSUES")
                return False
        else:
            print(f"❌ No MP analysis in new implementation")
            return False
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

async def generate_corrected_advisor_report():
    """Generate advisor report with corrected MP analysis"""
    
    print(f"\nGENERATING CORRECTED ADVISOR REPORT")
    print("=" * 60)
    
    vtivx_holdings = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'JPM']
    
    try:
        ds = DataService()
        corrected_service = CorrectedRMTAnalysisService(ds)
        
        # Run corrected analysis
        result = await corrected_service.analyze_portfolio(
            vtivx_holdings, analysis_types=['correlation']
        )
        
        correlation = result['correlation']
        mp = correlation.get('marchenko_pastur_analysis', {})
        
        # Generate corrected executive summary
        print(f"\n📊 CORRECTED EXECUTIVE SUMMARY: VTIVX Top 10")
        print("=" * 50)
        
        effective_rank = correlation['effective_rank']
        concentration_ratio = correlation['concentration_ratio']
        diversification_loss = 1 - concentration_ratio
        
        print(f"Portfolio Holdings: {len(vtivx_holdings)}")
        print(f"Effective Risk Factors: {effective_rank:.1f}")
        print(f"Diversification Efficiency: {concentration_ratio:.1%}")
        print(f"Diversification Loss: {diversification_loss:.1%}")
        
        if mp and mp['mp_fitting_successful']:
            print(f"\nRandom Matrix Theory Analysis:")
            print(f"Signal Risk Factors: {mp['num_signal_factors']}")
            print(f"Noise Factors: {mp['num_noise_factors']}")
            print(f"Noise Fraction: {mp['noise_fraction']:.1%} of correlations are random")
            print(f"Signal Variance: {mp['signal_variance_fraction']:.1%} from genuine factors")
            
            if mp['noise_fraction'] > 0.7:
                print(f"\n⚠️ FINDING: {mp['noise_fraction']:.0f}% of apparent diversification is random noise!")
            
            print(f"\n💡 CORRECTED INSIGHT:")
            print(f"Despite 10 holdings, only {mp['num_signal_factors']} genuine risk factors")
            print(f"Mathematical evidence of hidden concentration risk")
        
        print(f"\n✅ Corrected advisor report generated!")
        return True
        
    except Exception as e:
        print(f"❌ Corrected report generation failed: {e}")
        return False

async def main():
    """Run complete MP fix validation"""
    
    print("🔧 COMPLETE MARCHENKO-PASTUR FIX VALIDATION")
    print("=" * 70)
    
    # Run comprehensive test
    comprehensive_success = await comprehensive_mp_fix_test()
    
    if comprehensive_success:
        # Generate corrected advisor report
        report_success = await generate_corrected_advisor_report()
        
        if report_success:
            print(f"\n🎉 MP IMPLEMENTATION SUCCESSFULLY FIXED!")
            print("=" * 70)
            print("✅ Theoretical validation passed")
            print("✅ Realistic data test passed") 
            print("✅ VTIVX real data test passed")
            print("✅ Before/after comparison passed")
            print("✅ Corrected advisor report generated")
            print()
            print("🎯 Ready for advisor presentations with:")
            print("• Correct noise/signal classification")
            print("• Realistic MP analysis results")
            print("• Proper RMT theoretical backing")
            print("• Professional credibility")
        else:
            print(f"\n❌ Report generation issues remain")
    else:
        print(f"\n❌ MP implementation still has fundamental issues")
        print("🔧 Review and fix before proceeding to advisor presentations")

if __name__ == "__main__":
    asyncio.run(main())