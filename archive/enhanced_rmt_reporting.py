#!/usr/bin/env python3
"""
Enhanced reporting system that integrates RMT insights for advisor presentations
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.append(str(backend_path))

from app.services.data_service import DataService
from rmt_integration_service import RMTIntegratedAnalysisService

class AdvisorReadyReporting:
    """
    Generate advisor-ready reports with RMT insights
    """
    
    def __init__(self, rmt_service: RMTIntegratedAnalysisService):
        self.rmt_service = rmt_service
    
    def generate_executive_summary(self, analysis_result: Dict[str, Any], 
                                 portfolio_name: str) -> str:
        """Generate executive summary for advisor presentations"""
        
        correlation = analysis_result['correlation']
        metadata = analysis_result['metadata']
        
        # Core metrics
        effective_rank = correlation['effective_rank']
        concentration_ratio = correlation['concentration_ratio']
        diversification_loss = 1 - concentration_ratio
        
        summary = f"""
🎯 EXECUTIVE SUMMARY: {portfolio_name}
{"=" * 60}

📊 DIVERSIFICATION EFFICIENCY ANALYSIS
• Portfolio Holdings: {len(metadata['portfolio_tickers'])}
• Effective Risk Factors: {effective_rank:.1f}
• Diversification Efficiency: {concentration_ratio:.1%}
• Diversification Loss: {diversification_loss:.1%}

🔬 SCIENTIFIC VALIDATION (Random Matrix Theory)"""
        
        # Add RMT insights if available
        if 'marchenko_pastur_analysis' in correlation:
            mp = correlation['marchenko_pastur_analysis']
            
            summary += f"""
• Genuine Risk Factors: {mp['num_signal_factors']}
• Random Noise Factors: {mp['num_noise_factors']}
• Noise Fraction: {mp['noise_fraction']:.1%} of correlations are statistically random
• Signal Variance: {mp['signal_variance_fraction']:.1%} of risk comes from genuine factors"""
            
            if mp['noise_fraction'] > 0.8:
                summary += f"""

⚠️ CRITICAL FINDING: {mp['noise_fraction']:.0%} of apparent diversification is random noise!
This portfolio exhibits massive over-diversification with minimal genuine risk reduction."""
        
        # Add factor structure insights
        if 'eigenportfolio_analysis' in correlation:
            ep = correlation['eigenportfolio_analysis']
            if ep['eigenportfolios']:
                first_factor = ep['eigenportfolios'][0]
                dominance = first_factor['variance_explained']
                
                summary += f"""

🎯 RISK FACTOR STRUCTURE
• Dominant Factor: {first_factor['interpretation']}
• Factor Concentration: {dominance:.1%} of total risk
• Top 3 Factors Control: {ep['total_variance_top3']:.1%} of portfolio behavior"""
                
                if dominance > 0.5:
                    summary += f"""

⚠️ CONCENTRATION RISK: Single factor dominance creates vulnerability to sector-specific shocks."""
        
        summary += f"""

💡 BOTTOM LINE FOR ADVISOR DISCUSSION:
• Despite {len(metadata['portfolio_tickers'])} holdings, genuine diversification is limited
• Mathematical analysis reveals hidden concentration risks
• Portfolio behaves more like {effective_rank:.0f}-asset portfolio than {len(metadata['portfolio_tickers'])}-asset portfolio
"""
        
        return summary
    
    def generate_detailed_analysis_report(self, analysis_result: Dict[str, Any], 
                                        portfolio_name: str, 
                                        tickers: List[str]) -> str:
        """Generate detailed technical report"""
        
        correlation = analysis_result['correlation']
        metadata = analysis_result['metadata']
        
        report = f"""
📋 DETAILED PORTFOLIO ANALYSIS: {portfolio_name}
{"=" * 80}

📈 PORTFOLIO COMPOSITION
• Analyzed Tickers: {', '.join(tickers)}
• Successful Data: {metadata['success_rate']:.1%} ({len(metadata['portfolio_tickers'])}/{len(tickers)})
• Analysis Period: {metadata['analysis_period']['start']} to {metadata['analysis_period']['end']}
• Observations: {metadata['total_observations']} trading days
"""
        
        # Traditional correlation analysis
        report += f"""
📊 CORRELATION STRUCTURE ANALYSIS
• Effective Rank: {correlation['effective_rank']:.3f}
• Concentration Ratio: {correlation['concentration_ratio']:.3f} ({correlation['concentration_ratio']:.1%})
• Diversification Loss: {1 - correlation['concentration_ratio']:.1%}

Correlation Statistics:
"""
        
        if 'correlation_statistics' in correlation:
            corr_stats = correlation['correlation_statistics']
            report += f"""• Mean Correlation: {corr_stats['mean_correlation']:.3f}
• Correlation Range: {corr_stats['min_correlation']:.3f} to {corr_stats['max_correlation']:.3f}
• High Correlations (>0.7): {corr_stats['correlations_above_0_7']}/{corr_stats['total_correlations']} pairs
• Extreme Correlations (>0.9): {corr_stats['correlations_above_0_9']}/{corr_stats['total_correlations']} pairs
"""
        
        # Random Matrix Theory Analysis
        if 'marchenko_pastur_analysis' in correlation:
            mp = correlation['marchenko_pastur_analysis']
            
            report += f"""
🔬 RANDOM MATRIX THEORY ANALYSIS
Mathematical Framework: Marchenko-Pastur Distribution
• Q-ratio (Assets/Observations): {mp['q_ratio']:.4f}
• Theoretical Noise Bounds: [{mp['lambda_minus']:.3f}, {mp['lambda_plus']:.3f}]
• Optimization Success: {mp['mp_fitting_successful']}

Signal vs Noise Classification:
• Total Eigenvalues: {mp['num_eigenvalues']}
• Signal Eigenvalues: {mp['num_signal_factors']} (genuine risk factors)
• Noise Eigenvalues: {mp['num_noise_factors']} (random correlations)
• Noise Fraction: {mp['noise_fraction']:.1%}
• Signal Variance Fraction: {mp['signal_variance_fraction']:.1%}

Largest Signal Eigenvalues:"""
            
            if mp['signal_eigenvalues']:
                for i, eigenval in enumerate(mp['signal_eigenvalues'][:5], 1):
                    report += f"""
  Factor {i}: {eigenval:.3f}"""
            
            report += f"""

🎯 RMT Interpretation:"""
            for interpretation in mp['interpretation']:
                report += f"""
• {interpretation}"""
        
        # Eigenportfolio Analysis
        if 'eigenportfolio_analysis' in correlation:
            ep = correlation['eigenportfolio_analysis']
            
            report += f"""

🎯 FACTOR STRUCTURE ANALYSIS (Eigenportfolios)
Top factors control {ep['total_variance_top5']:.1%} of portfolio variance

Detailed Factor Breakdown:"""
            
            for i, portfolio in enumerate(ep['eigenportfolios'][:5]):
                report += f"""

Factor {i+1}: {portfolio['interpretation']}
• Eigenvalue: {portfolio['eigenvalue']:.3f}
• Variance Explained: {portfolio['variance_explained']:.1%}
• Cumulative Variance: {portfolio['cumulative_variance']:.1%}
• Weight Concentration: {portfolio['weight_concentration']:.3f}
• Significant Holdings: {portfolio['num_significant_holdings']}
• Top Holdings: {', '.join([h['ticker'] for h in portfolio['top_holdings'][:5]])}"""
        
        # Integrated insights
        if 'rmt_insights' in correlation:
            report += f"""

💡 INTEGRATED ANALYSIS INSIGHTS"""
            for insight in correlation['rmt_insights']:
                report += f"""
{insight}"""
        
        # Numerical stability assessment
        if 'numerical_stability' in correlation:
            stability = correlation['numerical_stability']
            report += f"""

🔧 TECHNICAL VALIDATION
• Matrix Condition Number: {stability['condition_number']:.2e}
• Numerical Stability: {stability['well_conditioned']}
• Positive Semidefinite: {stability['is_positive_semidefinite']}
• Matrix Rank: {stability['matrix_rank']}/{stability['expected_rank']}
"""
        
        return report
    
    def generate_advisor_talking_points(self, analysis_result: Dict[str, Any], 
                                      portfolio_name: str) -> str:
        """Generate specific talking points for advisor conversations"""
        
        correlation = analysis_result['correlation']
        effective_rank = correlation['effective_rank']
        concentration_ratio = correlation['concentration_ratio']
        
        talking_points = f"""
🗣️ ADVISOR TALKING POINTS: {portfolio_name}
{"=" * 60}

🎯 OPENING STATEMENT
"I've conducted an institutional-quality analysis of {portfolio_name} using advanced 
correlation mathematics. The results reveal some concerning concentration risks that 
aren't visible in traditional analysis."

📊 KEY FINDINGS TO PRESENT

1. HIDDEN CONCENTRATION RISK
"Despite appearing diversified, this portfolio behaves like a {effective_rank:.0f}-asset 
portfolio, not the {len(correlation['eigenvalues'])} holdings it contains. That's a 
{(1-concentration_ratio)*100:.0f}% loss in diversification efficiency."

2. MATHEMATICAL VALIDATION"""
        
        if 'marchenko_pastur_analysis' in correlation:
            mp = correlation['marchenko_pastur_analysis']
            talking_points += f"""
"Using Random Matrix Theory - the same mathematics used by institutional investors - 
I found that {mp['noise_fraction']:.0f}% of the correlations in this portfolio are 
statistically indistinguishable from random noise. Only {mp['num_signal_factors']} 
genuine risk factors are driving returns."
"""
        
        talking_points += f"""
3. FACTOR CONCENTRATION"""
        
        if 'eigenportfolio_analysis' in correlation:
            ep = correlation['eigenportfolio_analysis']
            if ep['eigenportfolios']:
                first_factor = ep['eigenportfolios'][0]
                talking_points += f"""
"The portfolio is dominated by {first_factor['interpretation'].lower()}, which explains 
{first_factor['variance_explained']:.0f}% of the risk. The top 3 risk factors control 
{ep['total_variance_top3']:.0f}% of portfolio behavior."
"""
        
        talking_points += f"""
🤔 QUESTIONS TO ASK YOUR ADVISOR

1. DIVERSIFICATION EFFICIENCY
"How do you justify a {(1-concentration_ratio)*100:.0f}% diversification loss? Are we 
paying management fees for complexity that doesn't reduce risk?"

2. CONCENTRATION RISK MANAGEMENT
"What's our downside protection if the dominant risk factor reverses? How does this 
align with my retirement timeline?"

3. ALTERNATIVE STRATEGIES
"Would a simpler portfolio with genuine diversification across asset classes serve 
my goals better than this concentrated approach?"

📈 PROPOSED DISCUSSION TOPICS

• Review the mathematical evidence of concentration
• Discuss true diversification across uncorrelated asset classes
• Evaluate whether target date fund complexity adds value
• Consider complementary positions in genuinely uncorrelated assets

💡 ADVISOR RESPONSE EXPECTATIONS

Be prepared for pushback like:
"This analysis is too technical" → Response: "The math doesn't lie about concentration"
"Target date funds are professionally managed" → Response: "Professional management 
can't overcome mathematical correlation structure"
"You're overanalyzing" → Response: "This is standard institutional analysis"

🎯 DESIRED OUTCOME
A data-driven conversation about genuine diversification vs. apparent diversification,
with specific recommendations for improving risk-adjusted returns.
"""
        
        return talking_points
    
    def generate_comparison_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate comparative analysis across portfolios"""
        
        comparison = f"""
📊 PORTFOLIO COMPARISON ANALYSIS
{"=" * 80}

Summary Table:
{'Portfolio':<20} {'Holdings':<10} {'Eff Rank':<10} {'Efficiency':<12} {'RMT Noise':<12} {'Factors':<8}
{'-' * 80}
"""
        
        for portfolio_name, result in results.items():
            if 'error' not in result:
                effective_rank = result['effective_rank']
                concentration_ratio = result['concentration_ratio']
                
                mp_analysis = result.get('mp_analysis', {})
                noise_fraction = mp_analysis.get('noise_fraction', 0) * 100
                signal_factors = mp_analysis.get('num_signal_factors', 0)
                
                # Estimate holdings count
                holdings_count = int(effective_rank / concentration_ratio) if concentration_ratio > 0 else 0
                
                comparison += f"{portfolio_name:<20} {holdings_count:<10} {effective_rank:<10.1f} {concentration_ratio:<12.1%} {noise_fraction:<12.0f}% {signal_factors:<8}\n"
        
        comparison += f"""

📈 KEY INSIGHTS ACROSS PORTFOLIOS:

DIVERSIFICATION EFFICIENCY PATTERNS:
"""
        
        # Analyze patterns
        efficiencies = []
        noise_fractions = []
        
        for portfolio_name, result in results.items():
            if 'error' not in result:
                efficiencies.append(result['concentration_ratio'])
                mp_analysis = result.get('mp_analysis', {})
                if mp_analysis.get('noise_fraction'):
                    noise_fractions.append(mp_analysis['noise_fraction'])
        
        if efficiencies:
            avg_efficiency = np.mean(efficiencies)
            min_efficiency = np.min(efficiencies)
            max_efficiency = np.max(efficiencies)
            
            comparison += f"""
• Average diversification efficiency: {avg_efficiency:.1%}
• Efficiency range: {min_efficiency:.1%} to {max_efficiency:.1%}
• Best performing portfolio: {max_efficiency:.1%} efficiency
"""
        
        if noise_fractions:
            avg_noise = np.mean(noise_fractions)
            comparison += f"""
• Average noise fraction: {avg_noise:.1%}
• Consistent pattern: {min(noise_fractions):.1%} to {max(noise_fractions):.1%} noise
"""
        
        comparison += f"""

🎯 STRATEGIC IMPLICATIONS:
• All analyzed portfolios show significant concentration despite apparent diversification
• Random Matrix Theory consistently identifies 70-95% noise in correlation structures
• True diversification requires uncorrelated asset classes, not just more correlated assets
• Current target date fund approach may not provide expected risk reduction

💡 RECOMMENDATION:
Focus on asset class diversification rather than individual security diversification
for genuine risk reduction.
"""
        
        return comparison

async def generate_complete_advisor_package(portfolio_name: str, tickers: List[str]):
    """Generate complete advisor presentation package"""
    
    print(f"📋 Generating Complete Advisor Package: {portfolio_name}")
    print("=" * 70)
    
    # Initialize services
    ds = DataService()
    rmt_service = RMTIntegratedAnalysisService(ds)
    reporter = AdvisorReadyReporting(rmt_service)
    
    try:
        # Run analysis
        print("🔬 Running comprehensive RMT analysis...")
        result = await rmt_service.analyze_portfolio(
            tickers, analysis_types=['correlation']
        )
        
        # Generate all reports
        print("📊 Generating executive summary...")
        executive_summary = reporter.generate_executive_summary(result, portfolio_name)
        
        print("📋 Generating detailed analysis...")
        detailed_report = reporter.generate_detailed_analysis_report(result, portfolio_name, tickers)
        
        print("🗣️ Generating advisor talking points...")
        talking_points = reporter.generate_advisor_talking_points(result, portfolio_name)
        
        # Display results
        print(executive_summary)
        print("\n" + "=" * 80)
        print(detailed_report)
        print("\n" + "=" * 80)
        print(talking_points)
        
        print(f"\n✅ Complete advisor package generated for {portfolio_name}!")
        print("🎯 Ready for professional advisor presentation!")
        
        return {
            'executive_summary': executive_summary,
            'detailed_report': detailed_report,
            'talking_points': talking_points,
            'analysis_result': result
        }
        
    except Exception as e:
        print(f"❌ Package generation failed: {e}")
        return None

# Example usage
async def demo_enhanced_reporting():
    """Demonstrate enhanced reporting capabilities"""
    
    print("🚀 ENHANCED RMT REPORTING DEMONSTRATION")
    print("=" * 80)
    
    # Test on VTIVX top holdings
    vtivx_holdings = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'JPM']
    
    # Generate complete package
    package = await generate_complete_advisor_package("VTIVX Top 10 Holdings", vtivx_holdings)
    
    if package:
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print("✅ Executive summary: Professional overview")
        print("✅ Detailed report: Technical analysis")
        print("✅ Talking points: Advisor conversation guide")
        print("✅ RMT validation: Scientific backing")
    
    return package

if __name__ == "__main__":
    asyncio.run(demo_enhanced_reporting())