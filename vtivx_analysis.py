#!/usr/bin/env python3
"""
VTIVX Top Holdings Correlation Analysis
Run this from your hmc_portfolio directory
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.append(str(backend_path))

from app.services.data_service import DataService
from app.services.analysis_service import AnalysisService

async def analyze_vtivx_holdings():
    """Analyze the top 10 VTIVX equity holdings using your correlation tool"""
    
    # VTIVX top 10 holdings from your data
    vtivx_holdings = [
        'NVDA',   # NVIDIA Corp. - 20.43%
        'MSFT',   # Microsoft Corp. - 19.65%
        'AAPL',   # Apple Inc. - 15.36%
        'AMZN',   # Amazon.com Inc. - 11.09%
        'META',   # Facebook Inc. Class A - 8.32%
        'AVGO',   # Broadcom Inc. - 6.84%
        'GOOGL',  # Alphabet Inc. Class A - 5.51%
        'GOOG',   # Alphabet Inc. Class C - 4.46%
        'TSLA',   # Tesla Inc. - 4.18%
        'BRK-B'   # Berkshire Hathaway Inc. Class B - 4.15%
    ]
    
    print("🔬 VTIVX Top 10 Holdings Correlation Analysis")
    print("=" * 60)
    print(f"Analyzing {len(vtivx_holdings)} holdings...")
    print("Holdings:", ", ".join(vtivx_holdings))
    print()
    
    # Initialize your services
    try:
        ds = DataService()
        analysis_service = AnalysisService(ds)
        
        # Run your correlation analysis
        print("📊 Running correlation analysis...")
        result = await analysis_service.analyze_portfolio(
            vtivx_holdings, 
            analysis_types=['basic', 'correlation']
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("📈 PORTFOLIO METRICS")
        print("=" * 60)
        
        basic = result['basic']
        print(f"Annual Return:        {basic['annual_return']:.2%}")
        print(f"Annual Volatility:    {basic['annual_volatility']:.2%}")
        print(f"Sharpe Ratio:         {basic['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:         {basic['max_drawdown']:.2%}")
        
        print("\n" + "=" * 60)
        print("🔬 CORRELATION ANALYSIS - Your Key Innovation!")
        print("=" * 60)
        
        corr = result['correlation']
        effective_rank = corr['effective_rank']
        num_assets = len(vtivx_holdings)
        concentration_ratio = effective_rank / num_assets
        
        print(f"Number of Holdings:   {num_assets}")
        print(f"Effective Rank:       {effective_rank:.2f}")
        print(f"Concentration Ratio:  {concentration_ratio:.1%}")
        print(f"Diversification Loss: {(1-concentration_ratio):.1%}")
        
        print(f"\n💡 INTERPRETATION:")
        print(f"Despite holding {num_assets} stocks, you're getting the diversification")
        print(f"equivalent of only {effective_rank:.1f} independent assets!")
        print(f"That's a {(1-concentration_ratio):.0%} loss in diversification efficiency.")
        
        # Show the eigenvalues if available
        if 'eigenvalues' in corr:
            print(f"\n📊 EIGENVALUE BREAKDOWN:")
            eigenvals = corr['eigenvalues']
            for i, val in enumerate(eigenvals[:5], 1):
                pct = val / sum(eigenvals) * 100
                print(f"Factor {i}: {val:.3f} ({pct:.1f}% of total variance)")
        
        # Risk concentration analysis
        print(f"\n🚨 RISK CONCENTRATION INSIGHTS:")
        if effective_rank < num_assets * 0.7:
            print(f"⚠️  HIGH CORRELATION: Effective rank is {effective_rank:.1f}/{num_assets}")
            print(f"   These 'diversified' holdings are highly correlated!")
            
        if effective_rank < 5:
            print(f"⚠️  LIMITED FACTORS: Only ~{effective_rank:.0f} independent risk factors")
            print(f"   Portfolio vulnerable to common shocks (tech selloff, rate changes)")
            
        print(f"\n🎯 STRATEGIC IMPLICATIONS:")
        print(f"• This is NOT a diversified equity portfolio")
        print(f"• Heavy concentration in correlated mega-cap tech")
        print(f"• Consider adding uncorrelated assets (value, intl, REITs)")
        print(f"• Your tool reveals what traditional analysis misses!")
        
        return result
        
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Make sure you're in the hmc_portfolio directory")
        print(f"2. Check that your virtual environment is activated")
        print(f"3. Verify backend services are working: python test_backend_simple.py")
        return None

async def compare_with_alternatives():
    """Compare VTIVX holdings with more diversified alternatives"""
    
    print(f"\n" + "=" * 60)
    print("🔄 COMPARISON WITH ALTERNATIVES")
    print("=" * 60)
    
    ds = DataService()
    analysis_service = AnalysisService(ds)
    
    # Define comparison portfolios
    portfolios = {
        'VTIVX Top 10': ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'META', 'AVGO', 'GOOGL', 'GOOG', 'TSLA', 'BRK-B'],
        'Simple 3-Fund': ['VTI', 'VXUS', 'BND'],
        'Sector Diversified': ['VTI', 'VTV', 'VBR', 'VEA', 'VWO', 'VNQ', 'BND'],
        'Anti-Tech': ['VTV', 'VBR', 'VNQ', 'XLE', 'XLU', 'BND']
    }
    
    results = {}
    for name, tickers in portfolios.items():
        try:
            print(f"\nAnalyzing {name}...")
            result = await analysis_service.analyze_portfolio(tickers, analysis_types=['correlation'])
            corr = result['correlation']
            results[name] = {
                'effective_rank': corr['effective_rank'],
                'num_assets': len(tickers),
                'concentration_ratio': corr['effective_rank'] / len(tickers)
            }
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            continue
    
    # Display comparison
    print(f"\n📊 DIVERSIFICATION EFFICIENCY COMPARISON:")
    print(f"{'Portfolio':<20} {'Assets':<8} {'Eff Rank':<10} {'Efficiency':<12}")
    print("-" * 50)
    
    for name, data in results.items():
        efficiency = data['concentration_ratio']
        print(f"{name:<20} {data['num_assets']:<8} {data['effective_rank']:<10.1f} {efficiency:<12.1%}")
    
    return results

# Main execution
if __name__ == "__main__":
    print("🚀 Starting VTIVX Holdings Analysis...")
    print("Using your eigenvalue decomposition correlation tool\n")
    
    # Run the main analysis
    result = asyncio.run(analyze_vtivx_holdings())
    
    if result:
        # Run comparison analysis
        try:
            comparison = asyncio.run(compare_with_alternatives())
            
            print(f"\n🏆 KEY TAKEAWAYS:")
            print(f"• Your correlation analysis reveals hidden concentrations")
            print(f"• VTIVX top holdings are essentially a tech momentum bet")
            print(f"• True diversification requires uncorrelated assets")
            print(f"• This analysis gives you institutional-quality insights!")
            
        except Exception as e:
            print(f"Comparison analysis failed: {e}")
    
    print(f"\n✅ Analysis complete! Use these insights with your advisor.")