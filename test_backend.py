#!/usr/bin/env python3
"""
Quick test of the backend implementation
"""
import asyncio
import sys
sys.path.append('backend')

from app.services.data_service import DataService
from app.services.analysis_service import AnalysisService

async def test_analysis():
    """Test the analysis service"""
    
    # Test portfolio (your retirement-focused stocks)
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'VTI', 'BND']
    print(f"Testing analysis with: {test_tickers}")
    
    # Initialize services
    data_service = DataService()
    analysis_service = AnalysisService(data_service)
    
    try:
        # Run analysis
        result = await analysis_service.analyze_portfolio(
            tickers=test_tickers,
            analysis_types=['basic', 'correlation']
        )
        
        # Print results
        print("\n" + "="*50)
        print("PORTFOLIO ANALYSIS RESULTS")
        print("="*50)
        
        # Metadata
        metadata = result['metadata']
        print(f"\nPortfolio: {', '.join(metadata['portfolio_tickers'])}")
        print(f"Analysis Period: {metadata['analysis_period']['start']} to {metadata['analysis_period']['end']}")
        print(f"Observations: {metadata['total_observations']}")
        
        # Basic metrics
        if 'basic' in result:
            basic = result['basic']
            print(f"\nBASIC METRICS:")
            print(f"  Annual Return: {basic['annual_return']:.1%}")
            print(f"  Annual Volatility: {basic['annual_volatility']:.1%}")
            print(f"  Sharpe Ratio: {basic['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {basic['max_drawdown']:.1%}")
        
        # Correlation analysis (YOUR KEY INSIGHT)
        if 'correlation' in result:
            corr = result['correlation']
            print(f"\nCORRELATION ANALYSIS (Your Innovation):")
            print(f"  Portfolio Assets: {len(metadata['portfolio_tickers'])}")
            print(f"  Effective Risk Factors: {corr['effective_rank']:.1f}")
            print(f"  Concentration Ratio: {corr['concentration_ratio']:.3f}")
            print(f"  Diversification Score: {1 - corr['concentration_ratio']:.3f}")
            
            if corr['concentration_ratio'] < 0.5:
                print(f"  ⚠️  Portfolio is more concentrated than it appears!")
        
        print("\n✅ Analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_analysis())
    sys.exit(0 if success else 1)
