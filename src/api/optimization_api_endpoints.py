"""
Enhanced FastAPI endpoints with portfolio optimization integration.

This module extends the existing API to include comprehensive portfolio
optimization, backtesting, and integrated analysis capabilities.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

# Import optimization modules with your specified paths
from analysis.portfolio_optimizer import (
    PortfolioOptimizer, OptimizationConstraints, OptimizationResult,
    optimize_max_sharpe_portfolio, optimize_risk_parity_portfolio
)
from analysis.portfolio_backtester import (
    PortfolioBacktester, BacktestConfig, BacktestResult,
    backtest_max_sharpe_strategy, compare_common_strategies
)
from analysis.optimization_integration import (
    IntegratedPortfolioAnalyzer, IntegratedAnalysisResult,
    quick_optimize_portfolio, compare_optimization_strategies
)

def convert_for_json(obj):
    """Convert numpy types and other non-serializable objects to JSON-compatible types."""
    import pandas as pd
    
    # Handle pandas objects first
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    # Handle numpy types
    elif hasattr(obj, 'item') and hasattr(obj, 'shape'):
        if obj.shape == ():  # Only call .item() on scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return str(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    else:
        return obj

# Pydantic models for optimization requests/responses
class OptimizationRequest(BaseModel):
    tickers: List[str]
    method: str = "max_sharpe"
    period: str = "1y"
    min_weight: float = 0.0
    max_weight: float = 0.3
    max_concentration: Optional[float] = None
    target_return: Optional[float] = None
    target_risk: Optional[float] = None
    risk_free_rate: float = 0.02
    risk_aversion: Optional[float] = None
    current_weights: Optional[List[float]] = None

class BacktestRequest(BaseModel):
    tickers: List[str]
    strategies: List[str] = ["max_sharpe", "min_variance", "risk_parity", "equal_weight"]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    rebalance_frequency: str = "M"
    transaction_cost: float = 0.001
    initial_capital: float = 1000000.0
    max_weight: float = 0.2

class IntegratedAnalysisRequest(BaseModel):
    tickers: List[str]
    period: str = "1y"
    optimization_methods: Optional[List[str]] = ["max_sharpe", "min_variance", "risk_parity"]
    include_backtesting: bool = True
    backtest_start_date: Optional[str] = None
    min_weight: float = 0.01
    max_weight: float = 0.3
    current_weights: Optional[List[float]] = None

class EfficientFrontierRequest(BaseModel):
    tickers: List[str]
    period: str = "1y"
    num_points: int = 25
    min_weight: float = 0.0
    max_weight: float = 0.3
    risk_free_rate: float = 0.02

def add_optimization_endpoints(app: FastAPI, data_service):
    """Add optimization endpoints to existing FastAPI app."""
    
    # Initialize optimization services
    optimizer = PortfolioOptimizer()
    backtester = PortfolioBacktester()
    integrated_analyzer = IntegratedPortfolioAnalyzer()
    
    @app.post("/optimize/portfolio")
    async def optimize_portfolio(request: OptimizationRequest):
        """
        Optimize portfolio weights using specified method.
        """
        try:
            # Fetch returns data
            returns_df = data_service.fetch_returns_data(
                request.tickers, period=request.period
            )
            
            if len(returns_df.columns) < 2:
                raise HTTPException(status_code=400, detail="Need at least 2 valid tickers")
            
            # Calculate expected returns and covariance
            expected_returns = returns_df.mean() * 252
            covariance_matrix = returns_df.cov() * 252
            
            # Set up constraints
            constraints = OptimizationConstraints(
                min_weight=request.min_weight,
                max_weight=request.max_weight,
                max_concentration=request.max_concentration,
                target_return=request.target_return,
                target_risk=request.target_risk
            )
            
            # Set up current weights if provided
            current_weights = None
            if request.current_weights:
                if len(request.current_weights) != len(returns_df.columns):
                    raise HTTPException(
                        status_code=400, 
                        detail="Current weights length doesn't match number of assets"
                    )
                current_weights = pd.Series(request.current_weights, index=returns_df.columns)
            
            # Optimization kwargs
            kwargs = {'risk_free_rate': request.risk_free_rate}
            if request.risk_aversion:
                kwargs['risk_aversion'] = request.risk_aversion
            
            # Optimize
            result = optimizer.optimize_portfolio(
                expected_returns, covariance_matrix, request.method,
                constraints, current_weights=current_weights, **kwargs
            )
            
            # Convert result to JSON-serializable format
            response = {
                'success': result.success,
                'message': result.message,
                'optimization_method': result.optimization_method,
                'weights': result.weights.to_dict() if len(result.weights) > 0 else {},
                'expected_return': float(result.expected_return),
                'expected_risk': float(result.expected_risk),
                'sharpe_ratio': float(result.sharpe_ratio),
                'objective_value': float(result.objective_value),
                'transaction_costs': float(result.transaction_costs),
                'turnover': float(result.turnover),
                'risk_decomposition': result.risk_decomposition or {}
            }
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    @app.post("/optimize/efficient-frontier")
    async def generate_efficient_frontier(request: EfficientFrontierRequest):
        """
        Generate efficient frontier points.
        """
        try:
            # Fetch returns data
            returns_df = data_service.fetch_returns_data(
                request.tickers, period=request.period
            )
            
            # Calculate expected returns and covariance
            expected_returns = returns_df.mean() * 252
            covariance_matrix = returns_df.cov() * 252
            
            # Set up constraints
            constraints = OptimizationConstraints(
                min_weight=request.min_weight,
                max_weight=request.max_weight
            )
            
            # Generate frontier
            frontier_data = optimizer.generate_efficient_frontier(
                expected_returns, covariance_matrix, 
                num_points=request.num_points,
                constraints=constraints,
                risk_free_rate=request.risk_free_rate
            )
            
            if not frontier_data.get('success', True):
                raise HTTPException(status_code=500, detail=frontier_data.get('error', 'Frontier generation failed'))
            
            # Convert to JSON-serializable format
            response = {
                'success': True,
                'num_points': len(frontier_data['returns']),
                'returns': [float(r) for r in frontier_data['returns']],
                'risks': [float(r) for r in frontier_data['risks']],
                'sharpe_ratios': [float(s) for s in frontier_data['sharpe_ratios']],
                'optimal_point': {
                    'return': float(max(frontier_data['returns'])),
                    'risk': float(frontier_data['risks'][np.argmax(frontier_data['sharpe_ratios'])]),
                    'sharpe': float(max(frontier_data['sharpe_ratios']))
                }
            }
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Efficient frontier generation failed: {str(e)}")
    
    @app.post("/backtest/strategies")
    async def backtest_strategies(request: BacktestRequest):
        """
        Backtest multiple portfolio optimization strategies.
        """
        try:
            # Fetch returns data
            returns_df = data_service.fetch_returns_data(
                request.tickers, period="3y"  # Need more data for backtesting
            )
            
            if len(returns_df) < 500:  # Need sufficient history
                raise HTTPException(status_code=400, detail="Insufficient data for backtesting (need ~2 years)")
            
            # Set up backtest configuration
            end_date = request.end_date or returns_df.index[-1].strftime('%Y-%m-%d')
            start_date = request.start_date or (returns_df.index[-1] - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
            
            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                rebalance_frequency=request.rebalance_frequency,
                transaction_cost=request.transaction_cost,
                initial_capital=request.initial_capital,
                max_weight=request.max_weight
            )
            
            # Run backtests
            results = {}
            for strategy in request.strategies:
                try:
                    backtest_result = backtester.backtest_strategy(
                        returns_df, strategy, config
                    )
                    
                    if backtest_result.success:
                        metrics = backtest_result.performance_metrics
                        results[strategy] = {
                            'success': True,
                            'total_return': float(metrics.total_return),
                            'annualized_return': float(metrics.annualized_return),
                            'annualized_volatility': float(metrics.annualized_volatility),
                            'sharpe_ratio': float(metrics.sharpe_ratio),
                            'sortino_ratio': float(metrics.sortino_ratio),
                            'max_drawdown': float(metrics.max_drawdown),
                            'calmar_ratio': float(metrics.calmar_ratio),
                            'win_rate': float(metrics.win_rate),
                            'portfolio_value_final': float(backtest_result.portfolio_value.iloc[-1]),
                            'total_transaction_costs': float(backtest_result.transaction_costs.sum()),
                            'average_turnover': float(backtest_result.turnover.mean())
                        }
                    else:
                        results[strategy] = {
                            'success': False,
                            'error': backtest_result.message
                        }
                        
                except Exception as e:
                    results[strategy] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Summary statistics
            successful_results = {k: v for k, v in results.items() if v.get('success', False)}
            if successful_results:
                best_sharpe = max(r['sharpe_ratio'] for r in successful_results.values())
                best_return = max(r['annualized_return'] for r in successful_results.values())
                best_strategy_sharpe = [k for k, v in successful_results.items() if v['sharpe_ratio'] == best_sharpe][0]
                best_strategy_return = [k for k, v in successful_results.items() if v['annualized_return'] == best_return][0]
                
                summary = {
                    'best_sharpe_strategy': best_strategy_sharpe,
                    'best_sharpe_ratio': float(best_sharpe),
                    'best_return_strategy': best_strategy_return,
                    'best_return': float(best_return),
                    'strategies_tested': len(request.strategies),
                    'successful_strategies': len(successful_results)
                }
            else:
                summary = {
                    'strategies_tested': len(request.strategies),
                    'successful_strategies': 0,
                    'message': 'No strategies completed successfully'
                }
            
            return {
                'backtest_config': {
                    'start_date': start_date,
                    'end_date': end_date,
                    'rebalance_frequency': request.rebalance_frequency,
                    'transaction_cost': request.transaction_cost,
                    'initial_capital': request.initial_capital
                },
                'results': results,
                'summary': summary
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backtesting failed: {str(e)}")
    
    @app.post("/analyze/integrated")
    async def integrated_analysis(request: IntegratedAnalysisRequest):
        """
        Perform comprehensive integrated analysis combining risk assessment and optimization.
        """
        try:
            # Fetch returns data
            returns_df = data_service.fetch_returns_data(
                request.tickers, period=request.period
            )
            
            if len(returns_df.columns) < 3:
                raise HTTPException(status_code=400, detail="Need at least 3 valid tickers for integrated analysis")
            
            # Set up current weights if provided
            current_weights = None
            if request.current_weights:
                if len(request.current_weights) != len(returns_df.columns):
                    raise HTTPException(
                        status_code=400, 
                        detail="Current weights length doesn't match number of assets"
                    )
                current_weights = pd.Series(request.current_weights, index=returns_df.columns)
                current_weights = current_weights / current_weights.sum()
            
            # Set up optimization constraints
            constraints = OptimizationConstraints(
                min_weight=request.min_weight,
                max_weight=request.max_weight,
                long_only=True
            )
            
            print("About to run integrated analysis...")  # Debug log
            
            # Run integrated analysis
            result = integrated_analyzer.analyze_and_optimize(
                returns_df,
                current_weights=current_weights,
                constraints=constraints,
                include_backtesting=request.include_backtesting,
                backtest_start_date=request.backtest_start_date,
                optimization_methods=request.optimization_methods
            )
            
            print("Integrated analysis completed, building response...")  # Debug log
            
            # Convert to JSON-serializable format
            response = {
                'analysis_successful': True,
                'risk_analysis': result.risk_analysis,
                'optimization_results': {},
                'recommended_portfolio': {},
                'efficient_frontier': result.efficient_frontier,
                'comparative_analysis': result.comparative_analysis.to_dict('records') if not result.comparative_analysis.empty else [],
                'insights': result.insights,
                'advisor_recommendations': result.advisor_recommendations,
                'backtesting_results': {}
            }
            
            print("Converting optimization results...")  # Debug log
            
            # Convert optimization results
            for method, opt_result in result.optimization_results.items():
                response['optimization_results'][method] = {
                    'success': opt_result.success,
                    'weights': opt_result.weights.to_dict() if len(opt_result.weights) > 0 else {},
                    'expected_return': float(opt_result.expected_return),
                    'expected_risk': float(opt_result.expected_risk),
                    'sharpe_ratio': float(opt_result.sharpe_ratio),
                    'message': opt_result.message
                }
            
            print("Converting recommended portfolio...")  # Debug log
            
            # Convert recommended portfolio
            if result.recommended_portfolio.success:
                response['recommended_portfolio'] = {
                    'method': result.recommended_portfolio.optimization_method,
                    'weights': result.recommended_portfolio.weights.to_dict(),
                    'expected_return': float(result.recommended_portfolio.expected_return),
                    'expected_risk': float(result.recommended_portfolio.expected_risk),
                    'sharpe_ratio': float(result.recommended_portfolio.sharpe_ratio),
                    'success': True
                }
            else:
                response['recommended_portfolio'] = {
                    'success': False,
                    'message': result.recommended_portfolio.message
                }
            
            print("Converting backtesting results...")  # Debug log
            
            # Convert backtesting results if available
            if result.backtesting_results:
                for method, backtest_result in result.backtesting_results.items():
                    if backtest_result.success:
                        metrics = backtest_result.performance_metrics
                        response['backtesting_results'][method] = {
                            'success': True,
                            'annualized_return': float(metrics.annualized_return),
                            'annualized_volatility': float(metrics.annualized_volatility),
                            'sharpe_ratio': float(metrics.sharpe_ratio),
                            'max_drawdown': float(metrics.max_drawdown),
                            'calmar_ratio': float(metrics.calmar_ratio),
                            'win_rate': float(metrics.win_rate)
                        }
                    else:
                        response['backtesting_results'][method] = {
                            'success': False,
                            'error': backtest_result.message
                        }
            
            print("Converting to JSON...")  # Debug log
            
            # Convert all numpy types to JSON-serializable format
            response = convert_for_json(response)
            
            print("Returning response...")  # Debug log
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"ERROR in integrated analysis: {str(e)}")  # Debug log
            import traceback
            traceback.print_exc()  # This will show the full stack trace
            raise HTTPException(status_code=500, detail=f"Integrated analysis failed: {str(e)}")
        
    @app.post("/optimize/compare-strategies")
    async def compare_strategies(
        tickers: List[str] = Query(...),
        strategies: List[str] = Query(["max_sharpe", "min_variance", "risk_parity", "equal_weight"]),
        period: str = Query("1y"),
        max_weight: float = Query(0.3)
    ):
        """
        Compare multiple optimization strategies side by side.
        """
        try:
            # Fetch returns data
            returns_df = data_service.fetch_returns_data(tickers, period=period)
            
            if len(returns_df.columns) < 2:
                raise HTTPException(status_code=400, detail="Need at least 2 valid tickers")
            
            # Calculate expected returns and covariance
            expected_returns = returns_df.mean() * 252
            covariance_matrix = returns_df.cov() * 252
            
            # Set up constraints
            constraints = OptimizationConstraints(
                min_weight=0.01,
                max_weight=max_weight,
                long_only=True
            )
            
            # Compare strategies
            results = {}
            comparison_data = []
            
            for strategy in strategies:
                try:
                    result = optimizer.optimize_portfolio(
                        expected_returns, covariance_matrix, strategy, constraints
                    )
                    
                    results[strategy] = result
                    
                    if result.success:
                        # Calculate additional metrics
                        concentration = np.sum(result.weights ** 2) if len(result.weights) > 0 else 0
                        max_asset_weight = result.weights.max() if len(result.weights) > 0 else 0
                        num_holdings = np.sum(result.weights > 0.01) if len(result.weights) > 0 else 0
                        
                        comparison_data.append({
                            'strategy': strategy.replace('_', ' ').title(),
                            'expected_return': float(result.expected_return),
                            'expected_risk': float(result.expected_risk),
                            'sharpe_ratio': float(result.sharpe_ratio),
                            'concentration': float(concentration),
                            'max_weight': float(max_asset_weight),
                            'num_holdings': int(num_holdings),
                            'success': True
                        })
                    else:
                        comparison_data.append({
                            'strategy': strategy.replace('_', ' ').title(),
                            'expected_return': None,
                            'expected_risk': None,
                            'sharpe_ratio': None,
                            'concentration': None,
                            'max_weight': None,
                            'num_holdings': None,
                            'success': False,
                            'error': result.message
                        })
                        
                except Exception as e:
                    comparison_data.append({
                        'strategy': strategy.replace('_', ' ').title(),
                        'success': False,
                        'error': str(e)
                    })
            
            # Identify best strategies
            successful_strategies = [s for s in comparison_data if s.get('success', False)]
            
            summary = {}
            if successful_strategies:
                best_sharpe = max(s['sharpe_ratio'] for s in successful_strategies)
                best_return = max(s['expected_return'] for s in successful_strategies)
                lowest_risk = min(s['expected_risk'] for s in successful_strategies)
                
                summary = {
                    'best_sharpe_strategy': next(s['strategy'] for s in successful_strategies if s['sharpe_ratio'] == best_sharpe),
                    'best_sharpe_ratio': float(best_sharpe),
                    'best_return_strategy': next(s['strategy'] for s in successful_strategies if s['expected_return'] == best_return),
                    'best_return': float(best_return),
                    'lowest_risk_strategy': next(s['strategy'] for s in successful_strategies if s['expected_risk'] == lowest_risk),
                    'lowest_risk': float(lowest_risk),
                    'successful_optimizations': len(successful_strategies),
                    'total_strategies': len(strategies)
                }
            
            return {
                'comparison_data': comparison_data,
                'summary': summary,
                'detailed_results': {
                    strategy: {
                        'weights': result.weights.to_dict() if result.success and len(result.weights) > 0 else {},
                        'success': result.success,
                        'message': result.message
                    } for strategy, result in results.items()
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Strategy comparison failed: {str(e)}")
    
    @app.get("/optimize/sample/quick")
    async def quick_sample_optimization():
        """
        Quick optimization of sample portfolio for testing.
        """
        try:
            # Load sample data (you'll need to implement this in your data service)
            from data.data_service import load_sample_portfolio_data
            returns_df, _ = load_sample_portfolio_data("sample")
            
            # Quick max Sharpe optimization
            result = quick_optimize_portfolio(returns_df, method='max_sharpe', max_weight=0.25)
            
            if result.success:
                return {
                    'success': True,
                    'method': result.optimization_method,
                    'weights': result.weights.to_dict(),
                    'expected_return': float(result.expected_return),
                    'expected_risk': float(result.expected_risk),
                    'sharpe_ratio': float(result.sharpe_ratio),
                    'concentration': float(np.sum(result.weights ** 2)),
                    'max_weight': float(result.weights.max()),
                    'message': result.message
                }
            else:
                return {
                    'success': False,
                    'error': result.message
                }
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Quick optimization failed: {str(e)}")
    
    @app.post("/reports/optimization")
    async def generate_optimization_report(
        analysis_data: Dict[str, Any],
        report_type: str = Query("comprehensive", description="Type of report: comprehensive, executive_summary, technical")
    ):
        """
        Generate formatted optimization report from analysis results.
        """
        try:
            report_content = f"PORTFOLIO OPTIMIZATION REPORT\n"
            report_content += "=" * 40 + "\n\n"
            
            if 'recommended_portfolio' in analysis_data and analysis_data['recommended_portfolio'].get('success'):
                rec = analysis_data['recommended_portfolio']
                report_content += f"RECOMMENDED STRATEGY: {rec.get('method', 'Unknown').replace('_', ' ').title()}\n"
                report_content += f"Expected Return: {rec.get('expected_return', 0):.1%}\n"
                report_content += f"Expected Risk: {rec.get('expected_risk', 0):.1%}\n"
                report_content += f"Sharpe Ratio: {rec.get('sharpe_ratio', 0):.3f}\n\n"
            
            if 'insights' in analysis_data:
                report_content += "KEY INSIGHTS:\n"
                for insight in analysis_data['insights']:
                    report_content += f"• {insight}\n"
                report_content += "\n"
            
            if 'advisor_recommendations' in analysis_data:
                report_content += "RECOMMENDATIONS:\n"
                for rec in analysis_data['advisor_recommendations']:
                    report_content += f"• {rec}\n"
            
            return {
                'report_type': report_type,
                'report_content': report_content,
                'generated_at': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
    
    # Enhanced health check with optimization components
    @app.get("/health/optimization")
    async def optimization_health_check():
        """Health check for optimization components."""
        try:
            # Test basic functionality
            import numpy as np
            test_data = pd.DataFrame(np.random.normal(0, 0.01, (100, 3)), columns=['A', 'B', 'C'])
            
            # Test optimizer
            test_optimizer = PortfolioOptimizer()
            
            # Test backtester
            test_backtester = PortfolioBacktester()
            
            # Test integrated analyzer
            test_integrated = IntegratedPortfolioAnalyzer()
            
            return {
                'status': 'healthy',
                'service': 'portfolio-optimization',
                'components': {
                    'portfolio_optimizer': 'operational',
                    'portfolio_backtester': 'operational',
                    'integrated_analyzer': 'operational'
                },
                'available_methods': {
                    'optimization': ['max_sharpe', 'min_variance', 'risk_parity', 'mean_variance'],
                    'backtesting': ['equal_weight', 'max_sharpe', 'min_variance', 'risk_parity'],
                    'analysis': ['integrated', 'efficient_frontier', 'strategy_comparison']
                },
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'degraded',
                'service': 'portfolio-optimization',
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }
    
    return app

# Usage example for your main.py:
"""
# In your main.py, add these endpoints:

from optimization_api_endpoints import add_optimization_endpoints

# After creating your FastAPI app:
app = add_optimization_endpoints(app, data_service)
"""