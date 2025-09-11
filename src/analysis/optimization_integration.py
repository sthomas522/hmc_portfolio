"""
Portfolio Optimization Integration Module

This module integrates the portfolio optimization engine with the existing
portfolio analyzer to provide comprehensive analysis and optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import sys
import os

# Add the analysis module to the path for imports
current_dir = os.path.dirname(__file__)
analysis_dir = os.path.abspath(os.path.join(current_dir, '..', 'analysis'))
sys.path.insert(0, analysis_dir)

# Import the existing analyzer and new optimization components
from portfolio_analyzer import EnhancedPortfolioAnalyzer, PortfolioCorrelationAnalyzer
from portfolio_optimizer import PortfolioOptimizer, OptimizationConstraints, OptimizationResult
from portfolio_backtester import PortfolioBacktester, BacktestConfig, BacktestResult

@dataclass
class IntegratedAnalysisResult:
    """Combined analysis and optimization results."""
    risk_analysis: Dict[str, Any]
    optimization_results: Dict[str, OptimizationResult]
    recommended_portfolio: OptimizationResult
    efficient_frontier: Dict[str, Any]
    backtesting_results: Optional[Dict[str, BacktestResult]]
    comparative_analysis: pd.DataFrame
    insights: List[str]
    advisor_recommendations: List[str]

class IntegratedPortfolioAnalyzer:
    """
    Integrated analyzer combining risk analysis with portfolio optimization.
    
    Provides comprehensive portfolio analysis including:
    - Risk structure analysis (correlation, tail risk, factor attribution)
    - Portfolio optimization across multiple methods
    - Efficient frontier generation
    - Strategy backtesting and comparison
    - Integrated insights and recommendations
    """
    
    def __init__(self):
        self.risk_analyzer = EnhancedPortfolioAnalyzer()
        self.optimizer = PortfolioOptimizer()
        self.backtester = PortfolioBacktester()
        
        # Default optimization methods to test
        self.default_methods = [
            'max_sharpe',
            'min_variance', 
            'risk_parity',
            'mean_variance'
        ]
    
    def analyze_and_optimize(self,
                           returns_data: pd.DataFrame,
                           current_weights: Optional[pd.Series] = None,
                           constraints: Optional[OptimizationConstraints] = None,
                           include_backtesting: bool = True,
                           backtest_start_date: Optional[str] = None,
                           optimization_methods: Optional[List[str]] = None) -> IntegratedAnalysisResult:
        """
        Perform comprehensive analysis and optimization.
        
        Args:
            returns_data: Historical returns data
            current_weights: Current portfolio weights (if any)
            constraints: Optimization constraints
            include_backtesting: Whether to include backtesting analysis
            backtest_start_date: Start date for backtesting (if different from data start)
            optimization_methods: List of optimization methods to compare
            
        Returns:
            IntegratedAnalysisResult with complete analysis
        """
        if optimization_methods is None:
            optimization_methods = self.default_methods
            
        if constraints is None:
            constraints = OptimizationConstraints(
                min_weight=0.01,
                max_weight=0.3,
                long_only=True
            )
        
        # Step 1: Risk Analysis
        print("Performing risk structure analysis...")
        risk_analysis = self.risk_analyzer.analyze_comprehensive_portfolio_risk(
            returns_data, current_weights, include_tail_risk=True, include_factor_attribution=True
        )
        
        # Step 2: Calculate Expected Returns and Covariance
        expected_returns = returns_data.mean() * 252  # Annualized
        covariance_matrix = returns_data.cov() * 252  # Annualized
        
        # Step 3: Portfolio Optimization
        print("Running portfolio optimization...")
        optimization_results = {}
        
        for method in optimization_methods:
            try:
                result = self.optimizer.optimize_portfolio(
                    expected_returns, covariance_matrix, method, 
                    constraints, current_weights=current_weights
                )
                optimization_results[method] = result
                print(f"✓ {method}: Sharpe={result.sharpe_ratio:.3f}, Return={result.expected_return:.1%}")
            except Exception as e:
                print(f"✗ {method}: Failed - {str(e)}")
        
        # Step 4: Generate Efficient Frontier
        print("Generating efficient frontier...")
        try:
            efficient_frontier = self.optimizer.generate_efficient_frontier(
                expected_returns, covariance_matrix, 
                num_points=25, constraints=constraints
            )
        except Exception as e:
            efficient_frontier = {'success': False, 'error': str(e)}
        
        # Step 5: Select Recommended Portfolio
        recommended_portfolio = self._select_recommended_portfolio(optimization_results)
        
        # Step 6: Backtesting (if requested)
        backtesting_results = None
        if include_backtesting and len(returns_data) > 252:  # Need at least 1 year of data
            print("Running strategy backtesting...")
            try:
                backtesting_results = self._run_strategy_backtesting(
                    returns_data, optimization_methods, backtest_start_date
                )
            except Exception as e:
                print(f"Backtesting failed: {str(e)}")
        
        # Step 7: Comparative Analysis
        comparative_analysis = self._create_comparative_analysis(
            optimization_results, current_weights, expected_returns, covariance_matrix
        )
        
        # Step 8: Generate Insights and Recommendations
        insights = self._generate_integrated_insights(
            risk_analysis, optimization_results, efficient_frontier, backtesting_results
        )
        
        advisor_recommendations = self._generate_advisor_recommendations(
            risk_analysis, recommended_portfolio, comparative_analysis
        )
        
        return IntegratedAnalysisResult(
            risk_analysis=risk_analysis,
            optimization_results=optimization_results,
            recommended_portfolio=recommended_portfolio,
            efficient_frontier=efficient_frontier,
            backtesting_results=backtesting_results,
            comparative_analysis=comparative_analysis,
            insights=insights,
            advisor_recommendations=advisor_recommendations
        )
    
    def _select_recommended_portfolio(self, optimization_results: Dict[str, OptimizationResult]) -> OptimizationResult:
        """Select the recommended portfolio from optimization results."""
        # Filter successful optimizations
        successful_results = {k: v for k, v in optimization_results.items() if v.success}
        
        if not successful_results:
            # Return a default equal weight portfolio if all optimizations failed
            return OptimizationResult(
                weights=pd.Series(),
                expected_return=0,
                expected_risk=0,
                sharpe_ratio=0,
                optimization_method='equal_weight',
                success=False,
                message="All optimizations failed",
                objective_value=0
            )
        
        # Scoring system for recommendation
        scores = {}
        
        for method, result in successful_results.items():
            score = 0
            
            # Sharpe ratio component (40% weight)
            if result.sharpe_ratio > 0:
                sharpe_scores = [r.sharpe_ratio for r in successful_results.values() if r.sharpe_ratio > 0]
                if sharpe_scores:
                    sharpe_percentile = (result.sharpe_ratio - min(sharpe_scores)) / (max(sharpe_scores) - min(sharpe_scores)) if max(sharpe_scores) > min(sharpe_scores) else 0.5
                    score += sharpe_percentile * 0.4
            
            # Risk level component (20% weight) - prefer moderate risk
            risk_scores = [r.expected_risk for r in successful_results.values()]
            if risk_scores:
                # Prefer portfolios with risk in middle range
                median_risk = np.median(risk_scores)
                risk_deviation = abs(result.expected_risk - median_risk) / median_risk if median_risk > 0 else 0
                risk_score = max(0, 1 - risk_deviation)
                score += risk_score * 0.2
            
            # Diversification component (20% weight)
            if hasattr(result, 'weights') and len(result.weights) > 0:
                # Calculate Herfindahl index (lower is better)
                concentration = np.sum(result.weights ** 2)
                diversification_score = max(0, 1 - concentration)
                score += diversification_score * 0.2
            
            # Method preference (20% weight)
            method_preferences = {
                'max_sharpe': 0.9,
                'risk_parity': 0.8,
                'min_variance': 0.6,
                'mean_variance': 0.7,
                'multi_objective': 0.8
            }
            method_score = method_preferences.get(method, 0.5)
            score += method_score * 0.2
            
            scores[method] = score
        
        # Select highest scoring method
        best_method = max(scores, key=scores.get)
        return successful_results[best_method]
    
    def _run_strategy_backtesting(self, returns_data: pd.DataFrame, 
                                 methods: List[str], 
                                 start_date: Optional[str] = None) -> Dict[str, BacktestResult]:
        """Run backtesting for optimization strategies."""
        # Use last 2 years for backtesting if no start date specified
        if start_date is None:
            end_date = returns_data.index[-1]
            start_date = (end_date - pd.DateOffset(years=2)).strftime('%Y-%m-%d')
        
        end_date = returns_data.index[-1].strftime('%Y-%m-%d')
        
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            rebalance_frequency='M',  # Monthly rebalancing
            lookback_window=252,      # 1 year lookback
            transaction_cost=0.001,   # 0.1% transaction cost
            max_weight=0.3           # 30% maximum weight
        )
        
        backtest_results = {}
        
        for method in methods:
            try:
                result = self.backtester.backtest_strategy(
                    returns_data, method, config
                )
                backtest_results[method] = result
            except Exception as e:
                print(f"Backtesting failed for {method}: {str(e)}")
        
        return backtest_results
    
    def _create_comparative_analysis(self, optimization_results: Dict[str, OptimizationResult],
                                   current_weights: Optional[pd.Series],
                                   expected_returns: pd.Series,
                                   covariance_matrix: pd.DataFrame) -> pd.DataFrame:
        """Create comparative analysis of optimization methods."""
        comparison_data = []
        
        # Add current portfolio if provided
        if current_weights is not None:
            current_return = np.dot(current_weights, expected_returns)
            current_risk = np.sqrt(np.dot(current_weights, np.dot(covariance_matrix, current_weights)))
            current_sharpe = current_return / current_risk if current_risk > 0 else 0
            current_concentration = np.sum(current_weights ** 2)
            
            comparison_data.append({
                'Strategy': 'Current Portfolio',
                'Expected Return': current_return,
                'Expected Risk': current_risk,
                'Sharpe Ratio': current_sharpe,
                'Concentration': current_concentration,
                'Max Weight': current_weights.max(),
                'Success': True
            })
        
        # Add optimization results
        for method, result in optimization_results.items():
            if result.success:
                concentration = np.sum(result.weights ** 2) if len(result.weights) > 0 else 0
                max_weight = result.weights.max() if len(result.weights) > 0 else 0
                
                comparison_data.append({
                    'Strategy': method.replace('_', ' ').title(),
                    'Expected Return': result.expected_return,
                    'Expected Risk': result.expected_risk,
                    'Sharpe Ratio': result.sharpe_ratio,
                    'Concentration': concentration,
                    'Max Weight': max_weight,
                    'Success': True
                })
            else:
                comparison_data.append({
                    'Strategy': method.replace('_', ' ').title(),
                    'Expected Return': np.nan,
                    'Expected Risk': np.nan,
                    'Sharpe Ratio': np.nan,
                    'Concentration': np.nan,
                    'Max Weight': np.nan,
                    'Success': False
                })
        
        return pd.DataFrame(comparison_data)
    
    def _generate_integrated_insights(self, risk_analysis: Dict[str, Any],
                                    optimization_results: Dict[str, OptimizationResult],
                                    efficient_frontier: Dict[str, Any],
                                    backtesting_results: Optional[Dict[str, BacktestResult]]) -> List[str]:
        """Generate integrated insights combining risk and optimization analysis."""
        insights = []
        
        # Risk structure insights
        if risk_analysis.get('analysis_successful'):
            effective_rank = risk_analysis.get('effective_rank', 0)
            num_assets = risk_analysis.get('num_assets', 0)
            diversification_loss = risk_analysis.get('diversification_loss', 0)
            
            insights.append(f"Portfolio Structure: {num_assets} assets behave like {effective_rank:.1f} independent investments ({diversification_loss:.0%} diversification loss)")
            
            if 'tail_risk_analysis' in risk_analysis:
                tail_risk = risk_analysis['tail_risk_analysis']
                max_dd = abs(tail_risk['drawdown_analysis']['maximum_drawdown'])
                var_99 = abs(tail_risk['var_cvar_analysis']['99.0%']['historical_var_daily'])
                insights.append(f"Tail Risk: {max_dd:.1%} maximum drawdown, {var_99:.2%} daily 99% VaR")
        
        # Optimization insights
        successful_opts = {k: v for k, v in optimization_results.items() if v.success}
        if successful_opts:
            best_sharpe = max(r.sharpe_ratio for r in successful_opts.values())
            best_method = [k for k, v in successful_opts.items() if v.sharpe_ratio == best_sharpe][0]
            
            insights.append(f"Optimization Results: {best_method.replace('_', ' ').title()} achieves highest Sharpe ratio ({best_sharpe:.3f})")
            
            # Risk-return improvement
            sharpe_range = max(r.sharpe_ratio for r in successful_opts.values()) - min(r.sharpe_ratio for r in successful_opts.values())
            if sharpe_range > 0.2:
                insights.append(f"Optimization Impact: Strategy selection can improve Sharpe ratio by up to {sharpe_range:.3f}")
        
        # Efficient frontier insights
        if efficient_frontier.get('success'):
            frontier_sharpes = efficient_frontier.get('sharpe_ratios', [])
            if frontier_sharpes:
                max_frontier_sharpe = max(frontier_sharpes)
                insights.append(f"Efficient Frontier: Maximum achievable Sharpe ratio is {max_frontier_sharpe:.3f}")
        
        # Backtesting insights
        if backtesting_results:
            successful_backtests = {k: v for k, v in backtesting_results.items() if v.success}
            if successful_backtests:
                best_backtest_return = max(v.performance_metrics.annualized_return for v in successful_backtests.values())
                best_backtest_method = [k for k, v in successful_backtests.items() 
                                      if v.performance_metrics.annualized_return == best_backtest_return][0]
                insights.append(f"Historical Performance: {best_backtest_method.replace('_', ' ').title()} delivered {best_backtest_return:.1%} annualized return")
        
        return insights
    
    def _generate_advisor_recommendations(self, risk_analysis: Dict[str, Any],
                                        recommended_portfolio: OptimizationResult,
                                        comparative_analysis: pd.DataFrame) -> List[str]:
        """Generate advisor-ready recommendations."""
        recommendations = []
        
        if recommended_portfolio.success:
            method_name = recommended_portfolio.optimization_method.replace('_', ' ').title()
            
            recommendations.append(f"Primary Recommendation: Implement {method_name} strategy (Sharpe: {recommended_portfolio.sharpe_ratio:.3f})")
            recommendations.append(f"Expected Performance: {recommended_portfolio.expected_return:.1%} annual return, {recommended_portfolio.expected_risk:.1%} volatility")
            
            # Weight distribution insights
            if len(recommended_portfolio.weights) > 0:
                max_weight = recommended_portfolio.weights.max()
                top_3_weights = recommended_portfolio.weights.nlargest(3)
                top_holdings = ", ".join([f"{ticker} ({weight:.1%})" for ticker, weight in top_3_weights.items()])
                
                recommendations.append(f"Portfolio Construction: Maximum position {max_weight:.1%}, top holdings: {top_holdings}")
                
                if max_weight > 0.25:
                    recommendations.append("Concentration Alert: Consider position size limits to reduce single-name risk")
        
        # Risk management recommendations
        if risk_analysis.get('analysis_successful'):
            diversification_loss = risk_analysis.get('diversification_loss', 0)
            if diversification_loss > 0.5:
                recommendations.append("Diversification Alert: High correlation reduces effective diversification - consider uncorrelated assets")
            
            if 'tail_risk_analysis' in risk_analysis:
                tail_risk = risk_analysis['tail_risk_analysis']
                max_dd = abs(tail_risk['drawdown_analysis']['maximum_drawdown'])
                if max_dd > 0.2:
                    recommendations.append(f"Downside Protection: Historical {max_dd:.1%} maximum loss suggests implementing risk management overlays")
        
        # Comparative insights
        successful_strategies = comparative_analysis[comparative_analysis['Success'] == True]
        if len(successful_strategies) > 1:
            sharpe_improvement = successful_strategies['Sharpe Ratio'].max() - successful_strategies['Sharpe Ratio'].min()
            if sharpe_improvement > 0.3:
                recommendations.append(f"Strategy Selection Impact: Optimization can improve risk-adjusted returns by {sharpe_improvement:.3f} Sharpe ratio points")
        
        return recommendations
    
    def generate_optimization_report(self, analysis_result: IntegratedAnalysisResult,
                                   report_type: str = 'comprehensive') -> str:
        """Generate formatted report from integrated analysis."""
        if report_type == 'executive_summary':
            return self._generate_executive_optimization_report(analysis_result)
        elif report_type == 'technical':
            return self._generate_technical_optimization_report(analysis_result)
        else:
            return self._generate_comprehensive_optimization_report(analysis_result)
    
    def _generate_executive_optimization_report(self, result: IntegratedAnalysisResult) -> str:
        """Generate executive summary report."""
        report = "PORTFOLIO OPTIMIZATION EXECUTIVE SUMMARY\n"
        report += "=" * 45 + "\n\n"
        
        # Key recommendation
        if result.recommended_portfolio.success:
            method = result.recommended_portfolio.optimization_method.replace('_', ' ').title()
            report += f"RECOMMENDED STRATEGY: {method}\n"
            report += f"Expected Return: {result.recommended_portfolio.expected_return:.1%}\n"
            report += f"Expected Risk: {result.recommended_portfolio.expected_risk:.1%}\n"
            report += f"Sharpe Ratio: {result.recommended_portfolio.sharpe_ratio:.3f}\n\n"
        
        # Key insights
        report += "KEY INSIGHTS:\n"
        for insight in result.insights[:3]:
            report += f"• {insight}\n"
        
        report += "\nRECOMMENDations:\n"
        for rec in result.advisor_recommendations[:3]:
            report += f"• {rec}\n"
        
        return report
    
    def _generate_technical_optimization_report(self, result: IntegratedAnalysisResult) -> str:
        """Generate technical analysis report."""
        report = "TECHNICAL PORTFOLIO OPTIMIZATION ANALYSIS\n"
        report += "=" * 45 + "\n\n"
        
        # Risk analysis summary
        if result.risk_analysis.get('analysis_successful'):
            report += "RISK STRUCTURE ANALYSIS:\n"
            report += f"Effective Rank: {result.risk_analysis.get('effective_rank', 'N/A'):.2f}\n"
            report += f"Diversification Loss: {result.risk_analysis.get('diversification_loss', 0):.1%}\n"
            
            if 'mp_fitting_successful' in result.risk_analysis:
                report += f"Signal Factors: {result.risk_analysis.get('num_signal_factors', 'N/A')}\n"
                report += f"Noise Fraction: {result.risk_analysis.get('noise_fraction', 0):.1%}\n"
            report += "\n"
        
        # Optimization results
        report += "OPTIMIZATION RESULTS:\n"
        successful_opts = {k: v for k, v in result.optimization_results.items() if v.success}
        for method, opt_result in successful_opts.items():
            report += f"{method.replace('_', ' ').title()}: "
            report += f"Sharpe={opt_result.sharpe_ratio:.3f}, "
            report += f"Return={opt_result.expected_return:.1%}, "
            report += f"Risk={opt_result.expected_risk:.1%}\n"
        
        # Backtesting results
        if result.backtesting_results:
            report += "\nBACKTESTING PERFORMANCE:\n"
            for method, backtest in result.backtesting_results.items():
                if backtest.success:
                    metrics = backtest.performance_metrics
                    report += f"{method.replace('_', ' ').title()}: "
                    report += f"Return={metrics.annualized_return:.1%}, "
                    report += f"Sharpe={metrics.sharpe_ratio:.3f}, "
                    report += f"MaxDD={abs(metrics.max_drawdown):.1%}\n"
        
        return report
    
    def _generate_comprehensive_optimization_report(self, result: IntegratedAnalysisResult) -> str:
        """Generate comprehensive optimization report."""
        report = "COMPREHENSIVE PORTFOLIO OPTIMIZATION REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += self._generate_executive_optimization_report(result)
        report += "\n" + "=" * 50 + "\n\n"
        report += self._generate_technical_optimization_report(result)
        
        # Add detailed insights
        report += "\n" + "=" * 50 + "\n"
        report += "DETAILED ANALYSIS:\n\n"
        
        for insight in result.insights:
            report += f"• {insight}\n"
        
        report += "\nCOMPREHENSIVE RECOMMENDATIONS:\n"
        for rec in result.advisor_recommendations:
            report += f"• {rec}\n"
        
        return report


# Convenience functions for quick analysis
def quick_optimize_portfolio(returns_data: pd.DataFrame,
                           method: str = 'max_sharpe',
                           max_weight: float = 0.3) -> OptimizationResult:
    """Quick portfolio optimization with default settings."""
    analyzer = IntegratedPortfolioAnalyzer()
    
    expected_returns = returns_data.mean() * 252
    covariance_matrix = returns_data.cov() * 252
    constraints = OptimizationConstraints(min_weight=0.01, max_weight=max_weight)
    
    return analyzer.optimizer.optimize_portfolio(
        expected_returns, covariance_matrix, method, constraints
    )

def compare_optimization_strategies(returns_data: pd.DataFrame,
                                  methods: List[str] = None) -> pd.DataFrame:
    """Compare multiple optimization strategies."""
    if methods is None:
        methods = ['equal_weight', 'max_sharpe', 'min_variance', 'risk_parity']
    
    analyzer = IntegratedPortfolioAnalyzer()
    
    expected_returns = returns_data.mean() * 252
    covariance_matrix = returns_data.cov() * 252
    constraints = OptimizationConstraints(min_weight=0.01, max_weight=0.3)
    
    results = {}
    for method in methods:
        result = analyzer.optimizer.optimize_portfolio(
            expected_returns, covariance_matrix, method, constraints
        )
        results[method] = result
    
    return analyzer._create_comparative_analysis(
        results, None, expected_returns, covariance_matrix
    )