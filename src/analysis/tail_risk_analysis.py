"""
Tail risk analysis module for portfolio risk assessment.

This module provides comprehensive tail risk metrics including VaR, CVaR, 
maximum drawdown, stress testing, and rolling volatility analysis to complement
the existing correlation structure analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class TailRiskAnalyzer:
    """
    Comprehensive tail risk analysis for portfolio returns.
    
    Provides Value at Risk, Conditional VaR, maximum drawdown analysis,
    stress testing against historical events, and rolling risk metrics.
    """
    
    def __init__(self):
        # Historical stress test scenarios (returns for major market events)
        self.stress_scenarios = {
            '2008_financial_crisis': {
                'name': '2008 Financial Crisis',
                'period': '2008-09-15 to 2009-03-09',
                'market_return': -0.567,  # S&P 500 peak to trough
                'description': 'Lehman Brothers collapse and financial system stress'
            },
            '2020_covid_crash': {
                'name': 'COVID-19 Market Crash',
                'period': '2020-02-19 to 2020-03-23',
                'market_return': -0.338,  # S&P 500 peak to trough
                'description': 'Pandemic-induced market sell-off'
            },
            '2018_volatility_spike': {
                'name': '2018 Volatility Spike',
                'period': '2018-01-26 to 2018-02-09',
                'market_return': -0.108,  # S&P 500 correction
                'description': 'VIX spike and algorithmic selling'
            },
            '2022_rate_shock': {
                'name': '2022 Rate Shock',
                'period': '2022-01-03 to 2022-06-17',
                'market_return': -0.238,  # S&P 500 bear market start
                'description': 'Federal Reserve aggressive rate hiking cycle'
            }
        }
    
    def analyze_tail_risk(self, returns_data: pd.DataFrame, 
                         weights: Optional[pd.Series] = None,
                         confidence_levels: List[float] = [0.95, 0.99, 0.995]) -> Dict[str, Any]:
        """
        Perform comprehensive tail risk analysis on portfolio returns.
        
        Args:
            returns_data: DataFrame with asset returns
            weights: Optional portfolio weights
            confidence_levels: Confidence levels for VaR/CVaR calculation
            
        Returns:
            Dictionary containing all tail risk metrics
        """
        # Calculate portfolio returns
        if weights is not None:
            portfolio_returns = (returns_data * weights).sum(axis=1)
        else:
            # Equal weights if not provided
            portfolio_returns = returns_data.mean(axis=1)
        
        results = {
            'analysis_successful': True,
            'portfolio_summary': self._calculate_return_summary(portfolio_returns),
            'var_cvar_analysis': self._calculate_var_cvar(portfolio_returns, confidence_levels),
            'drawdown_analysis': self._calculate_drawdown_metrics(portfolio_returns),
            'stress_testing': self._perform_stress_testing(returns_data, weights),
            'rolling_volatility': self._calculate_rolling_volatility(portfolio_returns),
            'tail_dependency': self._analyze_tail_dependency(returns_data),
            'extreme_value_analysis': self._perform_extreme_value_analysis(portfolio_returns),
            'risk_interpretation': []
        }
        
        # Generate comprehensive interpretation
        results['risk_interpretation'] = self._generate_risk_interpretation(results)
        results['advisor_risk_talking_points'] = self._generate_advisor_risk_points(results)
        
        return results
    
    def _calculate_return_summary(self, portfolio_returns: pd.Series) -> Dict[str, Any]:
        """Calculate basic return and volatility statistics."""
        return {
            'total_observations': len(portfolio_returns),
            'annualized_return': float(portfolio_returns.mean() * 252),
            'annualized_volatility': float(portfolio_returns.std() * np.sqrt(252)),
            'sharpe_ratio': float((portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))),
            'skewness': float(stats.skew(portfolio_returns)),
            'excess_kurtosis': float(stats.kurtosis(portfolio_returns)),
            'jarque_bera_test': float(stats.jarque_bera(portfolio_returns)[1])  # p-value
        }
    
    def _calculate_var_cvar(self, portfolio_returns: pd.Series, 
                           confidence_levels: List[float]) -> Dict[str, Any]:
        """
        Calculate Value at Risk and Conditional Value at Risk.
        
        Uses both historical simulation and parametric methods.
        """
        var_cvar_results = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            
            # Historical VaR (empirical quantile)
            historical_var = np.percentile(portfolio_returns, alpha * 100)
            
            # Historical CVaR (expected shortfall)
            tail_losses = portfolio_returns[portfolio_returns <= historical_var]
            historical_cvar = tail_losses.mean() if len(tail_losses) > 0 else historical_var
            
            # Parametric VaR (assuming normal distribution)
            mean_return = portfolio_returns.mean()
            std_return = portfolio_returns.std()
            parametric_var = stats.norm.ppf(alpha, mean_return, std_return)
            
            # Cornish-Fisher VaR (adjusted for skewness and kurtosis)
            skew = stats.skew(portfolio_returns)
            kurt = stats.kurtosis(portfolio_returns)
            z_score = stats.norm.ppf(alpha)
            
            # Cornish-Fisher adjustment
            cf_adjustment = (z_score + 
                           (z_score**2 - 1) * skew / 6 + 
                           (z_score**3 - 3*z_score) * kurt / 24 - 
                           (2*z_score**3 - 5*z_score) * skew**2 / 36)
            
            cornish_fisher_var = mean_return + std_return * cf_adjustment
            
            var_cvar_results[f'{confidence:.1%}'] = {
                'historical_var_daily': float(historical_var),
                'historical_cvar_daily': float(historical_cvar),
                'parametric_var_daily': float(parametric_var),
                'cornish_fisher_var_daily': float(cornish_fisher_var),
                'historical_var_annual': float(historical_var * np.sqrt(252)),
                'historical_cvar_annual': float(historical_cvar * np.sqrt(252)),
                'tail_observations': len(tail_losses)
            }
        
        return var_cvar_results
    
    def _calculate_drawdown_metrics(self, portfolio_returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate maximum drawdown and related metrics.
        """
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdowns.min()
        max_dd_date = drawdowns.idxmin()
        
        # Find peak before max drawdown
        max_dd_peak_date = running_max.loc[:max_dd_date].idxmax()
        
        # Recovery analysis
        recovery_date = None
        if max_dd_date < cumulative_returns.index[-1]:
            peak_value = running_max.loc[max_dd_date]
            post_trough = cumulative_returns.loc[max_dd_date:]
            recovery_mask = post_trough >= peak_value
            if recovery_mask.any():
                recovery_date = post_trough[recovery_mask].index[0]
        
        # Calculate drawdown duration
        dd_duration_days = (max_dd_date - max_dd_peak_date).days if max_dd_peak_date else None
        recovery_duration_days = (recovery_date - max_dd_date).days if recovery_date else None
        
        # Additional drawdown statistics
        negative_periods = drawdowns[drawdowns < 0]
        avg_drawdown = negative_periods.mean() if len(negative_periods) > 0 else 0
        
        # Calmar ratio (annual return / max drawdown)
        annual_return = portfolio_returns.mean() * 252
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'maximum_drawdown': float(max_drawdown),
            'max_drawdown_date': max_dd_date.strftime('%Y-%m-%d') if max_dd_date else None,
            'peak_date': max_dd_peak_date.strftime('%Y-%m-%d') if max_dd_peak_date else None,
            'recovery_date': recovery_date.strftime('%Y-%m-%d') if recovery_date else None,
            'drawdown_duration_days': dd_duration_days,
            'recovery_duration_days': recovery_duration_days,
            'average_drawdown': float(avg_drawdown),
            'calmar_ratio': float(calmar_ratio),
            'periods_in_drawdown': len(negative_periods),
            'total_periods': len(drawdowns)
        }
    
    def _perform_stress_testing(self, returns_data: pd.DataFrame, 
                               weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform stress testing against historical market scenarios.
        """
        stress_results = {}
        
        # Calculate portfolio beta to market (using correlation with first asset as proxy)
        if weights is not None:
            portfolio_returns = (returns_data * weights).sum(axis=1)
        else:
            portfolio_returns = returns_data.mean(axis=1)
        
        # Estimate portfolio beta (simplified - assume first asset represents market)
        market_proxy_returns = returns_data.iloc[:, 0]
        portfolio_beta = portfolio_returns.cov(market_proxy_returns) / market_proxy_returns.var()
        
        for scenario_id, scenario in self.stress_scenarios.items():
            # Apply scenario stress
            stressed_return = scenario['market_return'] * portfolio_beta
            
            # Calculate portfolio value impact
            portfolio_value_impact = stressed_return
            
            stress_results[scenario_id] = {
                'scenario_name': scenario['name'],
                'scenario_period': scenario['period'],
                'market_return': scenario['market_return'],
                'estimated_portfolio_return': float(stressed_return),
                'portfolio_value_impact': float(portfolio_value_impact),
                'description': scenario['description']
            }
        
        return stress_results
    
    def _calculate_rolling_volatility(self, portfolio_returns: pd.Series, 
                                    windows: List[int] = [21, 63, 252]) -> Dict[str, Any]:
        """
        Calculate rolling volatility metrics to identify regime changes.
        """
        rolling_vol_results = {}
        
        for window in windows:
            if len(portfolio_returns) >= window:
                rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
                
                rolling_vol_results[f'{window}_day'] = {
                    'window_days': window,
                    'current_volatility': float(rolling_vol.iloc[-1]) if not rolling_vol.empty else None,
                    'average_volatility': float(rolling_vol.mean()),
                    'max_volatility': float(rolling_vol.max()),
                    'min_volatility': float(rolling_vol.min()),
                    'volatility_percentile_current': float(rolling_vol.rank(pct=True).iloc[-1] * 100) if not rolling_vol.empty else None
                }
        
        return rolling_vol_results
    
    def _analyze_tail_dependency(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze tail dependency between assets.
        """
        n_assets = returns_data.shape[1]
        if n_assets < 2:
            return {'tail_dependency_available': False}
        
        # Calculate lower tail dependency (correlation in bottom 10% of returns)
        lower_threshold = returns_data.quantile(0.1)
        upper_threshold = returns_data.quantile(0.9)
        
        # Count simultaneous extreme events
        simultaneous_lower_tail = ((returns_data <= lower_threshold).sum(axis=1) >= 2).sum()
        simultaneous_upper_tail = ((returns_data >= upper_threshold).sum(axis=1) >= 2).sum()
        
        total_observations = len(returns_data)
        
        return {
            'tail_dependency_available': True,
            'simultaneous_lower_tail_events': int(simultaneous_lower_tail),
            'simultaneous_upper_tail_events': int(simultaneous_upper_tail),
            'lower_tail_dependency_ratio': float(simultaneous_lower_tail / total_observations),
            'upper_tail_dependency_ratio': float(simultaneous_upper_tail / total_observations),
            'total_observations': total_observations
        }
    
    def _perform_extreme_value_analysis(self, portfolio_returns: pd.Series) -> Dict[str, Any]:
        """
        Perform extreme value analysis using block maxima method.
        """
        # Split returns into blocks (monthly blocks)
        returns_monthly = portfolio_returns.resample('M').min()  # Monthly minimums (worst returns)
        
        if len(returns_monthly) < 12:
            return {'extreme_value_analysis_available': False}
        
        # Fit Generalized Extreme Value distribution
        try:
            # Fit GEV to monthly minimums (block minima for losses)
            gev_params = stats.genextreme.fit(-returns_monthly)  # Fit to negative returns (losses)
            
            # Estimate extreme quantiles
            extreme_1_year = -stats.genextreme.ppf(0.99, *gev_params)  # 1% annual probability
            extreme_10_year = -stats.genextreme.ppf(0.999, *gev_params)  # 0.1% annual probability
            
            return {
                'extreme_value_analysis_available': True,
                'gev_fitted': True,
                'estimated_1_year_worst_case': float(extreme_1_year),
                'estimated_10_year_worst_case': float(extreme_10_year),
                'monthly_observations': len(returns_monthly)
            }
        except:
            return {
                'extreme_value_analysis_available': True,
                'gev_fitted': False,
                'error': 'Could not fit GEV distribution'
            }
    
    def _generate_risk_interpretation(self, results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive risk interpretation."""
        interpretation = []
        
        # VaR/CVaR insights
        var_99 = results['var_cvar_analysis']['99.0%']['historical_var_daily']
        cvar_99 = results['var_cvar_analysis']['99.0%']['historical_cvar_daily']
        
        interpretation.append(f"Tail Risk Assessment:")
        interpretation.append(f"• 99% VaR: {var_99:.2%} daily loss expected 1% of the time")
        interpretation.append(f"• 99% CVaR: {cvar_99:.2%} average loss when exceeding VaR")
        
        # Drawdown insights
        max_dd = results['drawdown_analysis']['maximum_drawdown']
        dd_duration = results['drawdown_analysis']['drawdown_duration_days']
        recovery_duration = results['drawdown_analysis']['recovery_duration_days']
        
        interpretation.append(f"Drawdown Analysis:")
        interpretation.append(f"• Maximum drawdown: {max_dd:.1%}")
        if dd_duration:
            interpretation.append(f"• Drawdown lasted {dd_duration} days")
        if recovery_duration:
            interpretation.append(f"• Recovery took {recovery_duration} days")
        elif recovery_duration is None:
            interpretation.append(f"• Portfolio has not fully recovered from maximum drawdown")
        
        # Stress test insights
        worst_stress = min([s['estimated_portfolio_return'] for s in results['stress_testing'].values()])
        worst_scenario = [k for k, v in results['stress_testing'].items() 
                         if v['estimated_portfolio_return'] == worst_stress][0]
        
        interpretation.append(f"Stress Testing:")
        interpretation.append(f"• Worst scenario: {results['stress_testing'][worst_scenario]['scenario_name']}")
        interpretation.append(f"• Estimated loss: {worst_stress:.1%}")
        
        # Volatility regime
        vol_21d = results['rolling_volatility']['21_day']['current_volatility']
        vol_percentile = results['rolling_volatility']['21_day']['volatility_percentile_current']
        
        if vol_percentile:
            interpretation.append(f"Current Risk Environment:")
            interpretation.append(f"• Current volatility: {vol_21d:.1%} (annualized)")
            interpretation.append(f"• Volatility percentile: {vol_percentile:.0f}th percentile")
        
        return interpretation
    
    def _generate_advisor_risk_points(self, results: Dict[str, Any]) -> List[str]:
        """Generate advisor-ready risk talking points."""
        talking_points = []
        
        # Key risk headline
        var_95 = abs(results['var_cvar_analysis']['95.0%']['historical_var_daily'])
        max_dd = abs(results['drawdown_analysis']['maximum_drawdown'])
        
        talking_points.append(f"Risk Profile: 95% confident daily losses won't exceed {var_95:.1%}")
        talking_points.append(f"Historical Worst Case: {max_dd:.1%} maximum drawdown experienced")
        
        # Stress test summary
        stress_losses = [abs(s['estimated_portfolio_return']) for s in results['stress_testing'].values()]
        avg_stress_loss = np.mean(stress_losses)
        talking_points.append(f"Crisis Resilience: Average {avg_stress_loss:.1%} loss in major market stress scenarios")
        
        # Recovery characteristics
        recovery_days = results['drawdown_analysis']['recovery_duration_days']
        if recovery_days:
            talking_points.append(f"Recovery Profile: Historical recovery from major losses took {recovery_days} days")
        else:
            talking_points.append("Recovery Profile: Portfolio has experienced prolonged drawdown periods")
        
        # Current risk environment
        vol_percentile = results['rolling_volatility']['21_day'].get('volatility_percentile_current')
        if vol_percentile:
            if vol_percentile > 80:
                talking_points.append("Current Environment: Volatility in top 20% of historical range - elevated risk period")
            elif vol_percentile < 20:
                talking_points.append("Current Environment: Volatility in bottom 20% of historical range - calm market conditions")
            else:
                talking_points.append("Current Environment: Volatility in normal historical range")
        
        # Tail dependency warning
        if results['tail_dependency']['tail_dependency_available']:
            lower_tail_ratio = results['tail_dependency']['lower_tail_dependency_ratio']
            if lower_tail_ratio > 0.05:
                talking_points.append("Correlation Warning: Assets show elevated correlation during market stress")
        
        return talking_points


def integrate_tail_risk_with_correlation(correlation_results: Dict[str, Any], 
                                       tail_risk_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate tail risk analysis with existing correlation analysis results.
    
    Args:
        correlation_results: Results from portfolio correlation analysis
        tail_risk_results: Results from tail risk analysis
        
    Returns:
        Combined analysis results with integrated insights
    """
    integrated_results = {
        'correlation_analysis': correlation_results,
        'tail_risk_analysis': tail_risk_results,
        'integrated_insights': []
    }
    
    # Generate integrated insights
    insights = []
    
    # Correlation-risk relationship
    diversification_loss = correlation_results.get('diversification_loss', 0)
    max_drawdown = abs(tail_risk_results['drawdown_analysis']['maximum_drawdown'])
    
    if diversification_loss > 0.6 and max_drawdown > 0.3:
        insights.append("High correlation concentration amplifies tail risk - portfolio vulnerable to systematic shocks")
    elif diversification_loss > 0.4 and max_drawdown > 0.2:
        insights.append("Moderate correlation concentration increases downside risk beyond individual asset volatility")
    
    # Effective rank vs tail events
    effective_rank = correlation_results.get('effective_rank', 0)
    var_99 = abs(tail_risk_results['var_cvar_analysis']['99.0%']['historical_var_daily'])
    
    if effective_rank < 3 and var_99 > 0.03:
        insights.append(f"Low effective rank ({effective_rank:.1f}) consistent with high tail risk (99% VaR: {var_99:.1%})")
    
    # Factor concentration vs stress testing
    if correlation_results.get('mp_fitting_successful'):
        num_signal_factors = correlation_results.get('num_signal_factors', 0)
        stress_losses = [abs(s['estimated_portfolio_return']) for s in tail_risk_results['stress_testing'].values()]
        avg_stress_loss = np.mean(stress_losses)
        
        if num_signal_factors <= 2 and avg_stress_loss > 0.25:
            insights.append(f"Few risk factors ({num_signal_factors}) create vulnerability to systematic stress scenarios")
    
    integrated_results['integrated_insights'] = insights
    
    return integrated_results