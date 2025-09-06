export interface Portfolio {
  id: string;
  name: string;
  description?: string;
  tickers: string[];
  weights?: number[];
  benchmark_ticker: string;
  created_at: string;
  updated_at: string;
}

export interface AnalysisResult {
  portfolio_id: string;
  analysis_types: string[];
  computation_time: number;
  results: {
    metadata: {
      portfolio_tickers: string[];
      weights: number[];
      failed_tickers: string[];
      benchmark_ticker: string;
      analysis_period: {
        start: string;
        end: string;
      };
      total_observations: number;
    };
    basic?: BasicAnalysis;
    correlation?: CorrelationAnalysis;
    monte_carlo?: MonteCarloAnalysis;
  };
}

export interface BasicAnalysis {
  annual_return: number;
  annual_volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  var_95: number;
  var_99: number;
  cvar_95: number;
  skewness: number;
  kurtosis: number;
  total_return: number;
  win_rate: number;
  benchmark_comparison?: {
    beta: number;
    alpha: number;
    correlation: number;
    information_ratio: number;
    tracking_error: number;
    excess_return: number;
  };
}

export interface CorrelationAnalysis {
  effective_rank: number;
  concentration_ratio: number;
  condition_number: number;
  eigenvalues: number[];
  explained_variance_ratios: number[];
  portfolio_concentration: {
    average_correlation: number;
    diversification_ratio: number;
    portfolio_variance: number;
  };
}

export interface MonteCarloAnalysis {
  parameters: {
    annual_return: number;
    annual_volatility: number;
    years: number;
    simulations: number;
  };
  normal_simulation: {
    median: number;
    percentile_5: number;
    percentile_95: number;
    probability_positive: number;
    expected_value: number;
  };
  bootstrap_simulation: {
    median: number;
    percentile_5: number;
    percentile_95: number;
    probability_positive: number;
    expected_value: number;
  };
}