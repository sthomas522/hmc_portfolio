# Portfolio Correlation Analysis

Advanced portfolio diversification analysis using eigenvalue decomposition and Random Matrix Theory validation.

## Overview

This system analyzes portfolio correlation structures to reveal hidden concentration risks that traditional metrics miss. Using mathematical techniques from Random Matrix Theory, it distinguishes genuine risk factors from random correlation noise, providing institutional-quality insights for investment advisors.

**Key Insight**: Many portfolios that appear diversified actually behave like far fewer independent investments due to hidden correlations.

## Features

### Core Analysis
- **Effective Rank Calculation**: Determines how many truly independent assets your portfolio contains
- **Diversification Loss Metrics**: Quantifies the gap between apparent and actual diversification
- **Marchenko-Pastur Validation**: Uses Random Matrix Theory to separate signal from noise in correlations
- **Risk Concentration Analysis**: Identifies dominant risk factors driving portfolio volatility

### Professional Output
- **Advisor Talking Points**: Ready-to-use insights for client presentations
- **Scientific Validation**: Institutional-quality mathematical backing
- **Quality Assessment**: Confidence metrics for analysis results

## Technology Stack

- **Backend**: FastAPI with uvicorn
- **Analysis**: NumPy, Pandas for mathematical computations
- **Data**: yfinance for market data
- **Frontend**: Vanilla JavaScript with modern CSS
- **Configuration**: Pydantic settings with environment support

## Quick Start

### Prerequisites
- Python 3.11+
- uv package manager (or pip)

### Installation

1. **Clone the repository:**
```bash
git clone <repo-url>
cd portfolio-analysis
```

2. **Install dependencies:**
```bash
uv install
# or: pip install fastapi uvicorn pandas numpy yfinance pydantic-settings
```

3. **Start the API server:**
```bash
cd src
uv run python api/main.py
```

4. **Open the frontend:**
```bash
# Open frontend/index.html in your browser
# or serve with a simple HTTP server:
python -m http.server 8080 --directory frontend
```

### Access Points
- **Frontend Interface**: `frontend/index.html` (or http://localhost:8080)
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## Usage

### Web Interface
1. Open the frontend in your browser
2. Choose input method:
   - **Ticker List**: Enter comma-separated symbols (e.g., "AAPL,MSFT,GOOGL")
   - **CSV Upload**: Upload holdings file with Ticker and Portfolio Weight columns
   - **Sample Portfolio**: Test with pre-configured tech portfolio
3. Review analysis results and advisor talking points

### API Endpoints
- `POST /analyze/tickers` - Analyze portfolio from ticker list
- `POST /analyze/csv` - Analyze uploaded CSV file  
- `GET /analyze/sample` - Demo with sample portfolio
- `GET /validate/tickers` - Check ticker validity
- `GET /config` - View API configuration

### Example Analysis Result
```json
{
  "effective_rank": 3.2,
  "diversification_loss": 0.68,
  "num_assets": 10,
  "noise_fraction": 0.90,
  "num_signal_factors": 1,
  "interpretation": [
    "Portfolio acts like 3.2 independent investments despite holding 10 assets",
    "68% diversification loss due to correlation concentration"
  ]
}
```

## Project Structure

```
portfolio-analysis/
├── src/
│   ├── analysis/
│   │   ├── mp_analysis.py           # Marchenko-Pastur implementation
│   │   ├── correlation_analysis.py  # Core correlation functions
│   │   └── portfolio_analyzer.py    # Main analysis orchestrator
│   ├── data/
│   │   └── data_service.py          # Market data fetching
│   └── api/
│       └── main.py                  # FastAPI application
├── frontend/
│   └── index.html                   # Web interface
├── tests/
│   └── test_integration.py          # System tests
├── archive/                         # Development history
├── config.py                        # Configuration management
└── README.md
```

## Mathematical Foundation

### Effective Rank
Calculated as the participation ratio of eigenvalue weights:
```
effective_rank = 1 / Σ(λᵢ/Σλⱼ)²
```
Where λᵢ are the eigenvalues of the correlation matrix.

### Marchenko-Pastur Theory
For random correlation matrices with ratio q = N/T:
- Eigenvalues should fall within bounds: σ²(1 ± √q)²
- Values outside bounds indicate genuine risk factors
- Values within bounds represent random noise

### Diversification Loss
```
diversification_loss = 1 - (effective_rank / num_assets)
```

## Configuration

Create a `.env` file in the project root:
```bash
ENVIRONMENT=development
PORTFOLIO_DEBUG=true
PORTFOLIO_MAX_ASSETS_DEFAULT=50
PORTFOLIO_LOG_LEVEL=INFO
```

## Testing

Run the integration test:
```bash
python tests/test_integration.py
```

Test individual API endpoints:
```bash
curl http://localhost:8000/analyze/sample
```

## Limitations

- **Minimum Assets**: Requires at least 3 assets for meaningful analysis
- **Data Dependency**: Relies on yfinance for market data
- **Time Series Length**: Needs sufficient observations (typically 2-3x number of assets)
- **Market Hours**: Some data may be delayed or unavailable outside market hours

## Use Cases

### Financial Advisors
- Demonstrate hidden concentration risks to clients
- Validate diversification claims with mathematical proof
- Generate professional analysis reports

### Portfolio Managers
- Quantify true diversification in existing portfolios
- Identify dominant risk factors
- Support risk management decisions

### Academic Research
- Study correlation structures in financial markets
- Validate Random Matrix Theory applications
- Benchmark portfolio concentration metrics

## Development

### Adding New Analysis Features
1. Implement mathematical functions in `src/analysis/`
2. Add API endpoints in `src/api/main.py`
3. Update frontend interface as needed
4. Add tests to verify functionality

### Deployment
The system is designed for easy deployment to cloud platforms:
- API can be containerized with Docker
- Frontend can be served as static files
- No database required (stateless design)

## License

[Add your license information]

## Contributing

[Add contribution guidelines]

## Support

For questions about the mathematical methodology, refer to Random Matrix Theory literature. For technical issues, check the API documentation at `/docs` endpoint.