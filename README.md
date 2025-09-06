# Portfolio Analyzer MVP

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose

### Setup

1. **Clone and setup backend:**
```bash
git clone <repo>
cd portfolio-analyzer/backend
cp .env.example .env
# Edit .env with your settings
pip install -r requirements.txt
```

2. **Setup frontend:**
```bash
cd ../frontend
npm install
```

3. **Start with Docker Compose:**
```bash
cd ..
docker-compose up -d
```

4. **Run database migrations:**
```bash
docker-compose exec backend alembic upgrade head
```

### Access
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### First Steps
1. Register a user at http://localhost:3000/register
2. Create a portfolio with tickers like ["AAPL", "MSFT", "GOOGL"]
3. Run analysis to see correlation analysis and Monte Carlo projections

## Features

### Current (MVP)
- Portfolio creation and management
- Multi-provider data fetching with caching
- Standard portfolio metrics (Sharpe, max drawdown, etc.)
- Correlation analysis with eigenvalue decomposition
- Monte Carlo simulations (normal + bootstrap)
- Benchmark comparison (beta, alpha, tracking error)

### Coming Next
- Advanced visualizations (D3.js charts)
- PDF report generation
- Regime-switching models
- Bayesian portfolio optimization
- Factor attribution analysis