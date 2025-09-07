#!/usr/bin/env python3
"""
Data Pipeline Diagnostics - Find and fix data loading issues
"""

import csv
import re
import pandas as pd
import yfinance as yf
from collections import Counter
import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.append(str(backend_path))

from app.services.data_service import DataService

def load_and_diagnose_vsmpx():
    """Load VSMPX holdings with detailed diagnostics"""
    
    print("🔍 VSMPX Data Pipeline Diagnostics")
    print("=" * 60)
    
    csv_file = 'Holdings_details_Total_Stock_Market_Index_Fund_Institutional_Plus_Shares.csv'
    
    # Read raw CSV
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    print(f"📄 Raw CSV: {len(rows)} total rows")
    
    # Find header row
    header_row_idx = None
    for i, row in enumerate(rows):
        if 'TICKER' in row and '% OF FUNDS*' in row:
            header_row_idx = i
            break
    
    if header_row_idx is None:
        print("❌ Could not find header row")
        return None
    
    print(f"📋 Header found at row {header_row_idx + 1}")
    headers = rows[header_row_idx]
    print(f"📊 Columns: {headers}")
    
    # Extract all potential tickers
    raw_tickers = []
    valid_holdings = []
    invalid_rows = []
    
    for i in range(header_row_idx + 1, len(rows)):
        row = rows[i]
        if len(row) >= 9:
            ticker = row[3].strip() if row[3] else ""
            company = row[2] if len(row) > 2 else ""
            percent = row[4] if len(row) > 4 else ""
            
            raw_tickers.append(ticker)
            
            # Validate ticker
            if ticker and len(ticker) <= 6 and re.match(r'^[A-Z0-9.-]+$', ticker):
                try:
                    percent_val = float(percent.replace('%', '')) if percent else 0
                    valid_holdings.append({
                        'ticker': ticker,
                        'company': company,
                        'percent': percent_val,
                        'raw_row': i + 1
                    })
                except ValueError:
                    invalid_rows.append((i + 1, ticker, "Invalid percent"))
            else:
                invalid_rows.append((i + 1, ticker, "Invalid ticker format"))
    
    print(f"\n📈 Extraction Results:")
    print(f"• Raw ticker entries: {len(raw_tickers)}")
    print(f"• Valid holdings: {len(valid_holdings)}")
    print(f"• Invalid rows: {len(invalid_rows)}")
    
    # Analyze ticker patterns
    print(f"\n🔍 Ticker Analysis:")
    ticker_lengths = Counter(len(t) for t in raw_tickers if t)
    print(f"• Ticker lengths: {dict(ticker_lengths)}")
    
    # Show problematic tickers
    problematic = [t for t in raw_tickers if t and (len(t) > 5 or not re.match(r'^[A-Z0-9.-]*$', t))]
    if problematic:
        print(f"• Problematic tickers (first 10): {problematic[:10]}")
    
    # Show invalid rows sample
    if invalid_rows:
        print(f"\n❌ Invalid Rows (first 5):")
        for row_num, ticker, reason in invalid_rows[:5]:
            print(f"   Row {row_num}: '{ticker}' - {reason}")
    
    return valid_holdings

def test_ticker_availability(holdings, sample_size=100):
    """Test which tickers are available in yfinance"""
    
    print(f"\n🧪 Testing Ticker Availability (sample of {sample_size})")
    print("=" * 60)
    
    # Test a sample of tickers
    test_tickers = [h['ticker'] for h in holdings[:sample_size]]
    
    available = []
    unavailable = []
    problematic = []
    
    print("Testing tickers...")
    for i, ticker in enumerate(test_tickers):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(test_tickers)}")
        
        try:
            # Quick test download
            data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if not data.empty and 'Close' in data.columns:
                available.append(ticker)
            else:
                unavailable.append(ticker)
        except Exception as e:
            problematic.append((ticker, str(e)))
    
    print(f"\n📊 Availability Results:")
    print(f"• Available: {len(available)}/{len(test_tickers)} ({len(available)/len(test_tickers)*100:.1f}%)")
    print(f"• Unavailable: {len(unavailable)}")
    print(f"• Errors: {len(problematic)}")
    
    if unavailable:
        print(f"\n❌ Unavailable tickers (first 10): {unavailable[:10]}")
    
    if problematic:
        print(f"\n⚠️  Problematic tickers (first 5):")
        for ticker, error in problematic[:5]:
            print(f"   {ticker}: {error}")
    
    return available, unavailable, problematic

async def test_data_service_performance(holdings):
    """Test your DataService with different batch sizes"""
    
    print(f"\n⚡ DataService Performance Test")
    print("=" * 60)
    
    ds = DataService()
    
    # Test different portfolio sizes
    test_sizes = [10, 25, 50, 100]
    
    for size in test_sizes:
        tickers = [h['ticker'] for h in holdings[:size]]
        
        print(f"\n🔬 Testing {size} tickers...")
        try:
            price_data, failed_tickers, metadata = await ds.fetch_portfolio_data(tickers)
            
            success_rate = len(metadata['successful_tickers']) / len(tickers)
            
            print(f"   ✅ Success: {len(metadata['successful_tickers'])}/{len(tickers)} ({success_rate:.1%})")
            print(f"   📅 Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")
            print(f"   📊 Observations: {metadata['total_observations']}")
            
            if failed_tickers:
                print(f"   ❌ Failed: {failed_tickers[:5]}{'...' if len(failed_tickers) > 5 else ''}")
                
        except Exception as e:
            print(f"   💥 Error: {e}")

def improved_ticker_cleaning(raw_ticker):
    """Improved ticker cleaning and validation"""
    
    if not raw_ticker:
        return None
    
    # Clean the ticker
    ticker = raw_ticker.strip().upper()
    
    # Handle common variations
    ticker_mappings = {
        'BRK.B': 'BRK-B',  # Berkshire class B
        'BF.B': 'BF-B',    # Brown-Forman class B
    }
    
    if ticker in ticker_mappings:
        ticker = ticker_mappings[ticker]
    
    # Validate format
    if not re.match(r'^[A-Z0-9.-]{1,5}$', ticker):
        return None
    
    # Skip known problematic patterns
    skip_patterns = [
        r'^\d+$',  # Pure numbers
        r'^[.-]+$',  # Just dots/dashes
    ]
    
    for pattern in skip_patterns:
        if re.match(pattern, ticker):
            return None
    
    return ticker

def create_improved_holdings_loader():
    """Create an improved version of holdings loading"""
    
    print(f"\n🔧 Creating Improved Holdings Loader")
    print("=" * 60)
    
    csv_file = 'Holdings_details_Total_Stock_Market_Index_Fund_Institutional_Plus_Shares.csv'
    
    improved_holdings = []
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    # Find header
    header_row_idx = None
    for i, row in enumerate(rows):
        if 'TICKER' in row and '% OF FUNDS*' in row:
            header_row_idx = i
            break
    
    if header_row_idx is None:
        print("❌ Could not find header")
        return []
    
    # Improved extraction
    for i in range(header_row_idx + 1, len(rows)):
        row = rows[i]
        if len(row) >= 9:
            raw_ticker = row[3]
            clean_ticker = improved_ticker_cleaning(raw_ticker)
            
            if clean_ticker:
                try:
                    percent_val = float(row[4].replace('%', '')) if row[4] else 0
                    improved_holdings.append({
                        'ticker': clean_ticker,
                        'company': row[2],
                        'percent': percent_val,
                        'sub_industry': row[5] if len(row) > 5 else '',
                        'market_value': row[8] if len(row) > 8 else '',
                        'shares': row[9] if len(row) > 9 else ''
                    })
                except (ValueError, IndexError):
                    continue
    
    print(f"✅ Improved loader extracted {len(improved_holdings)} holdings")
    
    # Show improvement
    original_count = 815  # From your previous run
    improvement = len(improved_holdings) - original_count
    print(f"📈 Improvement: +{improvement} holdings ({improvement/original_count*100:.1f}% increase)")
    
    return improved_holdings

async def main():
    """Run complete data pipeline diagnostics"""
    
    # Step 1: Diagnose current data loading
    holdings = load_and_diagnose_vsmpx()
    
    if not holdings:
        print("❌ Could not load holdings data")
        return
    
    # Step 2: Test ticker availability
    available, unavailable, problematic = test_ticker_availability(holdings, sample_size=50)
    
    # Step 3: Test your DataService
    await test_data_service_performance(holdings)
    
    # Step 4: Create improved loader
    improved_holdings = create_improved_holdings_loader()
    
    # Step 5: Test improved loader
    if improved_holdings:
        print(f"\n🧪 Testing Improved Loader")
        print("=" * 60)
        await test_data_service_performance(improved_holdings)
    
    print(f"\n🎯 DIAGNOSIS COMPLETE")
    print("=" * 60)
    print("1. ✅ Identified data extraction issues")
    print("2. ✅ Tested ticker availability") 
    print("3. ✅ Measured DataService performance")
    print("4. ✅ Created improved holdings loader")
    print("\nNext: Implement numerical stability fixes!")

if __name__ == "__main__":
    asyncio.run(main())