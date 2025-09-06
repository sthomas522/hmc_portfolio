#!/usr/bin/env python3
"""
Fixed VSMPX data loader that handles the specific ticker format issues identified
"""

import csv
import re
from typing import List, Dict, Any

def enhanced_ticker_cleaning(raw_ticker: str) -> str:
    """Enhanced ticker cleaning that handles the specific issues found"""
    
    if not raw_ticker or not isinstance(raw_ticker, str):
        return None
    
    ticker = raw_ticker.strip().upper()
    
    # Handle the primary issue: slash to dash conversion
    if '/' in ticker:
        ticker = ticker.replace('/', '-')
    
    # Handle other common variations
    ticker_mappings = {
        'BRK.B': 'BRK-B',
        'BRK.A': 'BRK-A',
        'BF.B': 'BF-B',
        'BF.A': 'BF-A',
        'HEI.A': 'HEI-A',
        'MOG.A': 'MOG-A',
        'UHAL.B': 'UHAL-B',
        'LEN.B': 'LEN-B',
        'CWEN.A': 'CWEN-A',
        'CRD.B': 'CRD-B'
    }
    
    if ticker in ticker_mappings:
        ticker = ticker_mappings[ticker]
    
    # Validate final ticker format
    if not re.match(r'^[A-Z0-9.-]{1,6}$', ticker):
        return None
    
    # Skip problematic patterns
    skip_patterns = [
        r'^\d+$',      # Pure numbers
        r'^[.-]+$',    # Just dots/dashes
        r'^[^A-Z]',    # Doesn't start with letter
    ]
    
    for pattern in skip_patterns:
        if re.match(pattern, ticker):
            return None
    
    return ticker

def enhanced_percent_parsing(raw_percent: str) -> float:
    """Enhanced percent parsing with better error handling"""
    
    if not raw_percent:
        return 0.0
    
    try:
        # Remove % symbol and any extra whitespace
        clean_percent = str(raw_percent).strip().replace('%', '')
        
        # Handle empty strings after cleaning
        if not clean_percent:
            return 0.0
        
        # Handle common formatting issues
        clean_percent = clean_percent.replace(',', '')  # Remove commas
        clean_percent = clean_percent.replace('$', '')  # Remove dollar signs
        
        return float(clean_percent)
        
    except (ValueError, TypeError):
        return 0.0

def load_vsmpx_holdings_fixed() -> List[Dict[str, Any]]:
    """Load VSMPX holdings with all identified fixes applied"""
    
    print("🔧 Loading VSMPX Holdings with Enhanced Fixes")
    print("=" * 60)
    
    csv_file = 'Holdings_details_Total_Stock_Market_Index_Fund_Institutional_Plus_Shares.csv'
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)
    except FileNotFoundError:
        print(f"❌ Could not find {csv_file}")
        return []
    
    print(f"📄 Loaded {len(rows)} rows from CSV")
    
    # Find header row
    header_idx = None
    for i, row in enumerate(rows):
        if len(row) > 3 and 'TICKER' in row and '% OF FUNDS*' in row:
            header_idx = i
            break
    
    if header_idx is None:
        print("❌ Could not find header row")
        return []
    
    print(f"📋 Found header at row {header_idx + 1}")
    print(f"📊 Columns: {rows[header_idx]}")
    
    # Process holdings with enhanced cleaning
    holdings = []
    stats = {
        'total_processed': 0,
        'valid_tickers': 0,
        'fixed_slashes': 0,
        'fixed_percents': 0,
        'skipped_invalid': 0
    }
    
    for i in range(header_idx + 1, len(rows)):
        row = rows[i]
        if len(row) >= 9:
            stats['total_processed'] += 1
            
            raw_ticker = row[3] if len(row) > 3 else ""
            raw_percent = row[4] if len(row) > 4 else ""
            
            # Apply enhanced ticker cleaning
            original_ticker = raw_ticker
            clean_ticker = enhanced_ticker_cleaning(raw_ticker)
            
            if clean_ticker:
                # Track fixes applied
                if '/' in original_ticker:
                    stats['fixed_slashes'] += 1
                
                # Apply enhanced percent parsing
                original_percent = raw_percent
                percent_val = enhanced_percent_parsing(raw_percent)
                
                if original_percent != str(percent_val) + '%':
                    stats['fixed_percents'] += 1
                
                holdings.append({
                    'ticker': clean_ticker,
                    'company': row[2] if len(row) > 2 else "",
                    'percent': percent_val,
                    'sub_industry': row[5] if len(row) > 5 else "",
                    'country': row[6] if len(row) > 6 else "US",
                    'market_value': row[8] if len(row) > 8 else "",
                    'shares': row[9] if len(row) > 9 else "",
                    'original_ticker': original_ticker  # Keep for debugging
                })
                stats['valid_tickers'] += 1
            else:
                stats['skipped_invalid'] += 1
    
    # Sort by percentage weight
    holdings.sort(key=lambda x: x['percent'], reverse=True)
    
    # Display results
    print(f"\n📈 Processing Results:")
    print(f"• Total rows processed: {stats['total_processed']}")
    print(f"• Valid tickers extracted: {stats['valid_tickers']}")
    print(f"• Slash-to-dash fixes: {stats['fixed_slashes']}")
    print(f"• Percent parsing fixes: {stats['fixed_percents']}")
    print(f"• Invalid rows skipped: {stats['skipped_invalid']}")
    
    success_rate = stats['valid_tickers'] / stats['total_processed'] if stats['total_processed'] > 0 else 0
    print(f"• Success rate: {success_rate:.1%}")
    
    improvement = stats['valid_tickers'] - 815  # Previous count
    print(f"• Improvement: +{improvement} holdings ({improvement/815*100:.1f}% increase)")
    
    # Show examples of fixes
    if stats['fixed_slashes'] > 0:
        print(f"\n🔧 Examples of Slash Fixes:")
        slash_examples = [h for h in holdings if '/' in h['original_ticker']][:5]
        for example in slash_examples:
            print(f"   {example['original_ticker']} → {example['ticker']}")
    
    # Show top holdings
    print(f"\n📊 Top 10 Holdings After Fixes:")
    for i, holding in enumerate(holdings[:10], 1):
        print(f"   {i:2d}. {holding['ticker']:<6} {holding['percent']:5.2f}% - {holding['company']}")
    
    return holdings

def validate_ticker_fixes():
    """Validate that our ticker fixes work with yfinance"""
    
    print(f"\n🧪 Validating Ticker Fixes with yfinance")
    print("=" * 60)
    
    # Test problematic tickers that we should have fixed
    test_cases = [
        ('BRK/B', 'BRK-B'),
        ('BRK/A', 'BRK-A'), 
        ('BF/B', 'BF-B'),
        ('HEI/A', 'HEI-A'),
        ('MOG/A', 'MOG-A')
    ]
    
    import yfinance as yf
    
    print("Testing ticker format fixes...")
    
    for original, fixed in test_cases:
        try:
            # Test if the fixed ticker works
            data = yf.download(fixed, period="5d", progress=False)
            if not data.empty:
                print(f"   ✅ {original} → {fixed} (works)")
            else:
                print(f"   ❌ {original} → {fixed} (no data)")
        except Exception as e:
            print(f"   ⚠️  {original} → {fixed} (error: {str(e)[:50]})")

async def test_fixed_loader_performance():
    """Test the fixed loader with your analysis service"""
    
    print(f"\n⚡ Testing Fixed Loader with Analysis Service")
    print("=" * 60)
    
    # Load holdings with fixes
    holdings = load_vsmpx_holdings_fixed()
    
    if not holdings:
        print("❌ No holdings loaded")
        return
    
    # Test with your DataService
    import sys
    from pathlib import Path
    backend_path = Path(__file__).parent / 'backend'
    sys.path.append(str(backend_path))
    
    from app.services.data_service import DataService
    
    ds = DataService()
    test_sizes = [10, 25, 50, 100]
    
    print(f"Testing {len(test_sizes)} portfolio sizes...")
    
    for size in test_sizes:
        if size > len(holdings):
            continue
            
        tickers = [h['ticker'] for h in holdings[:size]]
        
        try:
            price_data, failed_tickers, metadata = await ds.fetch_portfolio_data(tickers)
            
            success_rate = len(metadata['successful_tickers']) / len(tickers)
            
            print(f"   Size {size:3d}: {success_rate:.1%} success ({len(metadata['successful_tickers'])}/{len(tickers)})")
            
            if failed_tickers:
                print(f"            Failed: {failed_tickers[:3]}{'...' if len(failed_tickers) > 3 else ''}")
                
        except Exception as e:
            print(f"   Size {size:3d}: Failed - {e}")

def main():
    """Run the fixed data loader test"""
    
    print("🚀 Testing Fixed VSMPX Data Loader")
    print("=" * 70)
    
    # Load with fixes
    holdings = load_vsmpx_holdings_fixed()
    
    # Validate ticker fixes
    validate_ticker_fixes()
    
    # Test performance
    import asyncio
    asyncio.run(test_fixed_loader_performance())
    
    print(f"\n✅ FIXED DATA LOADER TEST COMPLETE!")
    print(f"Expected improvement: ~2000+ additional holdings vs previous 815")

if __name__ == "__main__":
    main()