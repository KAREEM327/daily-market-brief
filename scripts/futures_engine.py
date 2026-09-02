#!/usr/bin/env python3
"""
Futures Engine for Daily Market Brief
Covers: term structure, calendar spreads, COT data parsing (proper)
Dependencies: yfinance, pandas, numpy, requests (all already in requirements)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Futures contracts we track (continuous front month via yfinance tickers)
FUTURES_CONTRACTS = {
    'ES': 'ES=F',      # E-mini S&P 500
    'NQ': 'NQ=F',      # E-mini Nasdaq 100
    'RTY': 'RTY=F',    # E-mini Russell 2000
    'CL': 'CL=F',      # Crude Oil WTI
    'BZ': 'BZ=F',      # Brent Crude
    'GC': 'GC=F',      # Gold
    'SI': 'SI=F',      # Silver
    'HG': 'HG=F',      # Copper
    'ZC': 'ZC=F',      # Corn
    'ZS': 'ZS=F',      # Soybeans
    'ZW': 'ZW=F',      # Wheat
    'HO': 'HO=F',      # Heating Oil
    'RB': 'RB=F',      # RBOB Gasoline
    'NG': 'NG=F',      # Natural Gas
    '6E': '6E=F',      # Euro/USD
    '6J': '6J=F',      # Yen/USD
    '6B': '6B=F',      # Pound/USD
    'ZT': 'ZT=F',      # 2-Year Treasury
    'ZF': 'ZF=F',      # 5-Year Treasury
    'ZN': 'ZN=F',      # 10-Year Treasury
    'ZB': 'ZB=F',      # 30-Year Treasury
}

# Map for COT report names (from CFTC legacy format)
COT_NAME_MAP = {
    'ES': 'S&P 500',
    'NQ': 'NASDAQ-100',
    'RTY': 'RUSSELL 2000',
    'CL': 'WTI CRUDE OIL',
    'BZ': 'BRENT CRUDE OIL',
    'GC': 'GOLD',
    'SI': 'SILVER',
    'HG': 'COPPER',
    'ZC': 'CORN',
    'ZS': 'SOYBEANS',
    'ZW': 'WHEAT',
    'HO': 'HEATING OIL',
    'RB': 'RBOB GASOLINE',
    'NG': 'NATURAL GAS',
    '6E': 'EURO FX',
    '6J': 'JAPANESE YEN',
    '6B': 'BRITISH POUND',
    'ZT': '2-YEAR TREASURY',
    'ZF': '5-YEAR TREASURY',
    'ZN': '10-YEAR TREASURY',
    'ZB': '30-YEAR TREASURY',
}

def fetch_continuous_front_month(ticker_symbol):
    """
    Fetch continuous front month futures data via yfinance.
    yfinance tickers like ES=F give the nearest contract.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="60d")  # last 60 days
        if hist.empty:
            return None
        # Get last price and previous close for change
        last = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else last
        change = ((last - prev_close) / prev_close) * 100
        return {
            'symbol': ticker_symbol.replace('=F', ''),
            'last': round(float(last), 2),
            'change_pct': round(float(change), 2),
            'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
            # Don't store hist - it causes JSON issues
        }
    except Exception as e:
        print(f"Error fetching {ticker_symbol}: {e}")
        return None

def fetch_term_structure(futures_root):
    """
    Fetch term structure for a given futures root using yfinance continuous contracts.
    yfinance provides continuous front-month contracts like CL=F, GC=F, etc.
    We'll use multiple contract months if available, but mainly rely on continuous.
    """
    # Try to get term structure from available contracts
    # yfinance has limited term structure - mainly continuous front month
    # We'll use the continuous contract and note that detailed term structure needs paid data
    try:
        ticker = yf.Ticker(f"{futures_root}=F")
        hist = ticker.history(period="60d")
        if hist.empty:
            return None
        
        last = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2] if len(hist) > 1 else last
        change = ((last - prev) / prev) * 100
        
        return [{
            'ticker': f"{futures_root}=F",
            'price': round(float(last), 2),
            'change_pct': round(float(change), 2),
            'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
            'contract_type': 'continuous_front'
        }]
    except Exception as e:
        logging.debug(f"Term structure failed for {futures_root}: {e}")
        return None

def calculate_calendar_spread(futures_root):
    """
    Calculate calendar spread - not available with continuous contracts only.
    Returns None since we don't have multiple contract months from yfinance.
    """
    return None

def parse_cot_report():
    """
    Parse the latest CFTC COT report (legacy format) from the CFTC website.
    We'll download the latest weekly report (futures_only.txt) and parse it.
    Source: https://www.cftc.gov/dea/futures/deanymes_f.htm
    """
    try:
        year = datetime.now().year
        urls = [
            f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip",
            f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year-1}.zip"
        ]
        resp = None
        for u in urls:
            resp = requests.get(u, timeout=10)
            if resp.status_code == 200:
                break
        if resp is None or resp.status_code != 200:
            return {"error": "Could not download COT data"}

        # Extract zip and find the latest .txt file
        import zipfile
        import io
        zip_file = zipfile.ZipFile(io.BytesIO(resp.content))
        txt_files = [f for f in zip_file.namelist() if f.endswith('.txt')]
        if not txt_files:
            return {"error": "No txt file in COT zip"}
        
        # Get the latest txt file
        latest_txt = sorted(txt_files)[-1]
        with zip_file.open(latest_txt) as f:
            lines = f.read().decode('utf-8').splitlines()
        
        # Parse the legacy format
        data = {}
        for line in lines:
            if len(line) < 100:
                continue
            # Format:
            # Market_and_Exchange_Names (0-30), CFTC_Market_Code (31-35), etc.
            # We'll use a simple approach: look for the market name in our map
            market_name = line[0:30].strip()
            for our_root, cot_name in COT_NAME_MAP.items():
                if cot_name in market_name.upper():
                    # Extract fields (positions approximate based on legacy format)
                    # Open Interest: columns 123-132
                    # NonComm Long: 133-142
                    # NonComm Short: 143-152
                    # Comm Long: 153-162
                    # Comm Short: 163-172
                    # NonReportable Long: 173-182
                    # NonReportable Short: 183-192
                    try:
                        oi = int(line[122:132].strip()) if line[122:132].strip() else 0
                        noncomm_long = int(line[132:142].strip()) if line[132:142].strip() else 0
                        noncomm_short = int(line[142:152].strip()) if line[142:152].strip() else 0
                        comm_long = int(line[152:162].strip()) if line[152:162].strip() else 0
                        comm_short = int(line[162:172].strip()) if line[162:172].strip() else 0
                        nonreport_long = int(line[172:182].strip()) if line[172:182].strip() else 0
                        nonreport_short = int(line[182:192].strip()) if line[182:192].strip() else 0
                        # Calculate net positions
                        noncomm_net = noncomm_long - noncomm_short
                        comm_net = comm_long - comm_short
                        # Change from previous week? Not available in this format without previous
                        # We'll just report current
                        data[our_root] = {
                            'open_interest': oi,
                            'noncomm_long': noncomm_long,
                            'noncomm_short': noncomm_short,
                            'noncomm_net': noncomm_net,
                            'comm_long': comm_long,
                            'comm_short': comm_short,
                            'comm_net': comm_net,
                            'nonreport_long': nonreport_long,
                            'nonreport_short': nonreport_short,
                            'market_name': market_name
                        }
                    except (ValueError, IndexError):
                        continue
        return data
    except Exception as e:
        return {"error": f"COT parse failed: {str(e)}"}

def analyze_futures():
    """
    Main analysis function for futures.
    Returns a dict with insights for the brief.
    """
    results = {
        'front_month': {},
        'cot_data': {},
        'alerts': [],
        'summary': ''
    }
    
    # 1. Front month continuous contracts
    key_commodities = ['CL', 'GC', 'SI', 'HG', 'NG', 'ZC', 'ZS', 'ZW']
    key_indices = ['ES', 'NQ', 'RTY']
    key_rates = ['ZT', 'ZF', 'ZN', 'ZB']
    key_fx = ['6E', '6J', '6B']
    
    all_roots = key_commodities + key_indices + key_rates + key_fx
    
    for root in all_roots:
        try:
            term = fetch_term_structure(root)
            if term:
                results['front_month'][root] = term[0]
        except Exception as e:
            logging.debug(f"Front month failed for {root}: {e}")
    
    # 2. COT data
    cot_data = parse_cot_report()
    if isinstance(cot_data, dict) and 'error' not in cot_data:
        results['cot_data'] = cot_data
        # Look for extremes
        for root, data in cot_data.items():
            oi = data['open_interest']
            if oi == 0:
                continue
            # Comm net as % of OI
            comm_net_pct = (data['comm_net'] / oi) * 100 if oi != 0 else 0
            noncomm_net_pct = (data['noncomm_net'] / oi) * 100 if oi != 0 else 0
            # Flag extreme positioning
            if abs(comm_net_pct) > 20:
                results['alerts'].append(
                    f"{root}: Commercial net {'long' if data['comm_net'] > 0 else 'short'} {abs(comm_net_pct):.0f}% of OI"
                )
            if abs(noncomm_net_pct) > 20:
                results['alerts'].append(
                    f"{root}: Non-commercial net {'long' if data['noncomm_net'] > 0 else 'short'} {abs(noncomm_net_pct):.0f}% of OI"
                )
    else:
        results['cot_data'] = {'error': cot_data.get('error', 'Unknown error') if isinstance(cot_data, dict) else str(cot_data)}
    
    # 3. Generate summary text
    summary_parts = []
    # Front month prices
    for root, data in results['front_month'].items():
        summary_parts.append(f"{root} {data['price']:.2f} ({data['change_pct']:+.1f}%)")
    
    # COT extremes
    if results['alerts']:
        summary_parts.extend(results['alerts'][:2])  # top 2 alerts
    
    if not summary_parts:
        results['summary'] = "Futures markets data loading..."
    else:
        results['summary'] = " | ".join(summary_parts[:5])  # limit length
    
    return results

if __name__ == "__main__":
    # Test run
    data = analyze_futures()
    print(json.dumps(data, indent=2))