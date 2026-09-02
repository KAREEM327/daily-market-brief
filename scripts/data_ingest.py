"""
Data Ingestion Layer — all free sources, no paid APIs.
Fetches macro, equities, options, internals, sentiment, foreign, smart money.
Outputs Parquet files to local data lake for downstream engines.
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import duckdb
from bs4 import BeautifulSoup

# ─── Config ──────────────────────────────────────────────────────────────
DATA_ROOT = Path(os.getenv("DMB_DATA_ROOT", Path(__file__).parent.parent / "data"))
RAW_DIR = DATA_ROOT / "raw"
PARQUET_DIR = DATA_ROOT / "parquet"
DUCKDB_PATH = DATA_ROOT / "market.duckdb"

for d in (RAW_DIR, PARQUET_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmb.ingest")

# ─── Free API Keys (from env) ────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# ─── Ticker Universes ────────────────────────────────────────────────────
MACRO_TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "DIA": "Dow Jones",
    "VIX": "VIX Index",
    "VX3M": "VIX 3M Futures",
    "VXMT": "VIX 6M Futures",
    "^TNX": "10Y Yield",
    "^TYX": "30Y Yield",
    "^FVX": "5Y Yield",
    "^IRX": "3M Yield",
    "DXY": "Dollar Index",
    "EURUSD=X": "EUR/USD",
    "USDJPY=X": "USD/JPY",
    "GC=F": "Gold",
    "CL=F": "Crude Oil (WTI)",
    "BZ=F": "Brent Crude",
    "HG=F": "Copper",
    "TLT": "20Y+ Treasuries",
    "IEF": "7-10Y Treasuries",
    "SHY": "1-3Y Treasuries",
    "LQD": "Investment Grade Corps",
    "HYG": "High Yield Corps",
    "EWG": "Germany",
    "EWJ": "Japan",
    "EEM": "Emerging Markets",
    "FXI": "China",
    "EWZ": "Brazil",
    "EWA": "Australia",
    "EWC": "Canada",
    "XLB": "Materials",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLC": "Communication Services",
    "XRT": "Retail",
    "XBI": "Biotech",
    "KRE": "Regional Banks",
    "SMH": "Semiconductors",
    "SOXX": "Semiconductors (alt)",
    "ARKK": "Innovation",
    "QQQ": "Nasdaq 100",
}

FRED_SERIES = {
    "DGS10": "10Y Treasury",
    "DGS2": "2Y Treasury",
    "T10Y2Y": "10Y-2Y Spread",
    "T10Y3M": "10Y-3M Spread",
    "DCOILWTICO": "WTI Crude",
    "DTWEXBGS": "Trade Weighted USD",
    "VIXCLS": "VIX Close",
    "BAMLH0A0HYM2": "HY Spread",
    "BAMLC0A0CM": "IG Spread",
    "TEDRATE": "TED Spread",
    "UNRATE": "Unemployment",
    "CPIAUCSL": "CPI",
    "FEDFUNDS": "Fed Funds Rate",
    "M2SL": "M2 Money Supply",
    "WALCL": "Fed Balance Sheet",
    "H8B1058NCBCHG": "Commercial Loans",
    "H8B1036NCBCHG": "Real Estate Loans",
}

SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLC"]

# ─── Helpers ─────────────────────────────────────────────────────────────
def save_parquet(df: pd.DataFrame, name: str, partition_cols: Optional[List[str]] = None) -> Path:
    """Save DataFrame as partitioned Parquet."""
    path = PARQUET_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if partition_cols:
        df.to_parquet(path, partition_cols=partition_cols, index=False)
    else:
        df.to_parquet(path, index=False)
    log.info(f"Saved {len(df)} rows to {path}")
    return path


def fetch_yfinance_batch(tickers: List[str], period: str = "2y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """Fetch multiple tickers from yfinance efficiently."""
    results = {}
    # yfinance handles ~50 tickers well in one call
    try:
        data = yf.download(tickers, period=period, interval=interval, group_by="ticker", auto_adjust=True, progress=False, threads=True)
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers:
                if ticker in data.columns.get_level_values(0):
                    df = data[ticker].dropna()
                    if not df.empty:
                        df = df.reset_index()
                        df["ticker"] = ticker
                        results[ticker] = df
        else:
            # Single ticker
            df = data.dropna().reset_index()
            df["ticker"] = tickers[0]
            results[tickers[0]] = df
    except Exception as e:
        log.error(f"yfinance batch error: {e}")
        # Fallback: individual
        for t in tickers:
            try:
                df = yf.download(t, period=period, interval=interval, auto_adjust=True, progress=False)
                if not df.empty:
                    df = df.reset_index()
                    df["ticker"] = t
                    results[t] = df
            except Exception as e2:
                log.error(f"yfinance {t} failed: {e2}")
    return results


def fetch_fred_series(series_id: str, start_date: str = "2000-01-01") -> Optional[pd.DataFrame]:
    """Fetch a single FRED series."""
    if not FRED_API_KEY:
        log.warning("FRED_API_KEY not set")
        return None
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start_date,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observations", [])
        if not obs:
            return None
        df = pd.DataFrame(obs)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df["series_id"] = series_id
        return df[["date", "value", "series_id"]]
    except Exception as e:
        log.error(f"FRED {series_id} failed: {e}")
        return None


def fetch_alpha_vantage_vix_term() -> Optional[pd.DataFrame]:
    """Fetch VIX term structure from Alpha Vantage (free tier)."""
    if not ALPHA_VANTAGE_KEY:
        log.warning("ALPHA_VANTAGE_KEY not set")
        return None
    url = "https://www.alphavantage.co/query"
    params = {"function": "VIX", "apikey": ALPHA_VANTAGE_KEY}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Alpha Vantage returns monthly VIX data
        if "Technical Analysis: VIX" in data:
            df = pd.DataFrame(data["Technical Analysis: VIX"]).T
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={"index": "date"})
            for col in df.columns:
                if col != "date":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
    except Exception as e:
        log.error(f"Alpha Vantage VIX failed: {e}")
    return None


def fetch_cboe_put_call_ratio() -> Optional[pd.DataFrame]:
    """Fetch CBOE put/call ratio from public CSV."""
    urls = [
        "https://cdn.cboe.com/api/global/delayed_quotes/put_call_ratio.csv",
        "https://markets.cboe.com/us/options/market_statistics/daily/",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if not df.empty:
                return df
        except Exception:
            continue
    return None


def fetch_cboe_skew() -> Optional[pd.DataFrame]:
    """Fetch CBOE SKEW index."""
    url = "https://cdn.cboe.com/api/global/delayed_quotes/skew.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        log.error(f"CBOE SKEW failed: {e}")
    return None


def fetch_openinsider_cluster_buys(days: int = 7) -> Optional[pd.DataFrame]:
    """Fetch recent insider cluster buys from OpenInsider (free)."""
    url = "http://openinsider.com/screener"
    params = {
        "s": "",
        "o": "",
        "pl": "",
        "ph": "",
        "ll": "",
        "lh": "",
        "fd": days,
        "fdr": "",
        "td": 0,
        "tdr": "",
        "fdlyl": "",
        "fdlyh": "",
        "daysago": "",
        "xp": 1,
        "xs": 1,
        "vl": 100000,
        "vh": "",
        "oc": 1,
        "oe": 1,
        "sicMin": "",
        "sicMax": "",
        "sortcol": 0,
        "cnt": 100,
        "page": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "tinytable"})
        if not table:
            return None
        rows = table.find_all("tr")[1:]  # Skip header
        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 13:
                data.append({
                    "ticker": cols[2].text.strip(),
                    "insider": cols[3].text.strip(),
                    "relation": cols[4].text.strip(),
                    "date": cols[5].text.strip(),
                    "type": cols[6].text.strip(),
                    "shares": cols[7].text.strip(),
                    "price": cols[8].text.strip(),
                    "value": cols[9].text.strip(),
                })
        return pd.DataFrame(data) if data else None
    except Exception as e:
        log.error(f"OpenInsider failed: {e}")
    return None


def fetch_cot_report() -> Optional[pd.DataFrame]:
    """Fetch CFTC Commitment of Traders report."""
    url = "https://www.cftc.gov/dea/newcot/f_disagg.txt"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.split("\n")
        # Parse the fixed-width format (simplified)
        data = []
        current_market = None
        for line in lines:
            if "FUTURES ONLY" in line or "OPTIONS AND FUTURES" in line:
                continue
            if line.strip() and not line.startswith(" ") and "," not in line:
                current_market = line.strip()
            elif "COMMERCIAL" in line and current_market:
                parts = line.split()
                if len(parts) >= 6:
                    data.append({
                        "market": current_market,
                        "commercial_long": parts[1].replace(",", ""),
                        "commercial_short": parts[2].replace(",", ""),
                        "noncomm_long": parts[3].replace(",", ""),
                        "noncomm_short": parts[4].replace(",", ""),
                    })
        return pd.DataFrame(data) if data else None
    except Exception as e:
        log.error(f"COT failed: {e}")
    return None


def fetch_aaii_sentiment() -> Optional[pd.DataFrame]:
    """Fetch AAII sentiment survey (weekly)."""
    url = "https://www.aaii.com/sentimentsurvey"
    try:
        resp = requests.get(url, timeout=30)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Parse the table - simplified
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")
            data = []
            for row in rows[1:]:
                cols = row.find_all("td")
                if len(cols) >= 4:
                    data.append({
                        "date": cols[0].text.strip(),
                        "bullish": cols[1].text.strip(),
                        "neutral": cols[2].text.strip(),
                        "bearish": cols[3].text.strip(),
                    })
            return pd.DataFrame(data) if data else None
    except Exception as e:
        log.error(f"AAII failed: {e}")
    return None


def fetch_fear_greed() -> Optional[pd.DataFrame]:
    """Fetch CNN Fear & Greed Index."""
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Returns current + history
        return pd.DataFrame(data.get("fear_and_greed_historical", {}).get("data", []))
    except Exception as e:
        log.error(f"Fear & Greed failed: {e}")
    return None


def fetch_market_internals() -> Optional[pd.DataFrame]:
    """Fetch NYSE/NASDAQ internals (advance/decline, new highs/lows)."""
    # Using StockCharts free API or yfinance proxies
    internals_tickers = {
        "^NYAD": "NYSE Advance-Decline",
        "^NAAD": "NASDAQ Advance-Decline",
        "^NYHL": "NYSE New Highs-Lows",
        "^NAHL": "NASDAQ New Highs-Lows",
        "^TRIN": "TRIN (Arms Index)",
        "^TICK": "NYSE TICK",
    }
    results = {}
    for ticker, name in internals_tickers.items():
        try:
            df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True, progress=False)
            if not df.empty:
                df = df.reset_index()
                df["ticker"] = ticker
                df["name"] = name
                results[ticker] = df
        except Exception as e:
            log.warning(f"Internals {ticker} failed: {e}")
    if results:
        return pd.concat(results.values(), ignore_index=True)
    return None


# ─── Main Ingestion Functions ────────────────────────────────────────────
def ingest_equities() -> Dict[str, pd.DataFrame]:
    """Fetch all equity/macro tickers from yfinance."""
    log.info("Fetching equity/macro data from yfinance...")
    tickers = list(MACRO_TICKERS.keys())
    # Batch in chunks of 50
    all_results = {}
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i+50]
        results = fetch_yfinance_batch(chunk, period="2y", interval="1d")
        all_results.update(results)
        time.sleep(1)  # Be nice
    return all_results


def ingest_fred() -> Dict[str, pd.DataFrame]:
    """Fetch all FRED series."""
    log.info("Fetching FRED macro series...")
    results = {}
    for series_id, name in FRED_SERIES.items():
        df = fetch_fred_series(series_id)
        if df is not None:
            results[series_id] = df
            log.info(f"  {series_id} ({name}): {len(df)} rows")
        time.sleep(0.1)
    return results


def ingest_options_data() -> Dict[str, pd.DataFrame]:
    """Fetch options-related data."""
    log.info("Fetching options data...")
    results = {}
    pc = fetch_cboe_put_call_ratio()
    if pc is not None:
        results["put_call_ratio"] = pc
    skew = fetch_cboe_skew()
    if skew is not None:
        results["skew"] = skew
    vix_term = fetch_alpha_vantage_vix_term()
    if vix_term is not None:
        results["vix_term"] = vix_term
    return results


def ingest_sentiment() -> Dict[str, pd.DataFrame]:
    """Fetch sentiment data."""
    log.info("Fetching sentiment data...")
    results = {}
    aaii = fetch_aaii_sentiment()
    if aaii is not None:
        results["aaii"] = aaii
    fg = fetch_fear_greed()
    if fg is not None:
        results["fear_greed"] = fg
    return results


def ingest_smart_money() -> Dict[str, pd.DataFrame]:
    """Fetch smart money flows."""
    log.info("Fetching smart money data...")
    results = {}
    insider = fetch_openinsider_cluster_buys(7)
    if insider is not None:
        results["insider_cluster"] = insider
    cot = fetch_cot_report()
    if cot is not None:
        results["cot"] = cot
    return results


def ingest_internals() -> Optional[pd.DataFrame]:
    """Fetch market internals."""
    log.info("Fetching market internals...")
    return fetch_market_internals()


def run_full_ingestion() -> Dict[str, Any]:
    """Run complete ingestion pipeline."""
    log.info("=" * 50)
    log.info("DAILY MARKET BRIEF — FULL INGESTION")
    log.info("=" * 50)

    start_time = time.time()
    all_data = {}

    # 1. Equities (core)
    all_data["equities"] = ingest_equities()

    # 2. FRED Macro
    all_data["fred"] = ingest_fred()

    # 3. Options
    all_data["options"] = ingest_options_data()

    # 4. Sentiment
    all_data["sentiment"] = ingest_sentiment()

    # 5. Smart Money
    all_data["smart_money"] = ingest_smart_money()

    # 6. Internals
    internals = ingest_internals()
    if internals is not None:
        all_data["internals"] = internals

    # Save everything to Parquet
    log.info("Saving to Parquet...")
    for category, datasets in all_data.items():
        if isinstance(datasets, dict) and datasets:
            for name, df in datasets.items():
                if df is not None and len(df) > 0:
                    save_parquet(df, f"{category}/{name}")
        elif datasets is not None and len(datasets) > 0:
            save_parquet(datasets, category)

    # Also save to DuckDB for fast queries
    log.info("Loading to DuckDB...")
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        for category, datasets in all_data.items():
            if isinstance(datasets, dict):
                for name, df in datasets.items():
                    if df is not None and len(df) > 0:
                        table_name = f"{category}_{name}".replace("-", "_").replace("^", "").replace("=", "_")
                        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
            elif datasets is not None and len(datasets) > 0:
                table_name = category.replace("-", "_").replace("^", "").replace("=", "_")
                con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM datasets")
    finally:
        con.close()

    elapsed = time.time() - start_time
    log.info(f"Full ingestion complete in {elapsed:.1f}s")
    return all_data


if __name__ == "__main__":
    run_full_ingestion()