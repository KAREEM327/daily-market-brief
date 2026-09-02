"""
Options Engine — Gamma Exposure (GEX), VRP, Put/Call Skew, Dealer Positioning.
Based on SpotGamma, SqueezeMetrics, and CBOE methodologies.
All from free/public data sources.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import duckdb

# ─── Config ──────────────────────────────────────────────────────────────
DATA_ROOT = Path("/Users/blackstarr/CLAUDE COWORK/daily-market-brief/data")
PARQUET_DIR = DATA_ROOT / "parquet"
DUCKDB_PATH = DATA_ROOT / "market.duckdb"
OUTPUT_DIR = DATA_ROOT / "options"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmb.options")

# Major index/ETF options to track
OPTION_TICKERS = ["SPY", "QQQ", "IWM", "SPX", "NDX", "RUT"]
VIX_TICKERS = ["VIX", "VX3M", "VXMT"]


# ─── Data Fetching ───────────────────────────────────────────────────────
def fetch_cboe_options_data(ticker: str = "SPX") -> Optional[pd.DataFrame]:
    """Fetch options chain data from CBOE (delayed, free)."""
    # CBOE provides delayed options data
    # For SPX, we can get from CBOE's delayed quotes
    urls = {
        "SPX": "https://cdn.cboe.com/api/global/delayed_quotes/options/SPX.json",
        "SPY": "https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json",
        "QQQ": "https://cdn.cboe.com/api/global/delayed_quotes/options/QQQ.json",
        "IWM": "https://cdn.cboe.com/api/global/delayed_quotes/options/IWM.json",
    }
    url = urls.get(ticker.upper())
    if not url:
        return None

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Parse options chain
        options = data.get("data", {}).get("options", [])
        if options:
            df = pd.DataFrame(options)
            return df
    except Exception as e:
        log.warning(f"CBOE options {ticker} failed: {e}")
    return None


def fetch_yfinance_options(ticker: str) -> Dict[str, pd.DataFrame]:
    """Fetch options chain from yfinance (free)."""
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return {}

        chains = {}
        for exp in expirations[:8]:  # Limit to first 8 expirations
            try:
                chain = stock.option_chain(exp)
                calls = chain.calls.copy()
                puts = chain.puts.copy()
                calls["expiration"] = exp
                puts["expiration"] = exp
                calls["type"] = "call"
                puts["type"] = "put"
                df = pd.concat([calls, puts], ignore_index=True)
                df["ticker"] = ticker
                chains[exp] = df
            except Exception as e:
                log.warning(f"yfinance options {ticker} {exp} failed: {e}")
        return chains
    except Exception as e:
        log.error(f"yfinance options {ticker} failed: {e}")
        return {}


def fetch_vix_term_structure() -> Optional[pd.DataFrame]:
    """Fetch VIX term structure from CBOE."""
    url = "https://cdn.cboe.com/api/global/delayed_quotes/vix_term_structure.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return pd.DataFrame(data.get("data", []))
    except Exception as e:
        log.warning(f"VIX term structure failed: {e}")
    return None


def fetch_cboe_skew_index() -> Optional[float]:
    """Fetch current CBOE SKEW index."""
    url = "https://cdn.cboe.com/api/global/delayed_quotes/skew.csv"
    try:
        df = pd.read_csv(url)
        if not df.empty:
            return float(df.iloc[-1, -1])
    except Exception as e:
        log.warning(f"SKEW index failed: {e}")
    return None


def fetch_put_call_ratio() -> Optional[pd.DataFrame]:
    """Fetch CBOE put/call ratio."""
    url = "https://cdn.cboe.com/api/global/delayed_quotes/put_call_ratio.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        log.warning(f"Put/Call ratio failed: {e}")
    return None


def fetch_open_interest_profile(ticker: str = "SPY") -> Optional[pd.DataFrame]:
    """Fetch open interest profile from yfinance or CBOE."""
    chains = fetch_yfinance_options(ticker)
    if not chains:
        return None

    all_oi = []
    for exp, df in chains.items():
        if "openInterest" in df.columns:
            df = df.copy()
            df["expiration"] = exp
            all_oi.append(df)

    if all_oi:
        return pd.concat(all_oi, ignore_index=True)
    return None


# ─── Gamma Exposure (GEX) Calculation ────────────────────────────────────
def calculate_gex(
    options_df: pd.DataFrame,
    spot_price: float,
    gamma_col: str = "gamma",
    oi_col: str = "openInterest",
    strike_col: str = "strike",
    type_col: str = "type",
) -> pd.DataFrame:
    """
    Calculate Gamma Exposure (GEX) by strike.
    GEX = Open Interest × Contract Size (100) × Gamma × Spot Price
    Positive GEX = dealers long gamma (hedge by selling into rallies, buying dips)
    Negative GEX = dealers short gamma (hedge by buying rallies, selling dips)
    """
    if options_df.empty:
        return pd.DataFrame()

    df = options_df.copy()

    # Ensure required columns
    required = [gamma_col, oi_col, strike_col, type_col]
    for col in required:
        if col not in df.columns:
            log.warning(f"Missing column {col} for GEX calculation")
            return pd.DataFrame()

    # Clean data
    df[gamma_col] = pd.to_numeric(df[gamma_col], errors="coerce").fillna(0)
    df[oi_col] = pd.to_numeric(df[oi_col], errors="coerce").fillna(0)
    df[strike_col] = pd.to_numeric(df[strike_col], errors="coerce")

    # GEX calculation
    # Call gamma is positive, put gamma is positive (but dealers are short puts typically)
    # Dealer GEX = OI × 100 × Gamma × Spot × (1 for calls, -1 for puts assuming dealers short)
    df["gex"] = df[oi_col] * 100 * df[gamma_col] * spot_price
    df.loc[df[type_col] == "put", "gex"] *= -1  # Dealers typically short puts

    # Aggregate by strike
    gex_by_strike = df.groupby(strike_col)["gex"].sum().reset_index()
    gex_by_strike.columns = ["strike", "gex"]
    gex_by_strike = gex_by_strike.sort_values("strike")

    # Net GEX
    net_gex = gex_by_strike["gex"].sum()

    # Zero gamma level (where GEX flips)
    gex_by_strike["cum_gex"] = gex_by_strike["gex"].cumsum()
    zero_gamma = None
    for i in range(len(gex_by_strike) - 1):
        if gex_by_strike.iloc[i]["cum_gex"] * gex_by_strike.iloc[i+1]["cum_gex"] <= 0:
            # Linear interpolation
            x1, y1 = gex_by_strike.iloc[i]["strike"], gex_by_strike.iloc[i]["cum_gex"]
            x2, y2 = gex_by_strike.iloc[i+1]["strike"], gex_by_strike.iloc[i+1]["cum_gex"]
            if y2 != y1:
                zero_gamma = x1 - y1 * (x2 - x1) / (y2 - y1)
            break

    return {
        "gex_by_strike": gex_by_strike.to_dict("records"),
        "net_gex": round(float(net_gex), 2),
        "zero_gamma_level": round(float(zero_gamma), 2) if zero_gamma else None,
        "spot_price": spot_price,
        "gex_flip_distance_pct": round((zero_gamma - spot_price) / spot_price * 100, 2) if zero_gamma else None,
    }


def calculate_vrp(spot_vix: float, realized_vol: float) -> Dict[str, float]:
    """Calculate Variance Risk Premium (VRP)."""
    if realized_vol <= 0:
        return {"vrp": None, "vrp_pct": None}
    vrp = spot_vix - realized_vol
    vrp_pct = vrp / realized_vol * 100
    return {"vrp": round(vrp, 2), "vrp_pct": round(vrp_pct, 2)}


def calculate_skew_metrics(options_df: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
    """Calculate put/call skew metrics."""
    if options_df.empty:
        return {}

    df = options_df.copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["impliedVolatility"] = pd.to_numeric(df.get("impliedVolatility", 0), errors="coerce")
    df = df.dropna(subset=["strike", "impliedVolatility"])

    # Separate calls and puts
    calls = df[df["type"] == "call"]
    puts = df[df["type"] == "put"]

    # ATM strikes
    atm_strike = calls.iloc[(calls["strike"] - spot_price).abs().argsort()[:1]]["strike"].values[0] if len(calls) > 0 else spot_price

    # 25-delta skew (approximate)
    # Find strikes closest to 25 delta (roughly ATM ± 0.5% for SPX)
    otm_puts = puts[puts["strike"] < spot_price]
    otm_calls = calls[calls["strike"] > spot_price]

    skew_25d = None
    if len(otm_puts) > 0 and len(otm_calls) > 0:
        put_iv = otm_puts.iloc[(otm_puts["strike"] - spot_price * 0.995).abs().argsort()[:1]]["impliedVolatility"].values[0]
        call_iv = otm_calls.iloc[(otm_calls["strike"] - spot_price * 1.005).abs().argsort()[:1]]["impliedVolatility"].values[0]
        skew_25d = put_iv - call_iv

    # ATM IV
    atm_iv = None
    atm_options = df[(df["strike"] - atm_strike).abs() < spot_price * 0.01]
    if len(atm_options) > 0:
        atm_iv = atm_options["impliedVolatility"].mean()

    # Term structure slope (front vs back month IV)
    expirations = df["expiration"].unique()
    term_slope = None
    if len(expirations) >= 2:
        front_exp = sorted(expirations)[0]
        back_exp = sorted(expirations)[1]
        front_atm = df[(df["expiration"] == front_exp) & (df["strike"] - atm_strike).abs() < spot_price * 0.02]
        back_atm = df[(df["expiration"] == back_exp) & (df["strike"] - atm_strike).abs() < spot_price * 0.02]
        if len(front_atm) > 0 and len(back_atm) > 0:
            term_slope = back_atm["impliedVolatility"].mean() - front_atm["impliedVolatility"].mean()

    return {
        "atm_strike": round(float(atm_strike), 2) if atm_strike else None,
        "atm_iv": round(float(atm_iv), 4) if atm_iv else None,
        "skew_25d": round(float(skew_25d), 4) if skew_25d else None,
        "term_slope": round(float(term_slope), 4) if term_slope else None,
        "put_call_iv_spread": round(float(puts["impliedVolatility"].mean() - calls["impliedVolatility"].mean()), 4) if len(puts) > 0 and len(calls) > 0 else None,
    }


def calculate_dealer_positioning(gex_result: Dict, spot_price: float) -> Dict[str, Any]:
    """Infer dealer positioning from GEX."""
    if not gex_result or "gex_by_strike" == 0:
        return {}

    gex_df = pd.DataFrame(gex_result["gex_by_strike"])
    net_gex = gex_result.get("net_gex", 0)
    zero_gamma = gex_result.get("zero_gamma_level")

    # Dealer gamma exposure
    if net_gex > 0:
        dealer_gamma = "LONG"  # Dealers long gamma = stabilizing
        hedging = "Sell rallies, buy dips (mean reversion)"
    else:
        dealer_gamma = "SHORT"  # Dealers short gamma = destabilizing
        hedging = "Buy rallies, sell dips (trend following)"

    # Key gamma walls (large GEX concentrations)
    gex_df["abs_gex"] = gex_df["gex"].abs()
    walls = gex_df.nlargest(5, "abs_gex")[["strike", "gex"]].to_dict("records")

    # Distance to zero gamma
    flip_dist = gex_result.get("gex_flip_distance_pct")

    return {
        "dealer_gamma": dealer_gamma,
        "hedging_behavior": hedging,
        "net_gex_billions": round(net_gex / 1e9, 2),
        "zero_gamma_level": zero_gamma,
        "flip_distance_pct": flip_dist,
        "gamma_walls": walls,
        "interpretation": (
            f"Dealers {dealer_gamma.lower()} gamma. "
            f"Zero gamma at {zero_gamma:.0f} ({flip_dist:+.2f}% from spot). "
            f"{'Stabilizing - expect mean reversion.' if dealer_gamma == 'LONG' else 'Destabilizing - expect trend continuation.'}"
        ) if zero_gamma else "Zero gamma level not found",
    }


def run_options_analysis() -> Dict[str, Any]:
    """Run full options analysis."""
    log.info("=" * 50)
    log.info("OPTIONS ENGINE — GEX, VRP, SKEW, DEALER POSITIONING")
    log.info("=" * 50)

    results = {}

    # Get spot prices
    spot_prices = {}
    for ticker in OPTION_TICKERS:
        try:
            data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
            if not data.empty:
                spot_prices[ticker] = float(data["Close"].iloc[-1])
        except Exception:
            pass

    # VIX
    vix_spot = spot_prices.get("VIX", 0)
    realized_vol = 0
    try:
        spy_data = yf.download("SPY", period="1mo", interval="1d", progress=False, auto_adjust=True)
        if not spy_data.empty:
            realized_vol = float(spy_data["Close"].pct_change().std() * np.sqrt(252) * 100)
    except Exception:
        pass

    # VRP
    vrp = calculate_vrp(vix_spot, realized_vol)
    results["vrp"] = vrp
    results["vix_spot"] = round(vix_spot, 2)
    results["realized_vol_21d"] = round(realized_vol, 2)

    # VIX term structure
    vix_term = fetch_vix_term_structure()
    if vix_term is not None:
        results["vix_term_structure"] = vix_term.to_dict("records")

    # SKEW index
    skew = fetch_cboe_skew_index()
    if skew:
        results["cboe_skew"] = round(skew, 2)

    # Put/Call ratio
    pc_ratio = fetch_put_call_ratio()
    if pc_ratio is not None:
        results["put_call_ratio"] = pc_ratio.to_dict("records")

    # Per-ticker GEX and skew
    for ticker in ["SPY", "QQQ", "IWM"]:
        if ticker not in spot_prices:
            continue

        spot = spot_prices[ticker]
        log.info(f"Analyzing options for {ticker} @ ${spot:.2f}")

        # Fetch options
        chains = fetch_yfinance_options(ticker)
        if not chains:
            continue

        # Combine all expirations
        all_options = pd.concat(chains.values(), ignore_index=True)

        # GEX
        gex_result = calculate_gex(all_options, spot)
        if gex_result:
            results[f"{ticker}_gex"] = gex_result

            # Dealer positioning
            dealer = calculate_dealer_positioning(gex_result, spot)
            results[f"{ticker}_dealer"] = dealer

        # Skew metrics
        skew_metrics = calculate_skew_metrics(all_options, spot)
        if skew_metrics:
            results[f"{ticker}_skew"] = skew_metrics

    # Save
    output_file = OUTPUT_DIR / f"options_{date.today().isoformat()}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved options analysis to {output_file}")

    # Summary
    log.info("\nOPTIONS SUMMARY:")
    log.info(f"  VIX: {vix_spot:.1f} | Realized Vol: {realized_vol:.1f}% | VRP: {vrp.get('vrp', 'N/A')}")
    log.info(f"  CBOE SKEW: {skew if skew else 'N/A'}")
    for ticker in ["SPY", "QQQ", "IWM"]:
        dealer = results.get(f"{ticker}_dealer", {})
        if dealer:
            log.info(f"  {ticker}: Dealers {dealer.get('dealer_gamma')} gamma | "
                     f"Zero γ: {dealer.get('zero_gamma_level', 'N/A')} | "
                     f"Net GEX: ${dealer.get('net_gex_billions', 0):.1f}B")

    return results


if __name__ == "__main__":
    run_options_analysis()