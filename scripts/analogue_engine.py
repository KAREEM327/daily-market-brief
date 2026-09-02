"""
Analogue Engine — Historical pattern matching via k-NN on market state vectors.
Based on Leinweber (2003), Kritzman et al. (2010) — "Regime Shifts and Asset Allocation".
Finds similar historical setups and their forward returns.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import duckdb
from scipy.spatial.distance import cdist
from scipy.stats import percentileofscore

# ─── Config ──────────────────────────────────────────────────────────────
DATA_ROOT = Path("/Users/blackstarr/CLAUDE COWORK/daily-market-brief/data")
PARQUET_DIR = DATA_ROOT / "parquet"
DUCKDB_PATH = DATA_ROOT / "market.duckdb"
OUTPUT_DIR = DATA_ROOT / "analogue"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = DATA_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmb.analogue")

# State vector features (20 factors)
FEATURE_CONFIG = {
    # Breadth
    "advance_decline_ratio": {"weight": 1.5, "window": 21},
    "new_highs_lows_ratio": {"weight": 1.2, "window": 21},
    "mccllellan_oscillator": {"weight": 1.0, "window": 21},
    "mccllellan_summation": {"weight": 1.0, "window": 63},

    # Momentum
    "spy_momentum_21": {"weight": 1.5, "window": 21},
    "spy_momentum_63": {"weight": 1.5, "window": 63},
    "spy_momentum_252": {"weight": 1.0, "window": 252},
    "qqq_spy_momentum_spread": {"weight": 1.2, "window": 21},
    "iwm_spy_momentum_spread": {"weight": 1.2, "window": 21},

    # Volatility
    "vix_level": {"weight": 2.0, "window": 1},
    "vix_term_structure": {"weight": 1.5, "window": 1},  # VIX/VX3M
    "vix_spot_vs_ma50": {"weight": 1.0, "window": 50},
    "realized_vol_21": {"weight": 1.0, "window": 21},

    # Term Structure / Macro
    "yield_curve_10y2y": {"weight": 1.5, "window": 1},
    "yield_curve_10y3m": {"weight": 1.5, "window": 1},
    "dxy_momentum_21": {"weight": 1.0, "window": 21},
    "oil_momentum_63": {"weight": 1.0, "window": 63},

    # Sentiment / Positioning
    "put_call_ratio": {"weight": 1.2, "window": 21},
    "aaii_bull_bear_spread": {"weight": 1.0, "window": 1},
    "insider_buy_sell_ratio": {"weight": 1.0, "window": 63},

    # Correlation Regime
    "avg_correlation_63": {"weight": 1.5, "window": 63},
    "correlation_trend": {"weight": 1.0, "window": 63},
}


# ─── Data Loading ────────────────────────────────────────────────────────
def load_market_data(years: int = 20) -> Dict[str, pd.Series]:
    """Load all required price series for feature construction."""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    tickers = [
        "SPY", "QQQ", "IWM", "DIA", "VIX", "VX3M",
        "^TNX", "^TYX", "^FVX", "^IRX",
        "DXY", "EURUSD=X", "USDJPY=X",
        "GC=F", "CL=F",
        "TLT", "IEF", "SHY", "LQD", "HYG",
        "EEM", "FXI", "EWG", "EWJ",
        "XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLU", "XLB", "XLC", "XLY",
        "^NYAD", "^NAAD", "^NYHL", "^NAHL", "^TRIN", "^TICK",
    ]

    results = {}
    for ticker in tickers:
        table_patterns = [
            f"equities_{ticker}",
            f"equities_{ticker.lower()}",
            f"equities_{ticker.replace('=', '_').replace('^', '')}",
        ]
        for table in table_patterns:
            try:
                df = con.execute(f"SELECT * FROM {table} ORDER BY Date").df()
                break
            except Exception:
                continue

        if df is not None and not df.empty:
            date_col = "Date" if "Date" in df.columns else "date"
            close_col = "Close" if "Close" in df.columns else "close"
            if date_col in df.columns and close_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col).sort_index()
                series = df[close_col].dropna()
                series.name = ticker
                cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
                series = series[series.index >= cutoff]
                results[ticker] = series
    con.close()
    return results


def load_fred_data() -> Dict[str, pd.Series]:
    """Load FRED macro series."""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    series_ids = ["DGS10", "DGS2", "T10Y2Y", "T10Y3M", "DCOILWTICO", "DTWEXBGS", "VIXCLS"]
    results = {}
    for sid in series_ids:
        try:
            df = con.execute(f"SELECT * FROM fred_{sid} ORDER BY date").df()
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                results[sid] = df["value"].dropna()
        except Exception:
            pass
    con.close()
    return results


# ─── Feature Construction ────────────────────────────────────────────────
def compute_breadth_features(prices: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compute market breadth features."""
    features = pd.DataFrame(index=prices["SPY"].index)

    # Advance-Decline Ratio (using NYSE data)
    if "^NYAD" in prices:
        adv_dec = prices["^NYAD"]
        features["advance_decline_ratio"] = adv_dec / adv_dec.rolling(21).mean()

    # New Highs/Lows Ratio
    if "^NYHL" in prices:
        nh_nl = prices["^NYHL"]
        features["new_highs_lows_ratio"] = nh_nl / nh_nl.rolling(21).mean()

    # McClellan Oscillator (19-day EMA of A/D - 39-day EMA of A/D)
    if "^NYAD" in prices:
        ad = prices["^NYAD"]
        ema19 = ad.ewm(span=19).mean()
        ema39 = ad.ewm(span=39).mean()
        features["mccllellan_oscillator"] = ema19 - ema39
        # Summation Index
        features["mccllellan_summation"] = (ema19 - ema39).cumsum()

    return features


def compute_momentum_features(prices: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compute momentum features."""
    features = pd.DataFrame(index=prices["SPY"].index)

    for ticker in ["SPY", "QQQ", "IWM"]:
        if ticker in prices:
            close = prices[ticker]
            for window in [21, 63, 252]:
                mom = close.pct_change(window)
                features[f"{ticker.lower()}_momentum_{window}"] = mom

    # Spreads
    if "QQQ" in prices and "SPY" in prices:
        features["qqq_spy_momentum_spread"] = (
            prices["QQQ"].pct_change(21) - prices["SPY"].pct_change(21)
        )
    if "IWM" in prices and "SPY" in prices:
        features["iwm_spy_momentum_spread"] = (
            prices["IWM"].pct_change(21) - prices["SPY"].pct_change(21)
        )

    return features


def compute_volatility_features(prices: Dict[str, pd.Series], fred: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compute volatility features."""
    features = pd.DataFrame(index=prices["SPY"].index)

    if "VIX" in prices:
        vix = prices["VIX"]
        features["vix_level"] = vix
        features["vix_spot_vs_ma50"] = vix / vix.rolling(50).mean() - 1

        # VIX term structure
        if "VX3M" in prices:
            features["vix_term_structure"] = vix / prices["VX3M"]

    # Realized vol
    spy_rets = prices["SPY"].pct_change().dropna()
    features["realized_vol_21"] = spy_rets.rolling(21).std() * np.sqrt(252)

    return features


def compute_macro_features(prices: Dict[str, pd.Series], fred: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compute macro/term structure features."""
    features = pd.DataFrame(index=prices["SPY"].index)

    # Yield curves from FRED
    if "T10Y2Y" in fred:
        features["yield_curve_10y2y"] = fred["T10Y2Y"].reindex(features.index).ffill()
    if "T10Y3M" in fred:
        features["yield_curve_10y3m"] = fred["T10Y3M"].reindex(features.index).ffill()

    # DXY momentum
    if "DXY" in prices:
        features["dxy_momentum_21"] = prices["DXY"].pct_change(21)

    # Oil momentum
    if "CL=F" in prices:
        features["oil_momentum_63"] = prices["CL=F"].pct_change(63)
    elif "USO" in prices:
        features["oil_momentum_63"] = prices["USO"].pct_change(63)

    return features


def compute_sentiment_features(prices: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compute sentiment/positioning features."""
    features = pd.DataFrame(index=prices["SPY"].index)

    # Put/Call ratio from CBOE (if available in parquet)
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        pc_df = con.execute("SELECT * FROM options_put_call_ratio ORDER BY date").df()
        if not pc_df.empty:
            pc_df["date"] = pd.to_datetime(pc_df["date"])
            pc_df = pc_df.set_index("date").sort_index()
            features["put_call_ratio"] = pc_df["ratio"].reindex(features.index).ffill().rolling(21).mean()
    except Exception:
        pass

    # AAII sentiment (if available)
    try:
        aaii_df = con.execute("SELECT * FROM sentiment_aaii ORDER BY date").df()
        if not aaii_df.empty:
            aaii_df["date"] = pd.to_datetime(aaii_df["date"])
            aaii_df = aaii_df.set_index("date").sort_index()
            features["aaii_bull_bear_spread"] = (
                pd.to_numeric(aaii_df["bullish"], errors="coerce") -
                pd.to_numeric(aaii_df["bearish"], errors="coerce")
            ).reindex(features.index).ffill()
    except Exception:
        pass

    # Insider buy/sell ratio (if available)
    try:
        insider_df = con.execute("SELECT * FROM smart_money_insider_cluster ORDER BY date").df()
        if not insider_df.empty:
            insider_df["date"] = pd.to_datetime(insider_df["date"])
            insider_df = insider_df.set_index("date").sort_index()
            # This would need parsing - placeholder
            pass
    except Exception:
        pass

    con.close()
    return features


def compute_correlation_features(prices: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compute correlation regime features."""
    features = pd.DataFrame(index=prices["SPY"].index)

    # Core assets for correlation
    core = ["SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "UUP", "EEM"]
    available = [a for a in core if a in prices]
    if len(available) >= 4:
        df = pd.DataFrame({a: prices[a] for a in available})
        returns = df.pct_change().dropna()

        # Rolling average correlation
        def avg_corr(window):
            corrs = []
            for i in range(window, len(returns) + 1):
                sub = returns.iloc[i-window:i]
                corr = sub.corr()
                mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
                corrs.append(corr.values[mask].mean())
            return pd.Series(corrs, index=returns.index[window-1:])

        features["avg_correlation_63"] = avg_corr(63).reindex(features.index).ffill()
        features["correlation_trend"] = features["avg_correlation_63"].pct_change(21)

    return features


def build_state_vector(prices: Dict[str, pd.Series], fred: Dict[str, pd.Series]) -> pd.DataFrame:
    """Build the complete state vector DataFrame."""
    log.info("Building state vector...")

    # Compute all feature groups
    breadth = compute_breadth_features(prices)
    momentum = compute_momentum_features(prices)
    volatility = compute_volatility_features(prices, fred)
    macro = compute_macro_features(prices, fred)
    sentiment = compute_sentiment_features(prices)
    correlation = compute_correlation_features(prices)

    # Combine
    all_features = pd.concat([breadth, momentum, volatility, macro, sentiment, correlation], axis=1)

    # Forward fill then drop rows with too many NaNs
    all_features = all_features.ffill()
    # Keep only rows with at least 80% of features
    min_features = int(len(all_features.columns) * 0.8)
    all_features = all_features.dropna(thresh=min_features)

    # Standardize each feature (rolling z-score for regime detection)
    standardized = pd.DataFrame(index=all_features.index)
    for col in all_features.columns:
        roll_mean = all_features[col].rolling(252, min_periods=63).mean()
        roll_std = all_features[col].rolling(252, min_periods=63).std()
        standardized[col] = (all_features[col] - roll_mean) / (roll_std + 1e-8)

    # Apply feature weights
    for feat, config in FEATURE_CONFIG.items():
        if feat in standardized.columns:
            standardized[feat] *= config["weight"]

    log.info(f"State vector: {standardized.shape[0]} observations, {standardized.shape[1]} features")
    return standardized.dropna()


# ─── k-NN Analogue Search ────────────────────────────────────────────────
def find_analogues(
    current_state: np.ndarray,
    historical_states: np.ndarray,
    historical_dates: pd.DatetimeIndex,
    k: int = 10,
    min_days_apart: int = 21,
) -> List[Dict[str, Any]]:
    """Find k nearest neighbors in state space."""
    # Compute distances
    distances = cdist([current_state], historical_states, metric="euclidean")[0]

    # Get sorted indices
    sorted_idx = np.argsort(distances)

    analogues = []
    for idx in sorted_idx:
        if len(analogues) >= k:
            break

        hist_date = historical_dates[idx]
        # Ensure minimum time separation
        if analogues:
            last_date = analogues[-1]["date"]
            if (hist_date - last_date).days < min_days_apart:
                continue

        analogues.append({
            "date": hist_date.strftime("%Y-%m-%d"),
            "distance": round(float(distances[idx]), 4),
            "similarity_score": round(1.0 / (1.0 + distances[idx]), 4),
        })

    return analogues


def compute_forward_returns(
    prices: Dict[str, pd.Series],
    analogue_dates: List[str],
    horizons: List[int] = [5, 21, 63],
) -> Dict[str, Dict[str, float]]:
    """Compute forward returns for analogue dates."""
    spy = prices["SPY"]
    results = {h: [] for h in horizons}

    for date_str in analogue_dates:
        dt = pd.Timestamp(date_str)
        # Find closest trading day
        idx = spy.index.get_indexer([dt], method="nearest")[0]
        if idx < 0 or idx >= len(spy) - max(horizons):
            continue

        entry_price = spy.iloc[idx]
        for h in horizons:
            if idx + h < len(spy):
                exit_price = spy.iloc[idx + h]
                ret = (exit_price - entry_price) / entry_price
                results[h].append(ret)

    # Aggregate statistics
    summary = {}
    for h in horizons:
        rets = np.array(results[h])
        if len(rets) > 0:
            summary[f"{h}d"] = {
                "mean": round(float(np.mean(rets)) * 100, 2),
                "median": round(float(np.median(rets)) * 100, 2),
                "std": round(float(np.std(rets)) * 100, 2),
                "win_rate": round(float((rets > 0).mean()) * 100, 1),
                "max": round(float(np.max(rets)) * 100, 2),
                "min": round(float(np.min(rets)) * 100, 2),
                "n": len(rets),
            }

    return summary


def run_analogue_analysis() -> Dict[str, Any]:
    """Run full analogue analysis."""
    log.info("=" * 50)
    log.info("ANALOGUE ENGINE — HISTORICAL PATTERN MATCHING")
    log.info("=" * 50)

    # Load data
    prices = load_market_data(years=20)
    fred = load_fred_data()

    if "SPY" not in prices:
        log.error("SPY not found")
        return {"error": "SPY not found"}

    # Build state vector
    state_vector = build_state_vector(prices, fred)

    if len(state_vector) < 500:
        log.error("Insufficient state vector history")
        return {"error": "Insufficient history"}

    # Current state (latest)
    current_state = state_vector.iloc[-1].values
    current_date = state_vector.index[-1]

    # Historical states (exclude last 63 days to avoid look-ahead)
    historical_states = state_vector.iloc[:-63].values
    historical_dates = state_vector.index[:-63]

    log.info(f"Current state: {current_date.date()}")
    log.info(f"Historical pool: {len(historical_states)} observations")

    # Find analogues
    analogues = find_analogues(current_state, historical_states, historical_dates, k=15)

    # Compute forward returns
    analogue_dates = [a["date"] for a in analogues]
    forward_returns = compute_forward_returns(prices, analogue_dates)

    # Regime-specific analogues
    try:
        from scripts.regime_engine import label_regimes, get_close_series
        spy_labels = label_regimes(get_close_series("SPY"))
        current_regime = STATES[int(spy_labels.iloc[-1])]

        # Filter historical states by same regime
        regime_mask = spy_labels.reindex(historical_dates).ffill() == list(STATES).index(current_regime)
        regime_states = historical_states[regime_mask.values]
        regime_dates = historical_dates[regime_mask.values]

        if len(regime_states) >= 20:
            regime_analogues = find_analogues(
                current_state, regime_states, regime_dates, k=10
            )
            regime_dates_list = [a["date"] for a in regime_analogues]
            regime_forward = compute_forward_returns(prices, regime_dates_list)
        else:
            regime_analogues = []
            regime_forward = {}
    except Exception as e:
        log.warning(f"Regime-specific analogues failed: {e}")
        regime_analogues = []
        regime_forward = {}

    # Compile results
    results = {
        "as_of": current_date.strftime("%Y-%m-%d"),
        "current_regime": current_regime if 'current_regime' in locals() else "Unknown",
        "top_analogues": analogues[:10],
        "forward_returns": forward_returns,
        "regime_specific_analogues": regime_analogues[:10],
        "regime_forward_returns": regime_forward,
        "state_vector_features": list(state_vector.columns),
        "historical_pool_size": len(historical_states),
    }

    # Save
    output_file = OUTPUT_DIR / f"analogue_{date.today().isoformat()}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved analogue analysis to {output_file}")

    # Summary
    log.info("\nANALOGUE SUMMARY:")
    log.info(f"  Current Regime: {results['current_regime']}")
    log.info(f"  Top 5 Analogues:")
    for a in analogues[:5]:
        log.info(f"    {a['date']} (sim: {a['similarity_score']:.2f})")
    if forward_returns:
        for h, stats in forward_returns.items():
            log.info(f"  {h} Forward: mean={stats['mean']:.2f}%, median={stats['median']:.2f}%, win={stats['win_rate']:.1f}%")

    return results


if __name__ == "__main__":
    run_analogue_analysis()