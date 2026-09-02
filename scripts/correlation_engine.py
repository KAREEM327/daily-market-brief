"""
Correlation Engine — Rolling correlations, regime-conditionals, divergence detection.
Based on Longin & Solnik (2001), Forbes & Rigobon (2002) — correlations converge to 1 in crises.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import duckdb

# ─── Config ──────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data"
PARQUET_DIR = DATA_ROOT / "parquet"
DUCKDB_PATH = DATA_ROOT / "market.duckdb"
OUTPUT_DIR = DATA_ROOT / "correlation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmb.correlation")

# Key asset pairs to monitor
CORE_ASSETS = [
    "SPY", "QQQ", "IWM", "DIA",       # US Equities
    "TLT", "IEF", "SHY", "LQD", "HYG", # Fixed Income
    "GLD", "USO", "DBC",               # Commodities
    "UUP", "FXE", "FXY",               # Currencies
    "EEM", "FXI", "EWG", "EWJ",        # International
    "XLF", "XLK", "XLE", "XLV", "XLY", # Sectors
    "^VIX",                             # Volatility
]

CORRELATION_WINDOWS = [21, 63, 126, 252]  # 1m, 3m, 6m, 12m

# Crisis periods for historical reference
CRISIS_PERIODS = [
    ("2008-09-15", "2009-03-09", "GFC"),
    ("2011-08-01", "2011-10-03", "Euro Debt Crisis"),
    ("2015-08-18", "2016-02-11", "China Devaluation"),
    ("2018-10-01", "2018-12-24", "Q4 2018 Selloff"),
    ("2020-02-19", "2020-03-23", "COVID Crash"),
    ("2022-01-03", "2022-10-12", "2022 Bear Market"),
    ("2023-03-08", "2023-05-04", "Regional Bank Crisis"),
]


# ─── Data Loading ────────────────────────────────────────────────────────
def load_price_data(tickers: List[str], years: int = 3) -> Dict[str, pd.Series]:
    """Load close prices from parquet or DuckDB."""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    results = {}
    try:
        for ticker in tickers:
            # Try multiple table name patterns
            table_patterns = [
                f"equities_{ticker}",
                f"equities_{ticker.lower()}",
                f"equities_{ticker.replace('=', '_')}",
            ]
            df = None
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
                    # Trim to requested years
                    cutoff = pd.Timestamp.now() - pd.DateOffset(years=years)
                    series = series[series.index >= cutoff]
                    results[ticker] = series
    finally:
        con.close()
    return results


def compute_returns(prices: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compute daily returns aligned across all assets."""
    df = pd.DataFrame(prices)
    returns = df.pct_change().dropna(how="all")
    # Forward fill small gaps (weekends/holidays)
    returns = returns.ffill().dropna()
    return returns


# ─── Core Correlation Functions ──────────────────────────────────────────
def rolling_correlations(
    returns: pd.DataFrame,
    windows: List[int] = CORRELATION_WINDOWS,
) -> Dict[int, pd.DataFrame]:
    """Compute rolling correlation matrices for multiple windows."""
    results = {}
    for window in windows:
        corr = returns.rolling(window).corr()
        # Get the latest matrix for each window
        if len(corr) >= window:
            # Last window-sized block
            latest = corr.iloc[-len(returns.columns):].copy()
            latest.index = returns.columns
            latest.columns = returns.columns
            results[window] = latest
    return results


def regime_conditional_correlations(
    returns: pd.DataFrame,
    spy_labels: pd.Series,
) -> Dict[str, pd.DataFrame]:
    """Compute correlations conditioned on SPY regime (Bear/Sideways/Bull)."""
    # Align
    common_idx = returns.index.intersection(spy_labels.index)
    returns_aligned = returns.loc[common_idx]
    labels_aligned = spy_labels.loc[common_idx]

    STATES = ["Bear", "Sideways", "Bull"]
    results = {}

    for regime_id, regime_name in enumerate(STATES):
        mask = labels_aligned == regime_id
        if mask.sum() >= 30:  # Minimum observations
            regime_returns = returns_aligned[mask]
            corr = regime_returns.corr()
            results[regime_name] = corr

    return results


def correlation_divergence_alerts(
    current_corr: pd.DataFrame,
    historical_corrs: Dict[int, pd.DataFrame],
    threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """Detect significant correlation divergences from historical norms."""
    alerts = []

    # Compare current (63d) vs 252d (1-year) average
    if 63 in historical_corrs and 252 in historical_corrs:
        corr_63 = historical_corrs[63]
        corr_252 = historical_corrs[252]

        for i, asset1 in enumerate(current_corr.index):
            for j, asset2 in enumerate(current_corr.columns):
                if i >= j:
                    continue  # Upper triangle only

                current = current_corr.iloc[i, j]
                hist_63 = corr_63.iloc[i, j] if i < len(corr_63) and j < len(corr_63.columns) else None
                hist_252 = corr_252.iloc[i, j] if i < len(corr_252) and j < len(corr_252.columns) else None

                if hist_252 is not None and not np.isnan(hist_252):
                    div_252 = abs(current - hist_252)
                    if div_252 > threshold:
                        alerts.append({
                            "pair": f"{asset1}/{asset2}",
                            "current": round(current, 3),
                            "historical_252d": round(hist_252, 3),
                            "divergence": round(div_252, 3),
                            "direction": "decoupling" if current < hist_252 else "converging",
                            "severity": "high" if div_252 > 0.5 else "medium",
                        })

    return alerts


def cross_asset_correlation_heatmap(
    returns: pd.DataFrame,
    window: int = 63,
) -> Dict[str, Any]:
    """Generate correlation heatmap data for key asset groups."""
    corr = returns.rolling(window).corr().iloc[-len(returns.columns):]
    corr.index = returns.columns
    corr.columns = returns.columns

    groups = {
        "US_Equities": ["SPY", "QQQ", "IWM", "DIA"],
        "Fixed_Income": ["TLT", "IEF", "SHY", "LQD", "HYG"],
        "Commodities": ["GLD", "USO", "DBC"],
        "Currencies": ["UUP", "FXE", "FXY"],
        "International": ["EEM", "FXI", "EWG", "EWJ"],
        "Sectors": ["XLF", "XLK", "XLE", "XLV", "XLY"],
    }

    result = {}
    for group_name, tickers in groups.items():
        available = [t for t in tickers if t in corr.index]
        if len(available) >= 2:
            sub = corr.loc[available, available]
            result[group_name] = sub.round(3).to_dict()

    # Cross-group correlations (first asset of each group)
    cross_assets = [tickers[0] for tickers in groups.values() if tickers[0] in corr.index]
    if len(cross_assets) >= 2:
        cross_corr = corr.loc[cross_assets, cross_assets]
        result["Cross_Group"] = cross_corr.round(3).to_dict()

    return result


def spy_correlation_beta(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Compute rolling beta of each asset to SPY."""
    if "SPY" not in returns.columns:
        return pd.DataFrame()

    spy_rets = returns["SPY"]
    betas = {}
    for col in returns.columns:
        if col == "SPY":
            continue
        # Rolling covariance / rolling variance
        cov = returns[col].rolling(window).cov(spy_rets)
        var = spy_rets.rolling(window).var()
        beta = cov / var
        betas[col] = beta

    beta_df = pd.DataFrame(betas)
    return beta_df.tail(1).T.round(3)  # Latest betas


def correlation_stability_metrics(
    returns: pd.DataFrame,
    window: int = 63,
) -> Dict[str, float]:
    """Metrics for correlation regime stability."""
    # Rolling average correlation (ex-diagonal)
    rolling_corrs = []
    for i in range(window, len(returns) + 1):
        sub = returns.iloc[i-window:i]
        corr = sub.corr()
        # Average of upper triangle
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        avg_corr = corr.values[mask].mean()
        rolling_corrs.append(avg_corr)

    rolling_corrs = np.array(rolling_corrs)
    rolling_corrs = rolling_corrs[~np.isnan(rolling_corrs)]

    return {
        "current_avg_correlation": round(float(rolling_corrs[-1]) if len(rolling_corrs) else 0, 3),
        "avg_correlation_1y": round(float(rolling_corrs.mean()) if len(rolling_corrs) else 0, 3),
        "correlation_volatility": round(float(rolling_corrs.std()) if len(rolling_corrs) > 1 else 0, 3),
        "max_correlation_1y": round(float(rolling_corrs.max()) if len(rolling_corrs) else 0, 3),
        "min_correlation_1y": round(float(rolling_corrs.min()) if len(rolling_corrs) else 0, 3),
        "pct_above_07": round(float((rolling_corrs > 0.7).mean()) if len(rolling_corrs) else 0, 3),
    }


def crisis_correlation_analysis(returns: pd.DataFrame) -> Dict[str, Any]:
    """Analyze correlation behavior during historical crisis periods."""
    results = {}

    for start_str, end_str, name in CRISIS_PERIODS:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)

        crisis_returns = returns[(returns.index >= start) & (returns.index <= end)]
        if len(crisis_returns) < 10:
            continue

        corr = crisis_returns.corr()
        # Average correlation (ex-diagonal)
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        avg_corr = corr.values[mask].mean()

        # SPY correlations
        spy_corrs = {}
        if "SPY" in corr.columns:
            for col in corr.columns:
                if col != "SPY":
                    spy_corrs[col] = round(corr.loc["SPY", col], 3)

        results[name] = {
            "period": f"{start_str} to {end_str}",
            "days": len(crisis_returns),
            "avg_correlation": round(float(avg_corr), 3),
            "spy_correlations": spy_corrs,
        }

    return results


def run_correlation_analysis() -> Dict[str, Any]:
    """Run full correlation analysis."""
    log.info("=" * 50)
    log.info("CORRELATION ENGINE — FULL ANALYSIS")
    log.info("=" * 50)

    # Load data
    prices = load_price_data(CORE_ASSETS, years=3)
    log.info(f"Loaded {len(prices)} price series")

    if len(prices) < 5:
        log.error("Insufficient price data")
        return {"error": "Insufficient data"}

    returns = compute_returns(prices)
    log.info(f"Returns matrix: {returns.shape}")

    # Load SPY labels for regime conditioning
    try:
        from scripts.regime_engine import label_regimes, get_close_series
        spy_close = get_close_series("SPY")
        spy_labels = label_regimes(spy_close)
    except Exception as e:
        log.warning(f"Could not load SPY labels: {e}")
        spy_labels = pd.Series()

    # 1. Rolling correlations (multiple windows)
    rolling_corrs = rolling_correlations(returns)
    current_corr = rolling_corrs.get(63, pd.DataFrame())

    # 2. Regime-conditional correlations
    cond_corrs = {}
    if not spy_labels.empty:
        cond_corrs = regime_conditional_correlations(returns, spy_labels)

    # 3. Divergence alerts
    alerts = correlation_divergence_alerts(current_corr, rolling_corrs)

    # 4. Heatmap by group
    heatmap = cross_asset_correlation_heatmap(returns)

    # 5. Rolling betas to SPY
    betas = spy_correlation_beta(returns)

    # 6. Stability metrics
    stability = correlation_stability_metrics(returns)

    # 7. Crisis analysis
    crisis = crisis_correlation_analysis(returns)

    def make_serializable(obj):
        """Recursively convert non-JSON-serializable objects."""
        import pandas as pd
        import numpy as np
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if not np.isnan(obj) else None
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return make_serializable(obj.round(3).to_dict())
        elif isinstance(obj, pd.Series):
            return make_serializable(obj.round(3).to_dict())
        elif obj is None:
            return None
        elif hasattr(obj, '__float__'):
            try:
                val = float(obj)
                return val if not np.isnan(val) else None
            except (TypeError, ValueError):
                return str(obj)
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    # Compile results
    results = {
        "as_of": date.today().isoformat(),
        "current_correlation_matrix": make_serializable(current_corr.round(3).to_dict()) if not current_corr.empty else {},
        "rolling_correlations": {str(k): make_serializable(v.round(3).to_dict()) for k, v in rolling_corrs.items()},
        "regime_conditional_correlations": make_serializable(cond_corrs),
        "divergence_alerts": make_serializable(alerts),
        "group_heatmaps": make_serializable(heatmap),
        "spy_betas": make_serializable(betas.to_dict()) if not betas.empty else {},
        "stability_metrics": make_serializable(stability),
        "crisis_analysis": make_serializable(crisis),
    }

    # Save
    output_file = OUTPUT_DIR / f"correlation_{date.today().isoformat()}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved correlation analysis to {output_file}")

    # Summary
    log.info("\nCORRELATION SUMMARY:")
    log.info(f"  Avg Correlation (63d): {stability['current_avg_correlation']:.2f}")
    log.info(f"  1Y Avg: {stability['avg_correlation_1y']:.2f} | Vol: {stability['correlation_volatility']:.2f}")
    log.info(f"  Divergence Alerts: {len(alerts)}")
    for alert in alerts[:5]:
        log.info(f"    {alert['pair']}: {alert['direction']} ({alert['divergence']:.2f})")

    return results


if __name__ == "__main__":
    run_correlation_analysis()