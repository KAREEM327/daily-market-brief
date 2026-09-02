"""
Regime Engine — Markov regime detection for SPY and key assets.
Uses the proven framework from markov-ai-analyst (Roan @RohOnChain).
Outputs regime state, transition matrix, persistence, forecasts.
"""

from __future__ import annotations

import json
import sys
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf
import duckdb
from hmmlearn import hmm

# ─── Config ──────────────────────────────────────────────────────────────
DATA_ROOT = Path("/Users/blackstarr/CLAUDE COWORK/daily-market-brief/data")
PARQUET_DIR = DATA_ROOT / "parquet"
DUCKDB_PATH = DATA_ROOT / "market.duckdb"
OUTPUT_DIR = DATA_ROOT / "regime"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmb.regime")

# Fixed state indices: 0 = Bear, 1 = Sideways, 2 = Bull
STATES = ["Bear", "Sideways", "Bull"]
STATE_COLORS = ["#FF4444", "#FFAA00", "#00AA44"]

DEFAULT_WINDOW = 20
DEFAULT_THRESHOLD = 0.05  # ±5% rolling return
DEFAULT_YEARS = 10
DEFAULT_MIN_TRAIN = 252

# Key tickers for regime analysis
REGIME_TICKERS = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "DIA": "Dow Jones",
    "TLT": "Long Treasuries",
    "GLD": "Gold",
    "USO": "Crude Oil",
    "UUP": "Dollar Index",
    "EEM": "Emerging Markets",
    "FXI": "China",
    "EWG": "Germany",
    "EWJ": "Japan",
    "XLF": "Financials",
    "XLK": "Technology",
    "XLE": "Energy",
}


# ─── Data Loading ────────────────────────────────────────────────────────
def fetch_close_series(ticker: str, years: int = DEFAULT_YEARS) -> pd.Series:
    """Fetch daily close series via yfinance."""
    end = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    start = end - pd.DateOffset(years=years)

    df = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna()
    close.name = ticker
    return close


def load_from_parquet(ticker: str) -> Optional[pd.Series]:
    """Load close series from parquet if available."""
    paths = list(PARQUET_DIR.glob(f"equities/{ticker}*/**/*.parquet"))
    if not paths:
        paths = list(PARQUET_DIR.glob(f"equities/**/{ticker}*.parquet"))
    if paths:
        try:
            df = pd.read_parquet(paths[0])
            if "ticker" in df.columns:
                df = df[df["ticker"] == ticker]
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            close_col = "Close" if "Close" in df.columns else "close"
            return df[close_col].dropna().rename(ticker)
        except Exception as e:
            log.warning(f"Parquet load failed for {ticker}: {e}")
    return None


def get_close_series(ticker: str, prefer_parquet: bool = True) -> pd.Series:
    """Get close series, trying parquet first then yfinance."""
    if prefer_parquet:
        series = load_from_parquet(ticker)
        if series is not None and len(series) > 200:
            return series
    return fetch_close_series(ticker)


# ─── Core Regime Functions (from markov-ai-analyst) ─────────────────────
def clean_futures_rolls(close: pd.Series, max_daily_move: float = 0.10) -> Tuple[pd.Series, int]:
    """Neutralize roll artifacts in continuous futures price series."""
    daily_returns = close.pct_change().fillna(0.0)
    roll_mask = daily_returns.abs() > max_daily_move
    rolls_removed = int(roll_mask.sum())

    if rolls_removed == 0:
        return close, 0

    cleaned = daily_returns.copy()
    cleaned[roll_mask] = 0.0
    cleaned.iloc[0] = 0.0

    factors = (1.0 + cleaned)
    factors.iloc[0] = 1.0
    clean_close = close.iloc[0] * factors.cumprod()
    clean_close.name = close.name
    return clean_close, rolls_removed


def label_regimes(
    close: pd.Series,
    window: int = DEFAULT_WINDOW,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.Series:
    """Label each day from the trailing `window`-day return."""
    rolling_return = close.pct_change(window)
    labels = pd.Series(1, index=close.index, dtype=int)  # default Sideways
    labels[rolling_return > threshold] = 2  # Bull
    labels[rolling_return < -threshold] = 0  # Bear
    return labels.loc[rolling_return.notna()]


def build_transition_matrix(labels: pd.Series) -> np.ndarray:
    """MLE estimate of the 3x3 transition matrix by counting transitions."""
    counts = np.zeros((3, 3), dtype=float)
    arr = np.asarray(labels, dtype=int)
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def nstep_forecast(matrix: np.ndarray, n: int) -> np.ndarray:
    """Chapman-Kolmogorov: the n-step transition matrix is P raised to n."""
    return np.linalg.matrix_power(matrix, n)


def stationary_distribution(matrix: np.ndarray) -> np.ndarray:
    """Left eigenvector of P for eigenvalue 1, normalised to sum to 1."""
    eigvals, eigvecs = np.linalg.eig(matrix.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.abs(np.real(eigvecs[:, idx]))
    return vec / vec.sum()


def regime_signal(matrix: np.ndarray, current_state: int) -> float:
    """Signal: P(next=Bull | current) - P(next=Bear | current)."""
    return float(matrix[current_state, 2] - matrix[current_state, 0])


def fit_hmm(returns: pd.Series, n_components: int = 3, random_state: int = 42):
    """Fit a Gaussian HMM on daily returns."""
    X = returns.dropna().to_numpy(dtype=float).reshape(-1, 1)
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=200,
        random_state=random_state,
    )
    model.fit(X)
    hidden_states = model.predict(X)
    return model, hidden_states


def hmm_summary(close: pd.Series) -> Dict[str, Any]:
    """Build HMM section of analysis."""
    try:
        model, hidden_states = fit_hmm(close.pct_change().dropna(), n_components=3)
    except Exception as exc:
        return {"available": False, "reason": f"hmm runtime error: {exc}"}

    if model is None:
        return {"available": False, "reason": "hmmlearn failed"}

    means = np.array([model.means_[k][0] for k in range(model.n_components)])
    order = np.argsort(means)  # ascending mean return
    rank_names = ["Bear", "Sideways", "Bull"]
    regimes = []
    for rank, k in enumerate(order):
        regimes.append({
            "label": rank_names[rank],
            "latent_state": int(k),
            "mean_daily_return": float(means[k]),
        })
    # Current regime
    current_latent = hidden_states[-1]
    current_regime_label = None
    for r in regimes:
        if r["latent_state"] == current_latent:
            current_regime_label = r["label"]
            break

    return {
        "available": True,
        "regimes": regimes,
        "current_regime": current_regime_label,
        "current_latent_state": int(current_latent),
        "transition_matrix": model.transmat_.tolist(),
    }


# ─── Walk-Forward Backtest ──────────────────────────────────────────────
def walk_forward_backtest(
    close: pd.Series,
    labels: pd.Series,
    min_train: int = DEFAULT_MIN_TRAIN,
) -> Dict[str, Any]:
    """No-lookahead walk-forward backtest."""
    daily_returns = close.pct_change().dropna()
    common_index = labels.index.intersection(daily_returns.index)
    labels = labels.loc[common_index]
    daily_returns = daily_returns.loc[common_index]

    if len(labels) < min_train + 30:
        return {"sharpe": float("nan"), "max_drawdown": float("nan"), "n_trades": 0}

    lab = np.asarray(labels, dtype=int)
    rets = daily_returns.to_numpy(dtype=float)

    counts = np.zeros((3, 3), dtype=float)
    for i in range(min_train - 1):
        counts[lab[i], lab[i + 1]] += 1.0

    strategy_returns = np.empty(len(lab) - 1 - min_train, dtype=float)
    for k, t in enumerate(range(min_train, len(lab) - 1)):
        row_sums = counts.sum(axis=1, keepdims=True)
        safe = np.where(row_sums == 0, 1.0, row_sums)
        P_t = counts / safe

        current_state = lab[t]
        signal = float(P_t[current_state, 2] - P_t[current_state, 0])
        position = float(np.sign(signal))
        strategy_returns[k] = position * rets[t + 1]

        counts[lab[t - 1], lab[t]] += 1.0

    sr = strategy_returns
    std = sr.std(ddof=1) if len(sr) > 1 else 0.0
    if std == 0 or not np.isfinite(std):
        sharpe = float("nan")
    else:
        sharpe = float(sr.mean() / std * np.sqrt(252))

    equity = (1.0 + sr).cumprod()
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min()) if len(drawdown) else float("nan")

    return {"sharpe": sharpe, "max_drawdown": max_dd, "n_trades": int(len(sr))}


# ─── Correlation Engine ─────────────────────────────────────────────────
def compute_correlation_matrix(tickers: List[str], window: int = 63) -> pd.DataFrame:
    """Compute rolling correlation matrix for given tickers."""
    all_closes = {}
    for t in tickers:
        try:
            series = get_close_series(t)
            all_closes[t] = series
        except Exception as e:
            log.warning(f"Failed to load {t}: {e}")

    if len(all_closes) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(all_closes)
    returns = df.pct_change().dropna()
    corr = returns.rolling(window).corr()

    # Get latest correlation matrix
    latest = corr.iloc[-len(tickers):, :].copy()
    latest.index = [t for t in tickers if t in all_closes]
    latest.columns = [t for t in tickers if t in all_closes]

    return latest


def compute_regime_conditional_correlations(
    spy_labels: pd.Series,
    tickers: List[str],
    window: int = 63
) -> Dict[str, pd.DataFrame]:
    """Compute correlations conditioned on SPY regime."""
    spy_returns = get_close_series("SPY").pct_change().dropna()
    common_idx = spy_labels.index.intersection(spy_returns.index)
    spy_labels = spy_labels.loc[common_idx]
    spy_returns = spy_returns.loc[common_idx]

    # Get returns for all tickers
    all_returns = {"SPY": spy_returns}
    for t in tickers:
        if t == "SPY":
            continue
        try:
            series = get_close_series(t)
            rets = series.pct_change().dropna()
            common = rets.index.intersection(common_idx)
            if len(common) > 50:
                all_returns[t] = rets.loc[common]
        except Exception:
            pass

    if len(all_returns) < 2:
        return {}

    returns_df = pd.DataFrame(all_returns).dropna()
    labels_aligned = spy_labels.reindex(returns_df.index).ffill().dropna()
    returns_df = returns_df.loc[labels_aligned.index]

    result = {}
    for regime_id, regime_name in enumerate(STATES):
        mask = labels_aligned == regime_id
        if mask.sum() > 20:
            regime_returns = returns_df[mask]
            corr = regime_returns.corr()
            result[regime_name] = corr

    return result


# ─── Main Analysis ──────────────────────────────────────────────────────
def analyze_ticker(ticker: str) -> Dict[str, Any]:
    """Full regime analysis for a single ticker."""
    log.info(f"Analyzing {ticker}...")

    close = get_close_series(ticker)

    # Clean futures rolls if needed
    if ticker.endswith("=F") or ticker in ["GC=F", "CL=F", "BZ=F", "HG=F"]:
        close, rolls = clean_futures_rolls(close)
        if rolls:
            log.info(f"  Cleaned {rolls} futures roll days for {ticker}")

    # Observable model (rolling return labels)
    labels = label_regimes(close)
    current_state = int(labels.iloc[-1])
    current_regime = STATES[current_state]

    # Transition matrix
    trans_matrix = build_transition_matrix(labels)

    # Persistence (diagonal)
    persist_prob = float(trans_matrix[current_state, current_state])

    # N-step forecasts
    forecast_5d = nstep_forecast(trans_matrix, 5)
    forecast_20d = nstep_forecast(trans_matrix, 20)

    # Stationary distribution
    stationary = stationary_distribution(trans_matrix)

    # Signal
    signal = regime_signal(trans_matrix, current_state)

    # Regime duration
    duration = 1
    for i in range(len(labels) - 2, -1, -1):
        if labels.iloc[i] == current_state:
            duration += 1
        else:
            break

    # Historical mix
    hist_mix = labels.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0)
    hist_mix_dict = {STATES[i]: float(hist_mix.iloc[i]) for i in range(3)}

    # Walk-forward backtest
    wf = walk_forward_backtest(close, labels)

    # HMM
    hmm_result = hmm_summary(close)

    # Chart data (last 365 days)
    chart_data = []
    recent = close.tail(365)
    recent_labels = labels.reindex(recent.index).ffill()
    for dt, price in recent.items():
        reg = recent_labels.get(dt, 1)
        chart_data.append({
            "date": dt.strftime("%Y-%m-%d"),
            "close": round(float(price), 2),
            "regime": int(reg),
        })

    return {
        "ticker": ticker,
        "name": REGIME_TICKERS.get(ticker, ticker),
        "current_price": round(float(close.iloc[-1]), 2),
        "current_regime": current_regime,
        "current_state": current_state,
        "regime_duration_days": duration,
        "persistence_probability": round(persist_prob, 3),
        "signal": round(signal, 3),
        "signal_label": "LONG BIAS" if signal > 0.1 else ("SHORT BIAS" if signal < -0.1 else "NEUTRAL"),
        "transition_matrix": [[round(x, 3) for x in row] for row in trans_matrix],
        "forecast_5d": [[round(x, 3) for x in row] for row in forecast_5d],
        "forecast_20d": [[round(x, 3) for x in row] for row in forecast_20d],
        "stationary_distribution": {STATES[i]: round(float(stationary[i]), 3) for i in range(3)},
        "historical_mix": {k: round(v, 3) for k, v in hist_mix_dict.items()},
        "walk_forward": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in wf.items()},
        "hmm": hmm_result,
        "chart_data": chart_data,
        "as_of": date.today().isoformat(),
    }


def run_regime_analysis() -> Dict[str, Any]:
    """Run regime analysis on all key tickers."""
    log.info("=" * 50)
    log.info("REGIME ENGINE — FULL ANALYSIS")
    log.info("=" * 50)

    results = {}
    for ticker in REGIME_TICKERS:
        try:
            results[ticker] = analyze_ticker(ticker)
        except Exception as e:
            log.error(f"Failed to analyze {ticker}: {e}")
            results[ticker] = {"error": str(e)}

    # Correlation analysis
    log.info("Computing correlation matrices...")
    tickers_list = list(REGIME_TICKERS.keys())
    corr_matrix = compute_correlation_matrix(tickers_list)
    if not corr_matrix.empty:
        results["correlation_matrix"] = corr_matrix.round(3).to_dict()

    # Regime-conditional correlations (using SPY labels)
    spy_labels = label_regimes(get_close_series("SPY"))
    cond_corrs = compute_regime_conditional_correlations(spy_labels, tickers_list)
    if cond_corrs:
        results["regime_conditional_correlations"] = {
            regime: corr.round(3).to_dict() for regime, corr in cond_corrs.items()
        }

    # Save results
    output_file = OUTPUT_DIR / f"regime_{date.today().isoformat()}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved regime analysis to {output_file}")

    # Also save latest for quick access
    latest_file = OUTPUT_DIR / "regime_latest.json"
    with open(latest_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    log.info("\nREGIME SUMMARY:")
    for ticker, data in results.items():
        if ticker in REGIME_TICKERS and "error" not in data:
            log.info(f"  {ticker}: {data['current_regime']} ({data['regime_duration_days']}d) | "
                     f"Signal: {data['signal_label']} ({data['signal']:.2f}) | "
                     f"Persist: {data['persistence_probability']:.0%}")

    return results


if __name__ == "__main__":
    run_regime_analysis()