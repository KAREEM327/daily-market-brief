"""
Smart Money Engine — Insider transactions, 13F filings, COT reports, Buybacks.
Tracks institutional and insider flows for conviction signals.
Based on Seyhun (1986), Lakonishok & Lee (2001), COT commercial hedger studies.
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
from bs4 import BeautifulSoup
import duckdb

# ─── Config ──────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data"
PARQUET_DIR = DATA_ROOT / "parquet"
DUCKDB_PATH = DATA_ROOT / "market.duckdb"
OUTPUT_DIR = DATA_ROOT / "smart_money"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmb.smart_money")

# Major funds to track in 13F (top holders with conviction)
TRACKED_FUNDS = [
    "Berkshire Hathaway",
    "Bridgewater Associates",
    "Renaissance Technologies",
    "Citadel Advisors",
    "Two Sigma",
    "D.E. Shaw",
    "Millennium Management",
    "Point72",
    "Tiger Global",
    "Coatue",
    "Viking Global",
    "Lone Pine",
    "Soros Fund",
    "Appaloosa",
    "Third Point",
    "Pershing Square",
    "ValueAct",
    "Elliott Management",
    "Starboard Value",
    "Jana Partners",
    "Engine No. 1",
]


# ─── Data Fetching ───────────────────────────────────────────────────────
def fetch_openinsider_recent(days: int = 7, min_value: int = 50000) -> Optional[pd.DataFrame]:
    """Fetch recent insider transactions from OpenInsider."""
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
        "vl": min_value,
        "vh": "",
        "oc": 1,
        "oe": 1,
        "sicMin": "",
        "sicMax": "",
        "sortcol": 0,
        "cnt": 500,
        "page": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"class": "tinytable"})
        if not table:
            return None

        rows = table.find_all("tr")[1:]
        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 13:
                try:
                    data.append({
                        "ticker": cols[2].text.strip(),
                        "company": cols[3].text.strip(),
                        "insider": cols[4].text.strip(),
                        "relation": cols[5].text.strip(),
                        "date": cols[6].text.strip(),
                        "type": cols[7].text.strip(),
                        "shares": int(cols[8].text.strip().replace(",", "")),
                        "price": float(cols[9].text.strip().replace("$", "").replace(",", "")),
                        "value": float(cols[10].text.strip().replace("$", "").replace(",", "")),
                        "shares_total": cols[11].text.strip(),
                        "sec_form": cols[12].text.strip(),
                    })
                except Exception:
                    continue
        return pd.DataFrame(data) if data else None
    except Exception as e:
        log.error(f"OpenInsider failed: {e}")
    return None


def fetch_openinsider_cluster_buys(days: int = 30, min_insiders: int = 2) -> pd.DataFrame:
    """Detect cluster buys (multiple insiders buying same stock)."""
    df = fetch_openinsider_recent(days, min_value=10000)
    if df is None or df.empty:
        return pd.DataFrame()

    # Filter buys only
    buys = df[df["type"].str.contains("B|Buy", case=False, na=False)].copy()
    if buys.empty:
        return pd.DataFrame()

    # Group by ticker, count unique insiders
    clusters = buys.groupby("ticker").agg(
        insider_count=("insider", "nunique"),
        total_value=("value", "sum"),
        avg_price=("price", "mean"),
        total_shares=("shares", "sum"),
        insiders=("insider", lambda x: list(x.unique())),
        dates=("date", lambda x: list(x.unique())),
    ).reset_index()

    clusters = clusters[clusters["insider_count"] >= min_insiders]
    clusters = clusters.sort_values("total_value", ascending=False)
    return clusters


def fetch_whalewisdom_13f(fund_name: str) -> Optional[pd.DataFrame]:
    """Fetch 13F holdings for a specific fund from WhaleWisdom (free tier)."""
    # WhaleWisdom has a free API but requires registration
    # For now, we'll use SEC EDGAR directly
    return None


def fetch_sec_13f_filings(cik: str, quarters: int = 4) -> Optional[pd.DataFrame]:
    """Fetch 13F filings from SEC EDGAR for a given CIK."""
    url = f"https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        "action": "getcompany",
        "CIK": cik,
        "type": "13F-HR",
        "dateb": "",
        "owner": "include",
        "count": quarters * 2,
        "search_text": "",
    }
    headers = {"User-Agent": "DailyMarketBrief/1.0 (kareem@example.com)"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Find filing links
        filings = []
        for row in soup.find_all("tr"):
            cols = row.find_all("td")
            if len(cols) >= 4 and "13F-HR" in cols[0].text:
                filing_date = cols[3].text.strip()
                doc_link = cols[1].find("a")
                if doc_link:
                    filings.append({
                        "date": filing_date,
                        "doc_url": "https://www.sec.gov" + doc_link["href"],
                    })
        return pd.DataFrame(filings) if filings else None
    except Exception as e:
        log.warning(f"SEC 13F for CIK {cik} failed: {e}")
    return None


def fetch_cot_report() -> Optional[pd.DataFrame]:
    """Fetch CFTC Commitment of Traders (disaggregated) report."""
    url = "https://www.cftc.gov/dea/newcot/f_disagg.txt"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.split("\n")

        data = []
        current_market = None
        in_section = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Market header
            if any(x in line for x in ["FUTURES ONLY", "OPTIONS AND FUTURES"]):
                in_section = True
                continue

            if in_section and line and not line.startswith(" "):
                current_market = line
                continue

            # Commercial/Non-commercial positions
            if current_market and any(x in line for x in ["COMMERCIAL", "NON-COMMERCIAL", "NONREPORTABLE"]):
                parts = re.split(r"\s+", line.strip())
                if len(parts) >= 7:
                    category = parts[0]
                    try:
                        data.append({
                            "market": current_market,
                            "category": category,
                            "long": int(parts[1].replace(",", "")),
                            "short": int(parts[2].replace(",", "")),
                            "spreading": int(parts[3].replace(",", "")) if len(parts) > 3 else 0,
                            "long_change": int(parts[4].replace(",", "")) if len(parts) > 4 else 0,
                            "short_change": int(parts[5].replace(",", "")) if len(parts) > 5 else 0,
                            "spread_change": int(parts[6].replace(",", "")) if len(parts) > 6 else 0,
                            "pct_oi": float(parts[7].replace("%", "")) if len(parts) > 7 else 0,
                        })
                    except Exception:
                        continue

        return pd.DataFrame(data) if data else None
    except Exception as e:
        log.error(f"COT report failed: {e}")
    return None


def fetch_buyback_announcements(days: int = 30) -> Optional[pd.DataFrame]:
    """Fetch recent buyback announcements (simplified - would need SEC 8-K parsing)."""
    # This would require parsing SEC 8-K filings for "Item 2.02" or "Item 5.07"
    # Placeholder for now - could use a news API or SEC RSS
    return None


def fetch_short_interest(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch short interest data for a ticker from NASDAQ (free)."""
    url = f"https://api.nasdaq.com/api/shortinterest/{ticker.upper()}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        si_data = data.get("data", {})
        return {
            "ticker": ticker.upper(),
            "short_interest": si_data.get("shortInterest"),
            "days_to_cover": si_data.get("daysToCover"),
            "short_pct_float": si_data.get("shortPercentFloat"),
            "settlement_date": si_data.get("settlementDate"),
        }
    except Exception as e:
        log.warning(f"Short interest {ticker} failed: {e}")
    return None


def fetch_institutional_ownership_changes(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch recent 13F changes for a ticker (simplified)."""
    # Would need to parse 13F-HR filings for all funds holding the ticker
    # Placeholder
    return None


# ─── Signal Generation ────────────────────────────────────────────────────
def analyze_insider_signals(insider_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze insider transactions for signals."""
    if insider_df is None or insider_df.empty:
        return {}

    # Cluster buys
    clusters = fetch_openinsider_cluster_buys(30, min_insiders=2)

    # Large single buys
    buys = insider_df[insider_df["type"].str.contains("B|Buy", case=False, na=False)]
    large_buys = buys[buys["value"] >= 100000].nlargest(10, "value")

    # CEO/CFO buys
    exec_buys = buys[buys["relation"].str.contains("CEO|CFO|President|Chief", case=False, na=False)]

    # Sell analysis
    sells = insider_df[insider_df["type"].str.contains("S|Sale|Sell", case=False, na=False)]
    large_sells = sells[sells["value"] >= 500000].nlargest(10, "value")

    return {
        "cluster_buys": clusters.head(10).to_dict("records"),
        "large_buys": large_buys.to_dict("records"),
        "exec_buys": exec_buys.to_dict("records"),
        "large_sells": large_sells.to_dict("records"),
        "net_buy_sell_ratio": len(buys) / max(len(sells), 1),
        "total_buy_value": float(buys["value"].sum()),
        "total_sell_value": float(sells["value"].sum()),
    }


def analyze_cot_signals(cot_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze COT report for commercial hedger signals."""
    if cot_df is None or cot_df.empty:
        return {}

    # Focus on commercial hedgers (smart money in futures)
    commercial = cot_df[cot_df["category"].str.contains("COMMERCIAL", case=False, na=False)]

    signals = {}
    for market in commercial["market"].unique():
        mkt_data = commercial[commercial["market"] == market]
        if len(mkt_data) == 0:
            continue

        # Net commercial position
        net_long = mkt_data["long"].sum() - mkt_data["short"].sum()
        net_change = mkt_data["long_change"].sum() - mkt_data["short_change"].sum()

        # Extreme positioning (historical percentiles would be better)
        signals[market] = {
            "net_commercial_long": int(net_long),
            "net_change": int(net_change),
            "bullish": net_long > 0 and net_change > 0,
            "bearish": net_long < 0 and net_change < 0,
        }

    # Key markets to highlight
    key_markets = ["S&P 500", "NASDAQ 100", "RUSSELL 2000", "US 10YR", "EURODOLLAR", "GOLD", "CRUDE OIL", "EURO FX", "YEN"]
    highlighted = {k: v for k, v in signals.items() if any(km.upper() in k.upper() for km in key_markets)}

    return {
        "all_markets": signals,
        "key_markets": highlighted,
        "summary": f"{len(highlighted)} key markets with commercial signals",
    }


def analyze_13f_conviction_changes() -> Dict[str, Any]:
    """Track conviction changes in tracked funds' 13F filings."""
    # This would require maintaining a database of historical 13F holdings
    # Placeholder - returns structure for when implemented
    return {
        "tracked_funds": TRACKED_FUNDS,
        "status": "requires 13F database build",
        "note": "Build historical 13F database from SEC EDGAR for conviction tracking",
    }


def run_smart_money_analysis() -> Dict[str, Any]:
    """Run full smart money analysis."""
    log.info("=" * 50)
    log.info("SMART MONEY ENGINE — INSIDER, 13F, COT, BUYBACKS")
    log.info("=" * 50)

    results = {}

    # 1. Insider Transactions (last 30 days)
    log.info("Fetching insider transactions...")
    insider_df = fetch_openinsider_recent(30, min_value=10000)
    if insider_df is not None:
        log.info(f"  Found {len(insider_df)} insider transactions")
        results["insider"] = analyze_insider_signals(insider_df)

    # 2. COT Report
    log.info("Fetching COT report...")
    cot_df = fetch_cot_report()
    if cot_df is not None:
        log.info(f"  Parsed {len(cot_df)} COT records")
        results["cot"] = analyze_cot_signals(cot_df)

    # 3. 13F Conviction (placeholder)
    results["13f_conviction"] = analyze_13f_conviction_changes()

    # 4. Short Interest for key tickers
    log.info("Fetching short interest for key tickers...")
    short_data = {}
    for ticker in ["SPY", "QQQ", "IWM", "TSLA", "NVDA", "AMD", "GME", "AMC"]:
        si = fetch_short_interest(ticker)
        if si:
            short_data[ticker] = si
    results["short_interest"] = short_data

    # 5. Buyback announcements (placeholder)
    results["buybacks"] = {"status": "requires SEC 8-K parser"}

    # Save
    output_file = OUTPUT_DIR / f"smart_money_{date.today().isoformat()}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved smart money analysis to {output_file}")

    # Summary
    log.info("\nSMART MONEY SUMMARY:")
    if "insider" in results:
        ins = results["insider"]
        log.info(f"  Insider: {ins.get('net_buy_sell_ratio', 0):.2f} buy/sell ratio | "
                 f"${ins.get('total_buy_value', 0)/1e6:.1f}M buys vs ${ins.get('total_sell_value', 0)/1e6:.1f}M sells")
        log.info(f"  Cluster buys: {len(ins.get('cluster_buys', []))}")
    if "cot" in results:
        cot = results["cot"]
        log.info(f"  COT: {cot.get('summary', 'N/A')}")
        for mkt, sig in cot.get("key_markets", {}).items():
            bias = "BULLISH" if sig.get("bullish") else ("BEARISH" if sig.get("bearish") else "NEUTRAL")
            log.info(f"    {mkt}: {bias} (net comm: {sig['net_commercial_long']:+,})")

    return results


if __name__ == "__main__":
    run_smart_money_analysis()