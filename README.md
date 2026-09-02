# Daily Market Brief — Personal Intelligence System

A free, automated daily market brief for personal capital allocation decisions. Built with proven quantitative techniques from academic literature and practitioner research.

## What It Does

Delivers a concise, actionable daily brief (<3 min read) at 5:30 AM ET covering:

| Layer | Technique | Sources |
|-------|-----------|---------|
| **Regime** | Markov-switching (Hamilton, Ang & Timmermann) | SPY, QQQ, IWM, sectors, macro, FX, commodities |
| **Correlation Risk** | Rolling + regime-conditional (Longin & Solnik, Forbes & Rigobon) | 63d/252d windows, crisis benchmarks |
| **Historical Analogues** | k-NN on 20-factor state vector (Leinweber, Kritzman) | 20+ years history, forward returns at 5/21/63d |
| **Options Structure** | GEX, VRP, Skew, Dealer positioning (SpotGamma, SqueezeMetrics) | SPY/QQQ/IWM chains, CBOE VIX/SKEW/PCR |
| **Smart Money** | Insider clusters, 13F conviction, COT commercials (Seyhun, Lakonishok) | OpenInsider, SEC EDGAR, CFTC |

## Output Format (Mobile-First)

```
## DAILY MARKET BRIEF — January 15, 2025

### REGIME
**Bull** | Confidence: 78% | Duration: 42d
SPY: $582.30 | Key Level: $585 (GEX wall) / $565 (support)
VIX: 14.2 | Term: Contango | SKEW: 128
OIL: $78.50 | DXY: 103.2 | 10Y: 4.18% | 2s10s: -35bps

### CORRELATION ALERT
XLE/SPY decoupling (current 0.31 vs 252d 0.68, div 0.37)

### HISTORICAL ANALOGUE
Top 10 analogues → 5d: +0.34% (win 60%), 21d: +1.82% (win 70%).
Best: 2021-03-15 (sim: 0.89)

### OPTIONS STRUCTURE
Dealers LONG gamma (Net GEX $2.3B). Zero gamma 585 (+0.46% from spot).
VRP +2.1 pts (+12.3%).

### SMART MONEY
Insiders: 1.8 buy/sell ($45M buy vs $25M sell). Clusters: 3.
COT: 4 key markets with commercial signals. S&P 500: BULLISH (net comm: +145K).

### ACTIONABLE
1. Long SPY 5800/5850 call spread (21d), risk $180, target $420, 2.3:1 — regime bull + dealers long gamma + positive analogue
2. Long XLE vs SPY pair — energy decoupling at extremes, mean reversion candidate
3. Watch 5850 GEX wall — break targets 5900, reject favors 5750
```

## Architecture

```
┌─────────────────┐
│  GitHub Actions │  (Free: 2000 min/mo, runs 5:30 AM ET Mon-Fri)
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Data Ingest    │────▶│  Parquet +       │
│  (yfinance,     │     │  DuckDB Lake     │
│  FRED, CBOE,    │     │                  │
│  OpenInsider)   │     └────────┬─────────┘
└─────────────────┘              │
         │                       ▼
         │              ┌──────────────────┐
         │              │  5 Engines       │
         │              │  (Parallel)      │
         │              │  • Regime        │
         │              │  • Correlation   │
         │              │  • Analogue      │
         │              │  • Options       │
         │              │  • Smart Money   │
         │              └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│         Brief Generator                 │
│  (OpenRouter free models: GLM-5.2,      │
│   Nemotron-3-Ultra, Gemma-4-31B)        │
└────────────────┬────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  Telegram /    │
        │  GitHub Artifact│
        └────────────────┘
```

## Free Execution

| Platform | Free Tier | Daily Cost |
|----------|-----------|------------|
| **GitHub Actions** | 2,000 min/mo | ~5 min/day = 150 min/mo ✓ |
| Oracle Cloud | 4 ARM CPUs always-free | $0 |
| Google Cloud Run | 2M requests/mo | $0 |

**Default: GitHub Actions** — no server management, native cron, artifact storage.

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/KAREEM327/daily-market-brief.git
cd daily-market-brief
pip install -r requirements.txt
```

### 2. API Keys (Free Tiers)

| Key | Source | Free Tier |
|-----|--------|-----------|
| `FRED_API_KEY` | https://fred.stlouisfed.org/docs/api/api_key.html | Unlimited |
| `ALPHA_VANTAGE_KEY` | https://www.alphavantage.co/support/#api-key | 25 req/day |
| `OPENROUTER_API_KEY` | https://openrouter.ai/keys | Free models available |
| `TELEGRAM_BOT_TOKEN` | @BotFather on Telegram | Free |
| `TELEGRAM_CHAT_ID` | @userinfobot on Telegram | Free |

Add to GitHub Secrets: Settings → Secrets → Actions → New repository secret

### 3. Test Locally

```bash
# Run full pipeline
python scripts/run_pipeline.py --force

# Or individual engines
python scripts/data_ingest.py
python scripts/regime_engine.py
python scripts/correlation_engine.py
python scripts/analogue_engine.py
python scripts/options_engine.py
python scripts/smart_money.py
python scripts/brief_generator.py
```

### 4. Deploy

Push to GitHub — the workflow runs automatically at 5:30 AM ET Mon-Fri.

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

Trigger manually: Actions → Daily Market Brief → Run workflow

## Data Sources (All Free)

| Category | Sources |
|----------|---------|
| **Equities/ETFs** | yfinance (Yahoo Finance) — unlimited |
| **Macro/FRED** | FRED API — unlimited with key |
| **Options** | yfinance chains, CBOE delayed quotes (PCR, SKEW, VIX term) |
| **Volatility** | CBOE VIX, VX3M, SKEW index |
| **Sentiment** | AAII (weekly), CNN Fear & Greed |
| **Insider** | OpenInsider screener |
| **COT** | CFTC disaggregated report |
| **Internals** | NYSE/NASDAQ A/D, new highs/lows, TRIN, TICK via yfinance |

## Project Structure

```
daily-market-brief/
├── .github/workflows/daily-brief.yml   # GitHub Actions workflow
├── requirements.txt                      # Python dependencies
├── scripts/
│   ├── run_pipeline.py                  # Master entry point
│   ├── data_ingest.py                   # All data fetching
│   ├── regime_engine.py                 # Markov regime detection
│   ├── correlation_engine.py            # Rolling/conditional correlations
│   ├── analogue_engine.py               # k-NN historical analogues
│   ├── options_engine.py                # GEX, VRP, skew, dealer pos
│   ├── smart_money.py                   # Insider, 13F, COT, buybacks
│   └── brief_generator.py               # LLM synthesis + delivery
├── data/
│   ├── raw/                             # Raw API responses
│   ├── parquet/                         # Partitioned parquet lake
│   ├── regime/                          # Engine outputs
│   ├── correlation/
│   ├── analogue/
│   ├── options/
│   ├── smart_money/
│   ├── briefs/                          # Final briefs (markdown)
│   └── logs/                            # Pipeline run logs
└── market.duckdb                        # DuckDB for fast queries
```

## Customization

### Add Tickers
Edit `REGIME_TICKERS` in `scripts/regime_engine.py` and `CORE_ASSETS` in `scripts/correlation_engine.py`.

### Adjust Regime Sensitivity
Modify `DEFAULT_THRESHOLD` (default 5%) in `scripts/regime_engine.py`.

### Change Analogue Features
Edit `FEATURE_CONFIG` in `scripts/analogue_engine.py` — weights control importance.

### Modify Brief Format
Edit `SYSTEM_PROMPT` and `USER_PROMPT_TEMPLATE` in `scripts/brief_generator.py`.

### Schedule
Change cron in `.github/workflows/daily-brief.yml`:
- `30 9 * * 1-5` = 5:30 AM ET Mon-Fri (current)
- `0 10 * * 1-5` = 6:00 AM ET
- `30 13 * * 1-5` = 9:30 AM ET (market open)

## Academic References

1. **Hamilton, J.D. (1989)** — "A New Approach to the Economic Analysis of Nonstationary Time Series"
2. **Ang, A. & Timmermann, A. (2012)** — "Regime Changes and Financial Markets"
3. **Longin, F. & Solnik, B. (2001)** — "Extreme Correlation of International Equity Markets"
4. **Forbes, K. & Rigobon, R. (2002)** — "No Contagion, Only Interdependence"
5. **Leinweber, D. (2003)** — "Nerds on Wall Street"
6. **Kritzman, M. et al. (2010)** — "Regime Shifts and Asset Allocation"
7. **Seyhun, H.N. (1986)** — "Insiders' Profits, Costs of Trading, and Market Efficiency"
8. **Lakonishok, J. & Lee, I. (2001)** — "Are Insider Trades Informative?"
9. **Aronson, D. (2013)** — "Evidence-Based Technical Analysis" (Ch. 5: Permutation Tests)

## License

MIT — Use freely for personal investment research.

## Disclaimer

This tool provides analytical information only. Not financial advice. Past patterns ≠ future results. Always do your own research and risk management.