#!/usr/bin/env python3
"""
Brief Generator — Synthesizes all engine outputs into the final daily brief.
Uses OpenRouter (free models) for intelligent LLM synthesis.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
from jinja2 import Template

# ─── Config ──────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_ROOT / "briefs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmb.brief")

# OpenRouter config
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Use free models
FREE_MODELS = [
    "z-ai/glm-5.2:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
]

# ─── Load Engine Outputs ─────────────────────────────────────────────────
def load_latest_engine_output(engine: str) -> Optional[Dict[str, Any]]:
    """Load the latest JSON output from an engine."""
    engine_dir = DATA_ROOT / engine
    if not engine_dir.exists():
        return None

    files = sorted(engine_dir.glob("*.json"), reverse=True)
    if not files:
        return None

    # Prefer today's file
    today = date.today().isoformat()
    for f in files:
        if today in f.name:
            with open(f) as fp:
                return json.load(fp)

    # Fallback to latest
    with open(files[0]) as fp:
        return json.load(fp)


def load_all_engines() -> Dict[str, Any]:
    """Load all engine outputs."""
    engines = ["regime", "correlation", "analogue", "options", "smart_money", "futures"]
    results = {}
    for engine in engines:
        data = load_latest_engine_output(engine)
        if data:
            results[engine] = data
            log.info(f"Loaded {engine}: {len(str(data))} chars")
        else:
            log.warning(f"No data for {engine}")
    return results


# ─── Prompt Templates ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior macro strategist writing a daily market brief for a sophisticated investor.
Your brief is concise, actionable, and written in plain English — no jargon without purpose.
You synthesize regime, correlation, historical analogues, options structure, smart money flows, and futures markets.
Every claim must be grounded in the data provided. No speculation. No fluff.

FORMAT (markdown, mobile-first, <3 min read):

## DAILY MARKET BRIEF — {{ date }}

### REGIME
**{{ spy_regime }}** | Confidence: {{ regime_confidence }}% | Duration: {{ regime_duration }}d
SPY: ${{ spy_price }} | Key Level: ${{ gex_wall }} (GEX wall) / ${{ support }} (support)
VIX: {{ vix }} | Term: {{ vix_term }} | SKEW: {{ skew }}
OIL: ${{ oil }} | DXY: {{ dxy }} | 10Y: {{ tnx }}% | 2s10s: {{ t10y2y }}bps

### CORRELATION ALERT
{{ correlation_alert }}

### HISTORICAL ANALOGUE
{{ analogue_summary }}

### OPTIONS STRUCTURE
{{ options_summary }}

### FUTURES MARKETS
{{ futures_summary }}

### SMART MONEY
{{ smart_money_summary }}

### ACTIONABLE
{{ actionable_ideas }}

---
RULES:
- Use $ for prices, % for yields/returns, bps for spreads
- Bold key numbers, not labels
- One line per insight
- Actionable = specific idea with risk/reward (e.g., "Long SPY 5600/5650 call spread, risk $150, target $350, 2.3:1")
- If data missing, say "Data unavailable" not "N/A"
- No emojis, no markdown headers beyond what's shown
"""

USER_PROMPT_TEMPLATE = Template("""Generate the daily market brief for {{ date }} using the following engine outputs.

=== REGIME ENGINE ==={{ regime_data }}

=== CORRELATION ENGINE ==={{ correlation_data }}

=== ANALOGUE ENGINE ==={{ analogue_data }}

=== OPTIONS ENGINE ==={{ options_data }}

=== FUTURES ENGINE ==={{ futures_data }}

=== SMART MONEY ENGINE ==={{ smart_money_data }}

Write the brief following the exact format in the system prompt. Be precise. Every number must come from the data above.
""")


def call_openrouter(messages: List[Dict[str, str]], model: str = None) -> Optional[str]:
    """Call OpenRouter API with fallback through free models."""
    if not OPENROUTER_API_KEY:
        log.warning("OPENROUTER_API_KEY not set — skipping LLM synthesis")
        return None

    models_to_try = [model] if model else FREE_MODELS

    for m in models_to_try:
        try:
            log.info(f"Calling OpenRouter with {m}...")
            resp = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/KAREEM327/daily-market-brief",
                    "X-Title": "Daily Market Brief",
                },
                json={
                    "model": m,
                    "messages": messages,
                    "max_tokens": 2000,
                    "temperature": 0.3,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            log.info(f"Success with {m}: {len(content)} chars")
            return content
        except Exception as e:
            log.warning(f"Model {m} failed: {e}")
            continue

    log.error("All models failed")
    return None


def extract_key_data(engines: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key data points for the brief template."""
    data = {"date": date.today().strftime("%B %d, %Y")}

    # Regime
    regime = engines.get("regime", {})
    if "SPY" in regime:
        spy = regime["SPY"]
        data["spy_regime"] = spy.get("current_regime", "Unknown")
        data["regime_confidence"] = int(spy.get("persistence_probability", 0) * 100)
        data["regime_duration"] = spy.get("regime_duration_days", 0)
        data["spy_price"] = spy.get("current_price", 0)

    # GEX wall from options
    options = engines.get("options", {})
    spy_gex = options.get("SPY_gex", {})
    dealer = options.get("SPY_dealer", {})
    if dealer.get("zero_gamma_level"):
        data["gex_wall"] = dealer["zero_gamma_level"]
    else:
        data["gex_wall"] = "N/A"

    # Support level (simplified)
    data["support"] = round(data["spy_price"] * 0.97, 1) if data["spy_price"] else "N/A"

    # VIX, term structure, skew
    data["vix"] = options.get("vix_spot", "N/A")
    vix_term = options.get("vix_term_structure", [])
    if vix_term and len(vix_term) >= 2:
        front = vix_term[0].get("value", 0)
        back = vix_term[1].get("value", 0)
        data["vix_term"] = "Contango" if back > front else "Backwardation"
    else:
        data["vix_term"] = "N/A"
    data["skew"] = options.get("cboe_skew", "N/A")

    # Macro
    regime_data = engines.get("regime", {})
    if "USO" in regime_data:
        data["oil"] = regime_data["USO"].get("current_price", "N/A")
    if "UUP" in regime_data:
        data["dxy"] = regime_data["UUP"].get("current_price", "N/A")
    if "^TNX" in regime_data:
        data["tnx"] = regime_data["^TNX"].get("current_price", "N/A")
    # 2s10s from FRED would need parsing
    data["t10y2y"] = "N/A"

    # Correlation alert
    corr = engines.get("correlation", {})
    alerts = corr.get("divergence_alerts", [])
    if alerts:
        top_alert = alerts[0]
        data["correlation_alert"] = f"{top_alert['pair']} {top_alert['direction']} (current {top_alert['current']} vs 252d {top_alert['historical_252d']}, div {top_alert['divergence']})"
    else:
        data["correlation_alert"] = "No significant divergences detected."

    # Analogue
    analogue = engines.get("analogue", {})
    if analogue.get("forward_returns"):
        fw = analogue["forward_returns"]
        h5 = fw.get("5d", {})
        h21 = fw.get("21d", {})
        data["analogue_summary"] = (
            f"Top 10 analogues → 5d: {h5.get('mean', 0):+.2f}% (win {h5.get('win_rate', 0):.0f}%), "
            f"21d: {h21.get('mean', 0):+.2f}% (win {h21.get('win_rate', 0):.0f}%). "
            f"Best: {analogue['top_analogues'][0]['date'] if analogue.get('top_analogues') else 'N/A'}"
        )
    else:
        data["analogue_summary"] = "Insufficient historical matches."

    # Options
    spy_dealer = options.get("SPY_dealer", {})
    if spy_dealer:
        data["options_summary"] = (
            f"Dealers {spy_dealer.get('dealer_gamma', 'UNKNOWN')} gamma "
            f"(Net GEX ${spy_dealer.get('net_gex_billions', 0):.1f}B). "
            f"Zero gamma {spy_dealer.get('zero_gamma_level', 'N/A')} "
            f"({spy_dealer.get('flip_distance_pct', 'N/A'):+.2f}% from spot). "
            f"VRP {options.get('vrp', {}).get('vrp', 'N/A')} pts "
            f"({options.get('vrp', {}).get('vrp_pct', 'N/A'):+.1f}%)."
        )
    else:
        data["options_summary"] = "Options data unavailable."

    # Futures
    futures = engines.get("futures", {})
    if futures and "error" not in futures:
        # Build futures summary
        parts = []
        # Term structure highlights
        ts = futures.get("term_structure", {})
        for comm, term in ts.items():
            if len(term) >= 2:
                front = term[0]["price"]
                second = term[1]["price"]
                spread = second - front
                if abs(spread) > 0.1:  # only mention if meaningful
                    direction = "contango" if spread > 0 else "backwardation"
                    parts.append(f"{comm} {direction} {abs(spread):.2f}")
        # Calendar spreads
        cs = futures.get("calendar_spreads", {})
        for comm, spread in cs.items():
            if abs(spread["spread"]) > 0.1:
                parts.append(f"{comm} cal spread {spread['spread']:+.2f}")
        # COT alerts
        alerts = futures.get("alerts", [])
        if alerts:
            parts.extend(alerts[:2])  # top 2 alerts
        if parts:
            data["futures_summary"] = " | ".join(parts[:3])
        else:
            data["futures_summary"] = "No extreme term structure or positioning detected."
    else:
        data["futures_summary"] = "Futures data unavailable."

    # Smart money
    sm = engines.get("smart_money", {})
    insider = sm.get("insider", {})
    cot = sm.get("cot", {})
    data["smart_money_summary"] = (
        f"Insiders: {insider.get('net_buy_sell_ratio', 0):.2f} buy/sell "
        f"(${insider.get('total_buy_value', 0)/1e6:.0f}M buy vs ${insider.get('total_sell_value', 0)/1e6:.0f}M sell). "
        f"Clusters: {len(insider.get('cluster_buys', []))}. "
        f"COT: {cot.get('summary', 'N/A')}."
    )

    # Actionable ideas (placeholder - LLM will generate from data)
    data["actionable_ideas"] = "See LLM synthesis."

    return data


def generate_brief_llm(engines: Dict[str, Any]) -> str:
    """Generate brief using LLM synthesis."""
    key_data = extract_key_data(engines)

    # Build user prompt with full engine data
    user_prompt = USER_PROMPT_TEMPLATE.render(
        date=key_data["date"],
        regime_data=json.dumps(engines.get("regime", {}), default=str)[:8000],
        correlation_data=json.dumps(engines.get("correlation", {}), default=str)[:8000],
        analogue_data=json.dumps(engines.get("analogue", {}), default=str)[:8000],
        options_data=json.dumps(engines.get("options", {}), default=str)[:8000],
        futures_data=json.dumps(engines.get("futures", {}), default=str)[:8000],
        smart_money_data=json.dumps(engines.get("smart_money", {}), default=str)[:8000],
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(**key_data)},
        {"role": "user", "content": user_prompt},
    ]

    brief = call_openrouter(messages)

    if brief is None:
        # Fallback: template-based brief
        brief = generate_brief_template(key_data, engines)

    return brief


def generate_brief_template(key_data: Dict[str, Any], engines: Dict[str, Any]) -> str:
    """Fallback template-based brief (no LLM)."""
    template = f"""## DAILY MARKET BRIEF — {key_data['date']}

### REGIME
**{key_data['spy_regime']}** | Confidence: {key_data['regime_confidence']}% | Duration: {key_data['regime_duration']}d
SPY: ${key_data['spy_price']} | Key Level: ${key_data['gex_wall']} (GEX wall) / ${key_data['support']} (support)
VIX: {key_data['vix']} | Term: {key_data['vix_term']} | SKEW: {key_data['skew']}
OIL: ${key_data['oil']} | DXY: {key_data['dxy']} | 10Y: {key_data['tnx']}% | 2s10s: {key_data['t10y2y']}bps

### CORRELATION ALERT
{key_data['correlation_alert']}

### HISTORICAL ANALOGUE
{key_data['analogue_summary']}

### OPTIONS STRUCTURE
{key_data['options_summary']}

### FUTURES MARKETS
{key_data['futures_summary']}

### SMART MONEY
{key_data['smart_money_summary']}

### ACTIONABLE
[LLM synthesis unavailable — review engine outputs manually]
"""
    return template


def save_brief(brief: str) -> Path:
    """Save brief to file."""
    today = date.today().isoformat()
    output_file = OUTPUT_DIR / f"brief_{today}.md"
    with open(output_file, "w") as f:
        f.write(brief)
    log.info(f"Saved brief to {output_file}")
    return output_file


def send_telegram(brief: str, brief_path: Path) -> bool:
    """Send brief via Telegram bot."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        log.warning("Telegram credentials not set")
        return False

    try:
        # Send as document
        with open(brief_path, "rb") as f:
            files = {"document": (brief_path.name, f, "text/markdown")}
            data = {
                "chat_id": chat_id,
                "caption": f"📊 Daily Market Brief — {date.today().strftime('%b %d, %Y')}",
            }
            resp = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendDocument",
                data=data,
                files=files,
                timeout=30,
            )
            resp.raise_for_status()
            log.info("Brief sent via Telegram")
            return True
    except Exception as e:
        log.error(f"Telegram send failed: {e}")
    return False


def run_brief_generation() -> Dict[str, Any]:
    """Run complete brief generation pipeline."""
    log.info("=" * 50)
    log.info("BRIEF GENERATOR — SYNTHESIS")
    log.info("=" * 50)

    # Load all engine outputs
    engines = load_all_engines()

    if not engines:
        log.error("No engine data available")
        return {"error": "No engine data"}

    # Generate brief
    brief = generate_brief_llm(engines)

    # Save
    brief_path = save_brief(brief)

    # Send via Telegram
    sent = send_telegram(brief, brief_path)

    log.info("Brief generation complete")
    return {
        "brief": brief,
        "path": str(brief_path),
        "telegram_sent": sent,
        "engines_used": list(engines.keys()),
    }


if __name__ == "__main__":
    result = run_brief_generation()
    if "brief" in result:
        print("\n" + "=" * 50)
        print("BRIEF PREVIEW:")
        print("=" * 50)
        print(result["brief"][:2000])