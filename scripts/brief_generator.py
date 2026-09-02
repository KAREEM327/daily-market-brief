#!/usr/bin/env python3
"""
Brief Generator — Synthesizes all engine outputs into the final daily brief.
"""
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dmb.brief")

DATA_ROOT = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_ROOT / "briefs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_latest_engine_output(engine: str) -> Optional[Dict[str, Any]]:
    """Load the latest JSON output from an engine."""
    engine_dir = DATA_ROOT / engine
    if not engine_dir.exists():
        return None
    files = sorted(engine_dir.glob("*.json"), reverse=True)
    if not files:
        return None
    today = date.today().isoformat()
    for f in files:
        if today in f.name:
            try:
                with open(f) as fp:
                    return json.load(fp)
            except json.JSONDecodeError:
                log.warning(f"Failed to decode JSON from {f}")
                continue
    try:
        with open(files[0]) as fp:
            return json.load(fp)
    except json.JSONDecodeError:
        log.warning(f"Failed to decode JSON from {files[0]}")
        return None


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


def get_regime_summary(regime: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key regime info for SPY."""
    spy = regime.get("SPY", {})
    return {
        "regime": spy.get("current_regime", "Unknown"),
        "price": spy.get("current_price", 0),
        "signal": spy.get("signal_label", "NEUTRAL"),
        "signal_val": spy.get("signal", 0),
        "duration": spy.get("regime_duration_days", 0),
        "persist": spy.get("persistence_probability", 0),
        "forecast_5d": spy.get("forecast_5d", {}),
        "forecast_20d": spy.get("forecast_20d", {}),
    }


def get_vix_from_options(options: Dict[str, Any]) -> float:
    """Get VIX spot from options data."""
    return options.get("vix_spot", 0) or 0


def get_oil_price(regime: Dict[str, Any]) -> float:
    """Get oil price from USO regime data."""
    uso = regime.get("USO", {})
    return uso.get("current_price", 0)


def get_dxy(regime: Dict[str, Any]) -> float:
    """Get DXY from UUP regime data."""
    uup = regime.get("UUP", {})
    return uup.get("current_price", 0)


def get_tnx(regime: Dict[str, Any]) -> float:
    """Get 10Y yield proxy from TLT."""
    tlt = regime.get("TLT", {})
    # TLT is inverse to yields, approximate
    return tlt.get("current_price", 0)


def get_2s10s(regime: Dict[str, Any]) -> float:
    """Get 2s10s spread proxy."""
    # Use IEF (7-10Y) vs SHY (1-3Y) spread
    ief = regime.get("IEF", {})
    shy = regime.get("SHY", {})
    if ief and shy:
        return ief.get("current_price", 0) - shy.get("current_price", 0)
    return 0


def format_correlation_alerts(corr: Dict[str, Any]) -> str:
    """Format correlation divergence alerts."""
    if not corr:
        return "No correlation data available"
    
    alerts = []
    # Check regime-conditional correlations
    rc = corr.get("regime_conditional_correlations", {})
    for regime_name, matrix in rc.items():
        if isinstance(matrix, dict):
            for pair, val in matrix.items():
                if isinstance(val, (int, float)) and abs(val) > 0.8:
                    alerts.append(f"{pair}: r={val:.2f} in {regime_name} regime")
    
    if not alerts:
        return "No significant correlation divergences detected"
    return "\n".join(f"• {a}" for a in alerts[:5])


def format_analogue(analogue: Dict[str, Any]) -> str:
    """Format historical analogue output."""
    if not analogue or "matches" not in analogue:
        return "Insufficient history for analogue matching"
    
    matches = analogue.get("matches", [])
    if not matches:
        return "No historical analogues found"
    
    lines = []
    for m in matches[:3]:
        dt = m.get("date", "N/A")
        fwd5 = m.get("forward_5d", 0)
        fwd21 = m.get("forward_21d", 0)
        fwd63 = m.get("forward_63d", 0)
        lines.append(f"{dt} → 5d: {fwd5:.1%} | 21d: {fwd21:.1%} | 63d: {fwd63:.1%}")
    return "\n".join(f"• {l}" for l in lines)


def format_options(options: Dict[str, Any]) -> str:
    """Format options structure."""
    if not options:
        return "No options data"
    
    vrp_obj = options.get("vrp", {})
    if isinstance(vrp_obj, dict):
        vrp = vrp_obj.get("vrp_pct") or vrp_obj.get("vrp")
    else:
        vrp = vrp_obj
    
    vix = options.get("vix_spot", 0)
    rv = options.get("realized_vol_21d", 0)
    
    lines = [
        f"VIX Spot: {vix:.1f}",
        f"21d Realized Vol: {rv:.1f}%",
    ]
    if vrp is not None and isinstance(vrp, (int, float)):
        lines.append(f"VRP: {vrp:.1f}%")
    
    return "\\n".join(lines)


def format_futures(futures: Dict[str, Any]) -> str:
    """Format futures term structure."""
    if not futures:
        return "No futures data"
    
    lines = []
    for name, data in futures.items():
        if isinstance(data, dict):
            spot = data.get("spot", 0)
            term = data.get("term_structure", {})
            if spot:
                lines.append(f"{name}: Spot ${spot:.2f}")
                if term:
                    # Show front/back spread
                    contracts = list(term.items())
                    if len(contracts) >= 2:
                        front = contracts[0][1].get("price", 0)
                        back = contracts[-1][1].get("price", 0)
                        if front and back:
                            spread = ((back - front) / front) * 100
                            lines.append(f"  Term: {spread:+.1f}% (front→back)")
    return "\n".join(lines) if lines else "Futures data processing..."


def format_smart_money(sm: Dict[str, Any]) -> str:
    """Format smart money signals."""
    if not sm:
        return "No smart money data"
    
    lines = []
    
    # Insider clusters
    insider = sm.get("insider_clusters", {})
    if insider and isinstance(insider, dict):
        clusters = insider.get("clusters", [])
        if clusters:
            lines.append("Insider Clusters:")
            for c in clusters[:3]:
                lines.append(f"  {c.get('ticker', 'N/A')}: {c.get('buyers', 0)} buyers, ${c.get('total_value', 0)/1e6:.1f}M")
    
    # COT
    cot = sm.get("cot", {})
    if cot and isinstance(cot, dict):
        for market, data in list(cot.items())[:3]:
            if isinstance(data, dict):
                comm = data.get("commercial_net", 0)
                noncomm = data.get("noncommercial_net", 0)
                lines.append(f"COT {market}: Comm {comm:+,}, NonComm {noncomm:+,}")
    
    # 13F
    f13 = sm.get("13f_conviction", {})
    if f13 and isinstance(f13, dict):
        top = f13.get("top_conviction_increases", [])
        if top:
            lines.append("13F Conviction Increases:")
            for t in top[:3]:
                lines.append(f"  {t.get('manager', 'N/A')}: {t.get('ticker', 'N/A')} +{t.get('change_pct', 0):.0f}%")
    
    return "\n".join(lines) if lines else "Smart money data processing..."


def generate_brief_fallback(engines: Dict[str, Any]) -> str:
    """Generate a structured brief without LLM synthesis."""
    today = date.today().strftime("%B %d, %Y")
    
    regime = engines.get("regime", {})
    correlation = engines.get("correlation", {})
    analogue = engines.get("analogue", {})
    options = engines.get("options", {})
    futures = engines.get("futures", {})
    smart_money = engines.get("smart_money", {})
    
    rs = get_regime_summary(regime)
    vix = get_vix_from_options(options)
    oil = get_oil_price(regime)
    dxy = get_dxy(regime)
    tnx = get_tnx(regime)
    two_s_ten = get_2s10s(regime)
    
    brief = f"""## DAILY MARKET BRIEF — {today}

### REGIME
**{rs['regime']}** | Confidence: {rs['persist']*100:.0f}% | Duration: {rs['duration']}d
SPY: ${rs['price']:.2f} | Signal: {rs['signal']} ({rs['signal_val']:+.2f})
VIX: {vix:.1f} | Term: N/A | SKEW: N/A
OIL: ${oil:.2f} | DXY: {dxy:.2f} | 10Y Proxy: ${tnx:.2f} | 2s10s Proxy: {two_s_ten:+.2f}

### CORRELATION ALERT
{format_correlation_alerts(correlation)}

### HISTORICAL ANALOGUE
{format_analogue(analogue)}

### OPTIONS STRUCTURE
{format_options(options)}

### FUTURES MARKETS
{format_futures(futures)}

### SMART MONEY
{format_smart_money(smart_money)}

### ACTIONABLE
Review engine outputs directly for specific trading ideas.
- SPY regime: {rs['regime']} — {rs['signal']}
- GLD: Bull regime (Long bias)
- XLE: Bull regime (Long bias)  
- USO: Bull regime (Long bias)

---
Generated at {datetime.now().strftime('%H:%M:%S ET')}
"""
    return brief


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
        import requests
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
    engines = load_all_engines()
    if not engines:
        log.error("No engine data available")
        return {"error": "No engine data"}
    brief = generate_brief_fallback(engines)
    brief_path = save_brief(brief)
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
        print(result["brief"])
