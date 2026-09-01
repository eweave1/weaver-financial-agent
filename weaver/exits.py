"""EOD exit scanner — checks open positions against stop and 2R targets.

Conditions checked (all require a numeric stop_price in the buy file):
  stop_breached    current <= stop
  approaching_stop current > stop and within approaching_stop_pct% above stop
  2r_profit        current >= entry + 2 * (entry - stop)

Called by `wf scan-exits`. Silence if no alerts. On stop_breached Telegram
failure, retries once then writes a distinct FAILED marker to stderr so a
grep across logs surfaces it.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Any, Optional

import requests

_SEVERITY: dict[str, int] = {
    "stop_breached": 0,
    "approaching_stop": 1,
    "2r_profit": 2,
}


def scan_exits(
    vault_path: Path,
    config: dict[str, Any],
    _positions: Optional[list[dict[str, Any]]] = None,
    _prices: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    """Return alert dicts for any triggered exit conditions across open positions.

    _positions and _prices are injection points for tests, bypassing file I/O
    and network calls. When omitted, reads from the vault and fetches via yfinance.

    Alerts are sorted by severity: stop_breached → approaching_stop → 2r_profit.
    A position without a numeric stop_price is skipped silently.
    """
    from weaver.journal import scan_open_positions

    positions = (
        _positions if _positions is not None
        else scan_open_positions(vault_path, {})
    )

    tickers = [p["ticker"] for p in positions if p.get("ticker")]
    current_prices = (
        _prices if _prices is not None
        else _fetch_current_prices(tickers)
    )

    cfg = config.get("exits", {})
    approaching_pct = float(cfg.get("approaching_stop_pct", 2.0)) / 100

    alerts: list[dict[str, Any]] = []

    for pos in positions:
        ticker = pos["ticker"]
        entry_raw = pos.get("entry_price")
        stop_raw = pos.get("stop_price")
        current_raw = current_prices.get(ticker)

        if entry_raw is None or current_raw is None or stop_raw is None:
            continue

        try:
            entry = float(entry_raw)
            stop = float(stop_raw)
            current = float(current_raw)
        except (TypeError, ValueError):
            continue

        risk = entry - stop
        if risk <= 0:
            continue  # invalid stop — above entry, skip

        pnl_pct = (current - entry) / entry

        if current <= stop:
            alerts.append({
                "ticker": ticker,
                "condition": "stop_breached",
                "current": current,
                "entry": entry,
                "stop": stop,
                "risk": risk,
                "pnl_pct": pnl_pct,
            })
        elif (current - stop) / stop <= approaching_pct:
            alerts.append({
                "ticker": ticker,
                "condition": "approaching_stop",
                "current": current,
                "entry": entry,
                "stop": stop,
                "risk": risk,
                "pct_above_stop": (current - stop) / stop,
                "pnl_pct": pnl_pct,
            })

        target_2r = entry + 2 * risk
        if current >= target_2r:
            alerts.append({
                "ticker": ticker,
                "condition": "2r_profit",
                "current": current,
                "entry": entry,
                "stop": stop,
                "risk": risk,
                "target_2r": target_2r,
                "pnl_pct": pnl_pct,
            })

    alerts.sort(key=lambda a: _SEVERITY.get(a["condition"], 99))
    return alerts


def format_exit_message(
    alerts: list[dict[str, Any]],
    scan_date: date,
) -> Optional[str]:
    """Format alerts as plain text for Telegram. Returns None when alerts is empty."""
    if not alerts:
        return None

    lines = [f"Exit Alerts — {scan_date} EOD", ""]

    for alert in alerts:
        ticker = alert["ticker"]
        current = alert["current"]
        entry = alert["entry"]
        stop = alert["stop"]
        pnl_pct = alert.get("pnl_pct", 0.0)
        condition = alert["condition"]

        if condition == "stop_breached":
            lines.append(f"STOP BREACHED: {ticker} @ ${current:,.2f} (stop ${stop:,.2f})")
            lines.append(f"  Entry ${entry:,.2f} | P&L {pnl_pct:+.1%}")
        elif condition == "approaching_stop":
            pct_above = alert.get("pct_above_stop", 0.0)
            lines.append(
                f"APPROACHING STOP: {ticker} @ ${current:,.2f} "
                f"({pct_above:.1%} above stop ${stop:,.2f})"
            )
            lines.append(f"  Entry ${entry:,.2f} | P&L {pnl_pct:+.1%}")
        elif condition == "2r_profit":
            target_2r = alert.get("target_2r", entry)
            lines.append(
                f"2R PROFIT: {ticker} @ ${current:,.2f} ({pnl_pct:+.1%})"
            )
            lines.append(
                f"  Entry ${entry:,.2f} | Stop ${stop:,.2f} | "
                f"2R target ${target_2r:,.2f}"
            )

        lines.append("")

    return "\n".join(lines).rstrip()


def send_telegram(message: str, bot_token: str, chat_id: str) -> bool:
    """POST message to Telegram Bot API. Returns True on success."""
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message},
            timeout=15,
        )
        return resp.status_code == 200 and bool(resp.json().get("ok"))
    except Exception:
        return False


def deliver_alerts(
    message: str,
    bot_token: str,
    chat_id: str,
    has_stop_breach: bool,
) -> bool:
    """Send alert message to Telegram, with one retry for stop_breached alerts.

    Returns True on success. On failure:
    - stop_breached present: retries once, then prints a loud FAILED marker to
      stderr so `grep FAILED logs/exits-*.log` surfaces it.
    - other conditions only: prints a quiet warning to stderr.
    """
    import sys

    success = send_telegram(message, bot_token, chat_id)

    if not success and has_stop_breach:
        time.sleep(2)
        success = send_telegram(message, bot_token, chat_id)

    if not success:
        if has_stop_breach:
            print(
                "TELEGRAM DELIVERY FAILED — STOP BREACH ALERT NOT SENT",
                file=sys.stderr,
            )
        else:
            print("Warning: Telegram delivery failed.", file=sys.stderr)

    return success


# ── private helpers ───────────────────────────────────────────────────────────


def _fetch_current_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch last close price for each ticker via yfinance. Skips failures silently."""
    if not tickers:
        return {}
    prices: dict[str, float] = {}
    try:
        import yfinance as yf

        for ticker in set(tickers):
            try:
                hist = yf.download(ticker, period="2d", progress=False, auto_adjust=True)
                if hist is None or hist.empty:
                    continue
                close = hist["Close"]
                if hasattr(close, "columns"):
                    close = close.iloc[:, 0]
                series = close.dropna()
                if not series.empty:
                    prices[ticker] = float(series.iloc[-1])
            except Exception:
                pass
    except Exception:
        pass
    return prices
