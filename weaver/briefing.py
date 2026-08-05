"""Morning briefing — scan full watchlist and write a prioritized note.

Writes to Trading/Briefings/YYYY-MM-DD.md in the Obsidian vault.
Without --analyze: data sections filled, synthesis section omitted.
With --analyze: one LLM call synthesizes all gathered data.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from weaver.predictions import _coerce_date, _parse_frontmatter
from weaver.research import fetch_news, fetch_snapshot

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_SYSTEM_PROMPT = (
    "You are a financial research assistant analyzing stocks for a human trader "
    "who makes all final decisions. Surface balanced analysis — do not give "
    "investment advice."
)

_MACRO_TICKERS = frozenset({"SPY", "QQQ"})


def generate_briefing(
    vault_path: Path,
    config: dict[str, Any],
    analyze: bool = False,
    briefing_date: Optional[date] = None,
    api_key: Optional[str] = None,
    model: str = "deepseek/deepseek-v4-pro",
    timeout: int = 90,
    _ticker_data: Optional[dict[str, Any]] = None,
    _vix: Optional[float] = None,
    _ai_synthesis: Optional[str] = None,
    _proposals: Optional[list[dict[str, Any]]] = None,
    progress_cb: Optional[Callable[[str, bool], None]] = None,
) -> tuple[Path, list[str]]:
    """Scan all watchlist tickers and write a morning briefing note.

    _ticker_data, _vix, and _ai_synthesis are injection points for tests.
    Returns (filepath, needs_attention_tickers).
    """
    from weaver.config import get_watchlist

    today = briefing_date or date.today()
    all_tickers = get_watchlist(config)
    all_ticker_set = set(all_tickers)

    if _ticker_data is None:
        extra_tickers = _get_open_item_tickers(vault_path, all_ticker_set)
        tickers_to_fetch = all_tickers + extra_tickers

        ticker_data: dict[str, Any] = {}
        for ticker in tickers_to_fetch:
            try:
                snap = fetch_snapshot(ticker)
                company_name = snap.get("name", "") if not snap.get("error") else ""
                if ticker in _MACRO_TICKERS:
                    news: list[dict[str, Any]] = []
                else:
                    news = fetch_news(ticker, limit=5, company_name=company_name)
                ticker_data[ticker] = {
                    "snapshot": snap,
                    "news": news,
                    "error": snap.get("error"),
                }
            except Exception as exc:
                ticker_data[ticker] = {
                    "snapshot": {},
                    "news": [],
                    "error": str(exc),
                }
            if progress_cb:
                progress_cb(ticker, not bool(ticker_data[ticker].get("error")))
    else:
        ticker_data = _ticker_data

    vix = _vix if _vix is not None else fetch_vix()
    sector_headlines = find_sector_news(ticker_data)

    watchlist_only = {
        t: d
        for t, d in ticker_data.items()
        if t in all_ticker_set and t not in _MACRO_TICKERS
    }
    buckets = bucket_tickers(watchlist_only, config)

    current_prices: dict[str, float] = {
        t: d["snapshot"]["current_price"]
        for t, d in ticker_data.items()
        if not d.get("error") and d["snapshot"].get("current_price") is not None
    }
    open_positions = scan_open_positions(vault_path, current_prices)
    open_predictions = scan_open_predictions(vault_path, current_prices=current_prices)

    synthesis: Optional[str] = None
    if analyze:
        if _ai_synthesis is not None:
            synthesis = _ai_synthesis
        else:
            synthesis = synthesize_with_ai(
                ticker_data=ticker_data,
                buckets=buckets,
                macro_tickers=_MACRO_TICKERS,
                vix=vix,
                open_positions=open_positions,
                open_predictions=open_predictions,
                model=model,
                api_key=api_key or "",
                timeout=timeout,
            )

    proposals: list[dict[str, Any]] = []
    if analyze:
        try:
            from weaver.proposals import generate_proposals as _gen_proposals
            proposals = _gen_proposals(
                ticker_data=ticker_data,
                buckets=buckets,
                open_positions=open_positions,
                open_predictions=open_predictions,
                current_prices=current_prices,
                vault_path=vault_path,
                config=config,
                api_key=api_key or "",
                model=model,
                timeout=timeout,
                _proposals=_proposals,
            )
        except Exception:
            proposals = []

    note = build_briefing_note(
        briefing_date=today,
        ticker_data=ticker_data,
        macro_tickers=_MACRO_TICKERS,
        vix=vix,
        sector_headlines=sector_headlines,
        buckets=buckets,
        open_positions=open_positions,
        open_predictions=open_predictions,
        synthesis=synthesis,
        vault_path=vault_path,
        proposals=proposals,
    )

    briefing_dir = vault_path / "Trading" / "Briefings"
    briefing_dir.mkdir(parents=True, exist_ok=True)
    filepath = briefing_dir / f"{today}.md"
    filepath.write_text(note, encoding="utf-8")

    needs_tickers = [e["ticker"] for e in buckets.get("needs_attention", [])]
    return filepath, needs_tickers


def fetch_vix() -> Optional[float]:
    """Fetch the current VIX level from yfinance."""
    try:
        import yfinance as yf

        info = yf.Ticker("^VIX").fast_info
        price = getattr(info, "last_price", None)
        return round(float(price), 2) if price else None
    except Exception:
        return None


def find_sector_news(
    ticker_data: dict[str, Any],
    min_tickers: int = 3,
) -> list[str]:
    """Return headlines appearing in the news feed of min_tickers+ distinct tickers.

    Cross-ticker headlines indicate sector-wide news rather than company-specific events.
    """
    headline_tickers: dict[str, set[str]] = {}
    for ticker, data in ticker_data.items():
        for item in data.get("news", []):
            title = item.get("title", "").strip()
            if title:
                headline_tickers.setdefault(title, set()).add(ticker)

    sector = [
        title
        for title, tickers in headline_tickers.items()
        if len(tickers) >= min_tickers
    ]
    return sector[:5]


def days_to_earnings(snapshot: dict[str, Any]) -> Optional[int]:
    """Return days until next earnings, or None if unavailable or already past."""
    ne = snapshot.get("next_earnings")
    if not ne or str(ne) == "N/A":
        return None
    try:
        dte = (date.fromisoformat(str(ne)[:10]) - date.today()).days
        return dte if dte >= 0 else None
    except (ValueError, TypeError):
        return None


def bucket_tickers(
    ticker_data: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, list]:
    """Sort tickers into needs_attention, watching, quiet, and failed buckets.

    Thresholds come from config.yaml briefing section.
    """
    cfg = config.get("briefing", {})
    # config stores as percentage (3.0 = 3%); snapshot day_change_pct is a decimal fraction
    move_thresh = float(cfg.get("move_threshold_pct", 3.0)) / 100
    vol_thresh = float(cfg.get("volume_threshold", 1.5))
    earn_days = int(cfg.get("earnings_warning_days", 7))
    watch_move = float(cfg.get("watching_move_threshold_pct", 1.5)) / 100
    watch_earn_days = int(cfg.get("watching_earnings_days", 14))

    needs_attention: list[dict[str, Any]] = []
    watching: list[dict[str, Any]] = []
    quiet: list[dict[str, Any]] = []
    failed: list[str] = []

    for ticker, data in ticker_data.items():
        if data.get("error"):
            failed.append(ticker)
            continue

        snap = data.get("snapshot", {})
        news = data.get("news", [])
        change_pct = float(snap.get("day_change_pct", 0) or 0)
        vol_ratio = float(
            snap.get("volume_ratio_projected") or snap.get("volume_ratio") or 1.0
        )
        dte = days_to_earnings(snap)
        abs_change = abs(change_pct)

        entry: dict[str, Any] = {
            "ticker": ticker,
            "snapshot": snap,
            "news": news,
            "change_pct": change_pct,
            "vol_ratio": vol_ratio,
            "days_to_earnings": dte,
        }

        # News is supporting detail only — not a qualifying criterion for any tier.
        if (
            abs_change >= move_thresh
            or vol_ratio >= vol_thresh
            or (dte is not None and dte <= earn_days)
        ):
            needs_attention.append(entry)
        elif abs_change >= watch_move or (dte is not None and earn_days < dte <= watch_earn_days):
            watching.append(entry)
        else:
            quiet.append(entry)

    needs_attention.sort(key=lambda x: abs(x.get("change_pct", 0) or 0), reverse=True)
    watching.sort(key=lambda x: abs(x.get("change_pct", 0) or 0), reverse=True)

    return {
        "needs_attention": needs_attention,
        "watching": watching,
        "quiet": quiet,
        "failed": failed,
    }


def scan_open_positions(
    vault_path: Path,
    current_prices: dict[str, float],
) -> list[dict[str, Any]]:
    """Return open buy entries from the journal, excluding those referenced by a sell.

    Approximation: buy entries linked by a sell's linked_buy field are treated as closed.
    """
    journal_dir = vault_path / "Trading" / "Journal"
    if not journal_dir.exists():
        return []

    sold_stems: set[str] = set()
    for f in journal_dir.glob("*.md"):
        fm = _parse_frontmatter(f.read_text(encoding="utf-8"))
        if fm.get("action") == "sell":
            lb = fm.get("linked_buy", "")
            if lb:
                m = re.search(r"\[\[(.+?)\]\]", str(lb))
                if m:
                    sold_stems.add(m.group(1))

    positions: list[dict[str, Any]] = []
    for f in sorted(journal_dir.glob("*.md")):
        fm = _parse_frontmatter(f.read_text(encoding="utf-8"))
        if fm.get("action") != "buy":
            continue
        if f.stem in sold_stems:
            continue

        ticker = str(fm.get("ticker", "")).upper()
        entry_price = fm.get("price")
        current = current_prices.get(ticker)

        pos: dict[str, Any] = {
            "ticker": ticker,
            "entry_date": fm.get("entry_date"),
            "entry_price": entry_price,
            "current_price": current,
            "quantity": fm.get("quantity"),
            "time_horizon": fm.get("time_horizon"),
            "file_stem": f.stem,
        }
        if entry_price is not None and current is not None:
            pos["pnl_pct"] = (float(current) - float(entry_price)) / float(entry_price)

        positions.append(pos)

    return positions


def scan_open_predictions(
    vault_path: Path,
    current_prices: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    """Return all open (unresolved) predictions sorted by file name.

    If current_prices is provided, each prediction gets current_price.
    trigger_price comes from the explicit frontmatter 'trigger' field only —
    never inferred from reasoning text.
    """
    pred_dir = vault_path / "Trading" / "Predictions"
    if not pred_dir.exists():
        return []

    today = date.today()
    prices = current_prices or {}
    preds: list[dict[str, Any]] = []
    for f in sorted(pred_dir.glob("*.md")):
        fm = _parse_frontmatter(f.read_text(encoding="utf-8"))
        if fm.get("status") != "open":
            continue

        resolve_by = _coerce_date(fm.get("resolve_by"))
        ticker = str(fm.get("ticker", "")).upper()
        current = prices.get(ticker)

        # Read explicit trigger from frontmatter only
        trigger: Optional[float] = None
        raw_trigger = fm.get("trigger")
        if raw_trigger is not None:
            try:
                trigger = float(raw_trigger)
            except (TypeError, ValueError):
                pass

        pred: dict[str, Any] = {
            "ticker": ticker,
            "direction": fm.get("direction"),
            "confidence": fm.get("confidence"),
            "resolve_by": resolve_by,
            "prediction_date": _coerce_date(fm.get("prediction_date")),
            "file_stem": f.stem,
            "current_price": current,
            "trigger_price": trigger,
        }

        if resolve_by is not None:
            days_left = (resolve_by - today).days
            pred["days_left"] = days_left
            pred["overdue"] = days_left < 0

        preds.append(pred)

    return preds


def synthesize_with_ai(
    ticker_data: dict[str, Any],
    buckets: dict[str, list],
    macro_tickers: frozenset[str],
    vix: Optional[float],
    open_positions: list[dict[str, Any]],
    open_predictions: list[dict[str, Any]],
    model: str,
    api_key: str,
    timeout: int = 90,
) -> str:
    """Call OpenRouter/DeepSeek for a 3-4 sentence synthesis of today's briefing.

    Returns the synthesis text or an error string — never raises.
    """
    if not api_key:
        return "AI synthesis unavailable: OPENROUTER_API_KEY not set"

    prompt = _build_synthesis_prompt(
        ticker_data=ticker_data,
        buckets=buckets,
        macro_tickers=macro_tickers,
        vix=vix,
        open_positions=open_positions,
        open_predictions=open_predictions,
    )

    try:
        resp = requests.post(
            _OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            },
            timeout=timeout,
        )
    except Exception as exc:
        return f"AI synthesis unavailable: network error: {exc}"

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text[:200])
        except Exception:
            detail = resp.text[:200]
        return f"AI synthesis unavailable: API returned {resp.status_code}: {detail}"

    try:
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"AI synthesis unavailable: unexpected response: {exc}"


def build_briefing_note(
    briefing_date: date,
    ticker_data: dict[str, Any],
    macro_tickers: frozenset[str],
    vix: Optional[float],
    sector_headlines: list[str],
    buckets: dict[str, list],
    open_positions: list[dict[str, Any]],
    open_predictions: list[dict[str, Any]],
    synthesis: Optional[str],
    vault_path: Path,
    proposals: Optional[list[dict[str, Any]]] = None,
) -> str:
    lines: list[str] = []

    lines += ["---", f"date: {briefing_date}", "type: briefing", "---", ""]
    lines += [f"# Morning Briefing — {briefing_date}", ""]

    # Trade proposals first — most actionable item at top of the briefing
    if proposals:
        lines += ["## Trade proposals", ""]
        for prop in proposals:
            lines.append(prop.get("text", f"**{prop['ticker']}** — proposal unavailable"))
            lines.append("")
    elif proposals is not None:
        # analyze ran but found no setups
        lines += ["## Trade proposals", "", "*No setups today.*", ""]

    if synthesis is not None:
        lines += ["## AI synthesis", "", synthesis, ""]

    # --- Tier 1: Macro ---
    lines += ["## Market overview", ""]
    macro_parts = []
    for ticker in ["SPY", "QQQ"]:
        snap = ticker_data.get(ticker, {}).get("snapshot", {})
        price = snap.get("current_price")
        change = snap.get("day_change_pct")
        if price is not None and change is not None:
            sign = "+" if change >= 0 else ""
            macro_parts.append(f"{ticker}: {sign}{change:.1%} (${price:,.2f})")
        elif price is not None:
            macro_parts.append(f"{ticker}: ${price:,.2f}")
    if vix is not None:
        macro_parts.append(f"VIX: {vix:.1f}")

    lines.append("  |  ".join(macro_parts) if macro_parts else "*Market data unavailable.*")
    lines.append("")

    # --- Tier 2: AI-sector news ---
    lines += ["## AI-sector news", ""]
    if sector_headlines:
        for headline in sector_headlines:
            lines.append(f"- {headline}")
    else:
        lines.append("*No major sector news.*")
    lines.append("")

    # --- Tier 3: Watchlist ---
    lines += ["## Watchlist", ""]

    needs = buckets.get("needs_attention", [])
    watching = buckets.get("watching", [])
    quiet = buckets.get("quiet", [])
    failed = buckets.get("failed", [])

    lines.append(f"### Needs attention ({len(needs)})")
    lines.append("")
    if needs:
        for entry in needs:
            lines.extend(_format_needs_attention_entry(entry, vault_path, briefing_date))
    else:
        lines += ["*Nothing flagged today.*", ""]

    lines.append(f"### Watching ({len(watching)})")
    lines.append("")
    if watching:
        for entry in watching:
            lines.append(_format_watching_entry(entry))
        lines.append("")
    else:
        lines += ["*Nothing in watch tier.*", ""]

    lines.append("### Quiet")
    lines.append("")
    if quiet:
        lines.append(", ".join(e["ticker"] for e in quiet))
        lines.append("")
    else:
        lines += ["*No tickers in quiet tier.*", ""]

    if failed:
        lines.append(f"*Data unavailable: {', '.join(sorted(failed))}*")
        lines.append("")

    # --- Open positions and predictions ---
    lines += ["## Open positions and predictions", ""]

    lines += ["### Open journal entries", ""]
    if open_positions:
        lines.append("| Ticker | Date | Entry | Current | P&L | Horizon |")
        lines.append("|--------|------|-------|---------|-----|---------|")
        for pos in open_positions:
            ticker = pos["ticker"]
            entry_date = pos.get("entry_date", "")
            entry_price = pos.get("entry_price")
            current = pos.get("current_price")
            pnl = pos.get("pnl_pct")
            horizon = pos.get("time_horizon", "")

            e_str = f"${float(entry_price):,.2f}" if entry_price is not None else "—"
            c_str = f"${float(current):,.2f}" if current is not None else "—"
            p_str = f"{pnl:+.1%}" if pnl is not None else "—"
            lines.append(f"| {ticker} | {entry_date} | {e_str} | {c_str} | {p_str} | {horizon} |")
        lines.append("")
    else:
        lines += ["*No open journal entries.*", ""]

    lines += ["### Open predictions", ""]
    if open_predictions:
        lines.append("| Ticker | Direction | Conf | Current | Trigger | Resolve by | Status |")
        lines.append("|--------|-----------|------|---------|---------|------------|--------|")
        for pred in open_predictions:
            ticker = pred["ticker"]
            direction = pred.get("direction", "")
            conf = pred.get("confidence")
            resolve_by = pred.get("resolve_by", "")
            days_left = pred.get("days_left")
            overdue = pred.get("overdue", False)
            current = pred.get("current_price")
            trigger = pred.get("trigger_price")

            conf_str = f"{int(float(conf) * 100)}%" if conf is not None else "—"
            current_str = f"${current:,.2f}" if current is not None else "—"
            trigger_str = _format_trigger(trigger, current, direction)

            if overdue and days_left is not None:
                status = f"overdue {abs(days_left)}d"
            elif days_left is not None:
                status = f"{days_left}d left"
            else:
                status = "open"

            lines.append(
                f"| {ticker} | {direction} | {conf_str} | {current_str} | {trigger_str} | {resolve_by} | {status} |"
            )
        lines.append("")
    else:
        lines += ["*No open predictions.*", ""]

    return "\n".join(lines)


# ── private helpers ───────────────────────────────────────────────────────────

def _format_trigger(
    trigger: Optional[float],
    current: Optional[float],
    direction: Optional[str],
) -> str:
    """Format a trigger price with unambiguous above/below/TRIGGERED status.

    'above' and 'below' describe where current is relative to the trigger.
    TRIGGERED means the condition implied by direction has been met.
    """
    if trigger is None:
        return "—"
    if current is None:
        return f"${trigger:,.2f}"

    dist_pct = abs(current - trigger) / trigger
    d = str(direction or "").lower()

    if d == "up" and current >= trigger:
        return f"${trigger:,.2f} (TRIGGERED)"
    if d == "down" and current <= trigger:
        return f"${trigger:,.2f} (TRIGGERED)"
    if current > trigger:
        return f"${trigger:,.2f} ({dist_pct:.1%} above)"
    if current < trigger:
        return f"${trigger:,.2f} ({dist_pct:.1%} below)"
    return f"${trigger:,.2f} (at trigger)"


def _get_open_item_tickers(vault_path: Path, watchlist_set: set[str]) -> list[str]:
    """Return tickers from open positions/predictions that aren't in the watchlist."""
    extra: set[str] = set()

    journal_dir = vault_path / "Trading" / "Journal"
    if journal_dir.exists():
        sold_stems: set[str] = set()
        for f in journal_dir.glob("*.md"):
            fm = _parse_frontmatter(f.read_text(encoding="utf-8"))
            if fm.get("action") == "sell":
                lb = fm.get("linked_buy", "")
                if lb:
                    m = re.search(r"\[\[(.+?)\]\]", str(lb))
                    if m:
                        sold_stems.add(m.group(1))
        for f in journal_dir.glob("*.md"):
            fm = _parse_frontmatter(f.read_text(encoding="utf-8"))
            if fm.get("action") == "buy" and f.stem not in sold_stems:
                t = str(fm.get("ticker", "")).upper()
                if t and t not in watchlist_set:
                    extra.add(t)

    pred_dir = vault_path / "Trading" / "Predictions"
    if pred_dir.exists():
        for f in pred_dir.glob("*.md"):
            fm = _parse_frontmatter(f.read_text(encoding="utf-8"))
            if fm.get("status") == "open":
                t = str(fm.get("ticker", "")).upper()
                if t and t not in watchlist_set:
                    extra.add(t)

    return sorted(extra)


def _build_synthesis_prompt(
    ticker_data: dict[str, Any],
    buckets: dict[str, list],
    macro_tickers: frozenset[str],
    vix: Optional[float],
    open_positions: list[dict[str, Any]],
    open_predictions: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "Synthesize today's market context for an AI/semiconductor-focused trading portfolio in 3-4 sentences.",
        "Focus on what genuinely matters. 'Quiet day, no action needed' is a valid and preferred output when nothing significant is happening.",
        "Do not invent information. Base your synthesis only on the data below.",
        "",
        "---",
        "",
        "## Macro",
    ]

    for ticker in ["SPY", "QQQ"]:
        if ticker in ticker_data:
            snap = ticker_data[ticker].get("snapshot", {})
            price = snap.get("current_price")
            change = snap.get("day_change_pct")
            if price is not None and change is not None:
                sign = "+" if change >= 0 else ""
                lines.append(f"{ticker}: ${price:,.2f} ({sign}{change:.1%})")

    if vix is not None:
        lines.append(f"VIX: {vix:.1f}")

    lines.append("")

    needs = buckets.get("needs_attention", [])
    if needs:
        lines.append(f"## Needs attention ({len(needs)} tickers)")
        for entry in needs:
            ticker = entry["ticker"]
            change = entry.get("change_pct", 0) or 0
            vol_ratio = entry.get("vol_ratio", 1.0)
            dte = entry.get("days_to_earnings")

            sign = "+" if change >= 0 else ""
            parts = [f"{sign}{change:.1%}", f"vol {vol_ratio:.1f}×"]
            if dte is not None:
                parts.append(f"earnings in {dte}d")
            lines.append(f"{ticker}: {' | '.join(parts)}")

            for item in entry.get("news", [])[:2]:
                title = item.get("title", "")
                if title:
                    lines.append(f"  - {title}")
        lines.append("")
    else:
        lines += ["No tickers needing attention today.", ""]

    if open_positions:
        lines.append(f"## Open positions ({len(open_positions)})")
        for pos in open_positions:
            pnl = pos.get("pnl_pct")
            pnl_str = f" P&L: {pnl:+.1%}" if pnl is not None else ""
            lines.append(f"{pos['ticker']}{pnl_str}")
        lines.append("")

    if open_predictions:
        lines.append(f"## Open predictions ({len(open_predictions)})")
        for pred in open_predictions:
            ticker = pred["ticker"]
            direction = pred.get("direction", "")
            days_left = pred.get("days_left")
            overdue = pred.get("overdue", False)
            if overdue and days_left is not None:
                timing = f" (overdue {abs(days_left)}d)"
            elif days_left is not None:
                timing = f" ({days_left}d left)"
            else:
                timing = ""
            lines.append(f"{ticker}: {direction}{timing}")
        lines.append("")

    return "\n".join(lines)


def _format_needs_attention_entry(
    entry: dict[str, Any],
    vault_path: Path,
    briefing_date: date,
) -> list[str]:
    """Format a single Needs Attention ticker block."""
    ticker = entry["ticker"]
    news = entry.get("news", [])
    change = entry.get("change_pct", 0) or 0
    vol_ratio = entry.get("vol_ratio", 1.0) or 1.0
    dte = entry.get("days_to_earnings")

    sign = "+" if change >= 0 else ""
    header_parts = [f"{sign}{change:.1%}"]
    if vol_ratio >= 1.2:
        header_parts.append(f"Vol {vol_ratio:.1f}× avg")
    if dte is not None:
        header_parts.append(f"Earnings in {dte}d")

    lines = [f"**{ticker}** {' | '.join(header_parts)}"]

    for item in news[:2]:
        title = item.get("title", "")
        publisher = item.get("publisher", "")
        age = item.get("age", "")
        if title:
            if publisher and age:
                meta = f" *({publisher}, {age})*"
            elif publisher:
                meta = f" *({publisher})*"
            else:
                meta = ""
            lines.append(f"- {title}{meta}")

    research_file = vault_path / "Trading" / "Research" / f"{ticker} - {briefing_date}.md"
    if research_file.exists():
        lines.append(f"→ researched today — see [[{ticker} - {briefing_date}]]")
    else:
        lines.append(f"→ `wf research --ticker {ticker} --analyze`")

    lines.append("")
    return lines


def _format_watching_entry(entry: dict[str, Any]) -> str:
    """Format a single Watching tier line."""
    ticker = entry["ticker"]
    change = entry.get("change_pct", 0) or 0
    news = entry.get("news", [])

    sign = "+" if change >= 0 else ""
    line = f"**{ticker}** {sign}{change:.1%}"

    if news:
        headline = news[0].get("title", "")
        if len(headline) > 70:
            headline = headline[:67] + "..."
        if headline:
            line += f" — {headline}"
    else:
        line += " — no specific news"

    return line
