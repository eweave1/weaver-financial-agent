"""Pre-trade research — fetch market data and write structured research notes.

Writes to Trading/Research/TICKER - YYYY-MM-DD.md in the Obsidian vault.
The analysis sections (Setup, Bull case, Bear case, Key risks) are left blank
for manual fill-in until the LLM layer is built.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from weaver.predictions import _parse_frontmatter


def generate_research_note(
    vault_path: Path,
    ticker: str,
    research_date: Optional[date] = None,
    _snapshot: Optional[dict[str, Any]] = None,
    _news: Optional[list[dict[str, Any]]] = None,
) -> Path:
    """Fetch data for ticker and write a research note to Trading/Research/.

    _snapshot and _news are injection points for tests — pass them to skip
    network calls. In normal use leave them as None.

    Returns the path of the written file.
    """
    ticker = ticker.upper()
    note_date = research_date or date.today()

    research_dir = vault_path / "Trading" / "Research"
    research_dir.mkdir(parents=True, exist_ok=True)

    snapshot = _snapshot if _snapshot is not None else fetch_snapshot(ticker)
    news = _news if _news is not None else fetch_news(ticker)
    prior = fetch_prior_views(vault_path, ticker)

    content = _build_research_note(ticker, note_date, snapshot, news, prior)

    filepath = research_dir / f"{ticker} - {note_date}.md"
    filepath.write_text(content, encoding="utf-8")
    return filepath


def fetch_snapshot(ticker: str) -> dict[str, Any]:
    """Fetch price, volume, fundamentals, and next earnings date from yfinance.

    Returns a dict with all available fields. Missing fields are omitted rather
    than set to None so callers can use .get() with a default cleanly.
    Retries up to 3 times with exponential backoff on failure.
    """
    import yfinance as yf

    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}

            hist = t.history(period="5d")

            current_price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or (float(hist["Close"].iloc[-1]) if not hist.empty else None)
            )
            prev_close = (
                info.get("previousClose")
                or info.get("regularMarketPreviousClose")
            )

            result: dict[str, Any] = {}

            if info.get("shortName") or info.get("longName"):
                result["name"] = info.get("shortName") or info.get("longName")
            if info.get("sector"):
                result["sector"] = info["sector"]
            if info.get("industry"):
                result["industry"] = info["industry"]
            if info.get("longBusinessSummary"):
                result["description"] = info["longBusinessSummary"]

            if current_price:
                result["current_price"] = current_price
            if prev_close:
                result["prev_close"] = prev_close
            if current_price and prev_close:
                result["day_change_pct"] = (current_price - prev_close) / prev_close

            if info.get("fiftyTwoWeekHigh"):
                result["fifty_two_week_high"] = info["fiftyTwoWeekHigh"]
            if info.get("fiftyTwoWeekLow"):
                result["fifty_two_week_low"] = info["fiftyTwoWeekLow"]

            vol = info.get("volume") or info.get("regularMarketVolume")
            avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day")
            if vol:
                result["volume"] = vol
            if avg_vol:
                result["avg_volume"] = avg_vol
            if vol and avg_vol:
                result["volume_ratio"] = vol / avg_vol

            if info.get("marketCap"):
                result["market_cap"] = info["marketCap"]
            if info.get("trailingPE"):
                result["pe_trailing"] = info["trailingPE"]
            if info.get("forwardPE"):
                result["pe_forward"] = info["forwardPE"]
            if info.get("trailingEps"):
                result["eps"] = info["trailingEps"]

            # Next earnings date
            next_earnings = _get_next_earnings(t)
            if next_earnings:
                result["next_earnings"] = next_earnings

            return result

        except Exception as exc:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return {"error": str(exc)}

    return {}


def fetch_news(ticker: str, limit: int = 8) -> list[dict[str, Any]]:
    """Fetch recent news headlines from yfinance.

    Returns a list of dicts with keys: title, publisher, age, url.
    Returns an empty list on failure rather than raising.
    """
    import yfinance as yf

    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            raw = t.news or []
            articles = []
            for item in raw[:limit]:
                pub_time = item.get("providerPublishTime")
                age: Optional[str] = None
                if pub_time:
                    try:
                        pub_dt = datetime.fromtimestamp(pub_time, tz=timezone.utc)
                        age = _format_age(pub_dt)
                    except Exception:
                        pass
                articles.append({
                    "title": item.get("title", ""),
                    "publisher": item.get("publisher", ""),
                    "age": age,
                    "url": item.get("link", ""),
                })
            return articles
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)

    return []


def fetch_prior_views(vault_path: Path, ticker: str) -> dict[str, Any]:
    """Scan journal and predictions for prior activity on this ticker.

    Handles nonexistent or empty folders gracefully — returns empty lists.
    Safe to call on a brand-new vault with no entries.
    """
    ticker = ticker.upper()

    trades: list[dict] = []
    journal_dir = vault_path / "Trading" / "Journal"
    if journal_dir.exists():
        for f in sorted(journal_dir.glob("*.md")):
            fm = _parse_frontmatter(f.read_text(encoding="utf-8"))
            if str(fm.get("ticker", "")).upper() == ticker:
                entry: dict[str, Any] = {"file": f.stem}
                for key in ("entry_date", "action", "price", "reason",
                            "status", "time_horizon", "target_exit"):
                    if fm.get(key) is not None:
                        entry[key] = fm[key]
                trades.append(entry)

    predictions: list[dict] = []
    pred_dir = vault_path / "Trading" / "Predictions"
    if pred_dir.exists():
        for f in sorted(pred_dir.glob("*.md")):
            fm = _parse_frontmatter(f.read_text(encoding="utf-8"))
            if str(fm.get("ticker", "")).upper() == ticker:
                pred: dict[str, Any] = {"file": f.stem}
                for key in ("prediction_date", "direction", "confidence",
                            "reasoning", "resolve_by", "status", "outcome"):
                    if fm.get(key) is not None:
                        pred[key] = fm[key]
                predictions.append(pred)

    prior_research: list[str] = []
    research_dir = vault_path / "Trading" / "Research"
    if research_dir.exists():
        for f in sorted(research_dir.glob(f"{ticker} - *.md")):
            prior_research.append(f.stem)

    return {
        "trades": trades,
        "predictions": predictions,
        "prior_research": prior_research,
    }


# ── note builder ─────────────────────────────────────────────────────────────

def _build_research_note(
    ticker: str,
    note_date: date,
    snapshot: dict[str, Any],
    news: list[dict[str, Any]],
    prior: dict[str, Any],
) -> str:
    lines: list[str] = []

    # Frontmatter
    lines += ["---", f"ticker: {ticker}", f"date: {note_date}", "type: research"]
    if snapshot.get("current_price"):
        lines.append(f"price: {snapshot['current_price']:.2f}")
    lines += ["---", ""]

    # Title
    name = snapshot.get("name", "")
    lines.append(f"# {ticker} Research — {note_date}")
    if name and name != ticker:
        lines.append(f"*{name}*")
    lines.append("")

    # ── Snapshot ──────────────────────────────────────────────────────────────
    lines += ["## Snapshot", ""]

    if snapshot.get("error"):
        lines.append(f"> Could not fetch market data: {snapshot['error']}")
    else:
        price = snapshot.get("current_price")
        change = snapshot.get("day_change_pct")

        if price:
            price_str = f"**Price:** ${price:,.2f}"
            if change is not None:
                sign = "+" if change >= 0 else ""
                price_str += f" ({sign}{change:.1%} today)"
            lines.append(price_str + "  ")

        hi = snapshot.get("fifty_two_week_high")
        lo = snapshot.get("fifty_two_week_low")
        if hi and lo:
            lines.append(f"**52-week range:** ${lo:,.2f} – ${hi:,.2f}  ")

        vol = snapshot.get("volume")
        avg_vol = snapshot.get("avg_volume")
        vol_ratio = snapshot.get("volume_ratio")
        if vol:
            vol_str = f"**Volume:** {_fmt_num(vol)}"
            if avg_vol:
                vol_str += f" vs {_fmt_num(avg_vol)} avg"
            if vol_ratio:
                vol_str += f" ({vol_ratio:.1f}x)"
            lines.append(vol_str + "  ")

        if snapshot.get("market_cap"):
            lines.append(f"**Market cap:** {_fmt_money(snapshot['market_cap'])}  ")

        pe_t = snapshot.get("pe_trailing")
        pe_f = snapshot.get("pe_forward")
        if pe_t or pe_f:
            parts = []
            if pe_t:
                parts.append(f"trailing {pe_t:.1f}")
            if pe_f:
                parts.append(f"forward {pe_f:.1f}")
            lines.append(f"**P/E:** {' | '.join(parts)}  ")

        earnings = snapshot.get("next_earnings")
        if earnings:
            lines.append(f"**Next earnings:** {earnings}  ")

        sector = snapshot.get("sector")
        industry = snapshot.get("industry")
        if sector:
            label = f"{sector} — {industry}" if industry else sector
            lines.append(f"**Sector:** {label}  ")

    lines.append("")

    # ── Company overview ──────────────────────────────────────────────────────
    desc = snapshot.get("description", "")
    if desc:
        lines += ["## Company overview", ""]
        if len(desc) > 500:
            desc = desc[:500].rsplit(" ", 1)[0] + "..."
        lines += [desc, ""]

    # ── Recent news ───────────────────────────────────────────────────────────
    lines += ["## Recent news", ""]
    if news:
        for item in news:
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            age = item.get("age", "")
            url = item.get("url", "")

            pub_str = f" *({publisher})*" if publisher else ""
            age_str = f" — {age}" if age else ""
            if url:
                lines.append(f"- [{title}]({url}){pub_str}{age_str}")
            else:
                lines.append(f"- {title}{pub_str}{age_str}")
    else:
        lines.append("*No news available.*")
    lines.append("")

    # ── Prior views ───────────────────────────────────────────────────────────
    trades = prior.get("trades", [])
    preds = prior.get("predictions", [])
    prior_res = prior.get("prior_research", [])

    if trades or preds or prior_res:
        lines += ["## Prior views on this ticker", ""]

        if trades:
            lines += ["### Trades", ""]
            for t in trades:
                action = str(t.get("action", "")).capitalize()
                tdate = t.get("entry_date", "")
                price_val = t.get("price")
                price_s = f" @ ${price_val}" if price_val else ""
                reason = t.get("reason", "")
                status = t.get("status", "")
                status_s = f" [{status}]" if status else ""
                lines.append(f"- **{tdate}** {action}{price_s}{status_s} — {reason}")
            lines.append("")

        if preds:
            lines += ["### Predictions", ""]
            for p in preds:
                pdate = p.get("prediction_date", "")
                direction = str(p.get("direction", "")).upper()
                conf = p.get("confidence")
                conf_s = f" {int(float(conf) * 100)}%" if conf is not None else ""
                reasoning = p.get("reasoning", "")
                outcome = p.get("outcome")
                status = p.get("status", "")
                result = f" → **{outcome}**" if outcome else (f" [{status}]" if status else "")
                lines.append(f"- **{pdate}** {direction}{conf_s}{result} — {reasoning}")
            lines.append("")

        if prior_res:
            lines += ["### Prior research notes", ""]
            for r in prior_res:
                lines.append(f"- [[{r}]]")
            lines.append("")

    # ── Analysis sections (manual fill-in now, LLM later) ────────────────────
    for section in ["Setup", "Bull case", "Bear case", "Key risks"]:
        lines += [f"## {section}", "", ""]

    # ── My call ───────────────────────────────────────────────────────────────
    lines += [
        "## My call",
        "",
        "Complete your analysis above, then log a prediction:",
        "",
        f"```",
        f"wf log-prediction --ticker {ticker} --direction [up/down/flat] "
        f"--timeframe \"...\" --confidence 0.0 --reasoning \"...\" --resolve-by YYYY-MM-DD",
        f"```",
        "",
    ]

    return "\n".join(lines)


# ── private helpers ───────────────────────────────────────────────────────────

def _get_next_earnings(ticker_obj: Any) -> Optional[str]:
    """Return the next earnings date as a string, or None if unavailable."""
    try:
        calendar = ticker_obj.calendar
        if calendar is None:
            return None

        # yfinance returns a dict with an 'Earnings Date' key containing a list
        earnings_dates = None
        if isinstance(calendar, dict):
            earnings_dates = calendar.get("Earnings Date")
        elif hasattr(calendar, "columns") and "Earnings Date" in calendar.columns:
            # Older yfinance returned a DataFrame
            earnings_dates = calendar.loc["Earnings Date"].tolist()

        if not earnings_dates:
            return None

        today = date.today()
        for ed in earnings_dates:
            try:
                # May be a Timestamp, datetime, date, or string
                if hasattr(ed, "date"):
                    ed_date = ed.date() if callable(ed.date) else ed.date
                elif isinstance(ed, date):
                    ed_date = ed
                else:
                    from datetime import datetime as dt
                    ed_date = dt.strptime(str(ed)[:10], "%Y-%m-%d").date()

                if ed_date >= today:
                    return str(ed_date)
            except Exception:
                continue

        return None
    except Exception:
        return None


def _fmt_num(n: float) -> str:
    """Format a large number as 1.2M, 3.4B, 1.2T."""
    if n >= 1e12:
        return f"{n / 1e12:.1f}T"
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))


def _fmt_money(n: float) -> str:
    return f"${_fmt_num(n)}"


def _format_age(dt: datetime) -> str:
    """Return a human-readable age like '3h ago' or '2d ago'."""
    now = datetime.now(tz=timezone.utc)
    seconds = int((now - dt).total_seconds())
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"
