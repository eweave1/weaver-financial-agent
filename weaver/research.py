"""Pre-trade research — fetch market data and write structured research notes.

Writes to Trading/Research/TICKER - YYYY-MM-DD.md in the Obsidian vault.
Without --analyze: analysis sections are blank for manual fill-in.
With --analyze: calls DeepSeek via OpenRouter to fill in analysis sections
and adds an AI suggested call block before the My call section.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import requests

from weaver.predictions import _parse_frontmatter

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_ET = ZoneInfo("America/New_York")
_MARKET_OPEN_MINUTES = 9 * 60 + 30   # 570  (9:30 AM ET)
_MARKET_CLOSE_MINUTES = 16 * 60       # 960  (4:00 PM ET)
_TRADING_DAY_MINUTES = 390

_SYSTEM_PROMPT = (
    "You are a financial research assistant analyzing stocks for a human trader "
    "who makes all final decisions. Surface balanced analysis — do not give "
    "investment advice."
)


def generate_research_note(
    vault_path: Path,
    ticker: str,
    research_date: Optional[date] = None,
    _snapshot: Optional[dict[str, Any]] = None,
    _news: Optional[list[dict[str, Any]]] = None,
    analyze: bool = False,
    analyze_model: str = "deepseek/deepseek-v4-pro",
    openrouter_api_key: Optional[str] = None,
    openrouter_timeout: int = 90,
    _ai_analysis: Optional[dict[str, Any]] = None,
) -> Path:
    """Fetch data for ticker and write a research note to Trading/Research/.

    _snapshot, _news, and _ai_analysis are injection points for tests.
    When analyze=True, calls DeepSeek via OpenRouter unless _ai_analysis is
    provided directly.

    Returns the path of the written file.
    """
    ticker = ticker.upper()
    note_date = research_date or date.today()

    research_dir = vault_path / "Trading" / "Research"
    research_dir.mkdir(parents=True, exist_ok=True)

    snapshot = _snapshot if _snapshot is not None else fetch_snapshot(ticker)
    news = _news if _news is not None else fetch_news(
        ticker, company_name=snapshot.get("name", "")
    )
    prior = fetch_prior_views(vault_path, ticker)

    ai_analysis: Optional[dict[str, Any]] = None
    if analyze:
        if _ai_analysis is not None:
            ai_analysis = _ai_analysis
        else:
            ai_analysis = analyze_with_ai(
                ticker=ticker,
                snapshot=snapshot,
                news=news,
                prior=prior,
                model=analyze_model,
                api_key=openrouter_api_key or "",
                timeout=openrouter_timeout,
            )

    content = _build_research_note(ticker, note_date, snapshot, news, prior, ai_analysis)

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

            if vol:
                now_et = datetime.now(tz=_ET)
                projected = _project_intraday_volume(int(vol), now_et)
                if projected is not None:
                    result["volume_projected"] = projected
                    result["snapshot_time_et"] = (
                        f"{now_et.hour}:{now_et.minute:02d} ET"
                    )
                    if avg_vol:
                        result["volume_ratio_projected"] = projected / avg_vol

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


def fetch_news(
    ticker: str,
    limit: int = 8,
    company_name: str = "",
) -> list[dict[str, Any]]:
    """Fetch recent news headlines from yfinance, filtered to this ticker.

    Parses all raw items returned by yfinance, drops articles that don't
    mention the ticker symbol or company name in the headline, then returns
    the first `limit` that pass. This removes the general market/sector
    articles Yahoo Finance occasionally mixes into ticker feeds.

    Returns an empty list on failure rather than raising.
    """
    import yfinance as yf

    for attempt in range(3):
        try:
            t = yf.Ticker(ticker)
            raw = t.news or []
            parsed = [_parse_news_item(item) for item in raw if item]
            relevant = [
                item for item in parsed
                if _is_relevant_to_ticker(item["title"], ticker, company_name)
            ]
            return relevant[:limit]
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)

    return []


_GENERIC_NAME_WORDS = {
    "corp", "inc", "ltd", "llc", "plc", "group", "trust", "fund", "funds",
    "index", "etf", "about", "after", "their", "which", "would", "could",
}


def _is_relevant_to_ticker(
    title: str,
    ticker: str,
    company_name: str = "",
) -> bool:
    """Return True if the headline is relevant to the given ticker.

    Accepts articles that mention the ticker symbol (word-boundary match) or
    contain a distinctive word from the company name (4+ chars, not generic).
    Rejects empty titles and articles with no connection to the ticker.
    """
    if not title:
        return False

    title_lower = title.lower()

    if re.search(r"\b" + re.escape(ticker.lower()) + r"\b", title_lower):
        return True

    if company_name:
        words = [
            w.lower().rstrip(".,")
            for w in company_name.split()
            if len(w.rstrip(".,")) >= 4
            and w.lower().rstrip(".,") not in _GENERIC_NAME_WORDS
        ]
        if any(re.search(r"\b" + re.escape(w) + r"\b", title_lower) for w in words):
            return True

    return False


def _parse_news_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalise a single yfinance news item into {title, publisher, age, url}.

    Current yfinance nests all article fields under a 'content' key.
    Falls back to top-level fields so old-format items still parse correctly.
    """
    # Current API: everything lives under item["content"]
    content: dict[str, Any] = item.get("content") or item

    title = content.get("title", "")

    # Publisher: content["provider"]["displayName"]
    provider = content.get("provider") or {}
    publisher = provider.get("displayName", "") or item.get("publisher", "")

    # URL: prefer canonicalUrl, fall back to clickThroughUrl, then legacy "link"
    canonical = content.get("canonicalUrl") or {}
    click_through = content.get("clickThroughUrl") or {}
    url = (
        canonical.get("url")
        or click_through.get("url")
        or item.get("link", "")
    )

    # Timestamp: current API uses ISO 8601 pubDate; legacy used Unix providerPublishTime
    age: Optional[str] = None
    pub_date_str = content.get("pubDate")
    pub_time_unix = item.get("providerPublishTime")

    if pub_date_str:
        try:
            pub_dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            age = _format_age(pub_dt)
        except Exception:
            pass
    elif pub_time_unix:
        try:
            pub_dt = datetime.fromtimestamp(int(pub_time_unix), tz=timezone.utc)
            age = _format_age(pub_dt)
        except Exception:
            pass

    return {"title": title, "publisher": publisher, "age": age, "url": url}


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
                            "reasoning", "resolve_by", "status", "outcome", "trigger"):
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


# ── AI analysis ──────────────────────────────────────────────────────────────

def analyze_with_ai(
    ticker: str,
    snapshot: dict[str, Any],
    news: list[dict[str, Any]],
    prior: dict[str, Any],
    model: str,
    api_key: str,
    timeout: int = 90,
) -> dict[str, Any]:
    """Call OpenRouter/DeepSeek to generate analysis sections.

    Returns a dict with keys: setup, bull_case, bear_case, key_risks,
    direction, confidence, reasoning.
    On any failure returns {"error": "description"} — never raises.
    """
    if not api_key:
        return {"error": "OPENROUTER_API_KEY not set"}

    prompt = _build_analysis_prompt(ticker, snapshot, news, prior)

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
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
            },
            timeout=timeout,
        )
    except Exception as exc:
        return {"error": f"network error: {exc}"}

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text[:200])
        except Exception:
            detail = resp.text[:200]
        return {"error": f"API returned {resp.status_code}: {detail}"}

    try:
        raw_text = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        return {"error": f"unexpected response shape: {exc}"}

    return _parse_ai_response(raw_text)


def _build_analysis_prompt(
    ticker: str,
    snapshot: dict[str, Any],
    news: list[dict[str, Any]],
    prior: dict[str, Any],
) -> str:
    """Build the compact market-data context block sent to the AI."""
    lines: list[str] = [
        f"Analyze {ticker} using the data below.",
        "",
        "Return JSON with exactly these keys:",
        '  "setup"      – 2-3 sentences on current technical/fundamental setup',
        '  "bull_case"  – 2-3 sentences, genuine steelmanned bull case',
        '  "bear_case"  – 2-3 sentences, genuine steelmanned bear case',
        '  "key_risks"  – 2-3 sentences on the most important risks and catalysts; flag upcoming earnings prominently',
        '  "direction"  – exactly one of "up", "down", "flat"',
        '  "confidence" – float 0.0–1.0',
        '  "reasoning"  – 1-2 sentences summarizing the directional call',
        "",
        "Rules:",
        "• Steelman both bull and bear cases genuinely — do not strawman either.",
        '• "flat" is a fully valid and often correct call. If evidence is genuinely balanced, use "flat" and low confidence rather than manufacturing a lean.',
        "• Base analysis only on the data below. Do not invent price targets, earnings estimates, or facts not present.",
        "• If data is missing or thin, note the uncertainty rather than speculating.",
        "• Flag the single most important upcoming catalyst (e.g. earnings date) prominently in key_risks.",
        "• Frame as analysis for a human who makes all final decisions — not as advice.",
        "",
        "---",
        "",
        f"## {ticker} — Market Data",
    ]

    price = snapshot.get("current_price")
    prev_close = snapshot.get("prev_close")
    change = snapshot.get("day_change_pct")
    if price:
        price_str = f"Price: ${price:,.2f}"
        if change is not None:
            sign = "+" if change >= 0 else ""
            price_str += f" ({sign}{change:.1%} today)"
        if prev_close:
            price_str += f" | Prev close: ${prev_close:,.2f}"
        lines.append(price_str)

    hi = snapshot.get("fifty_two_week_high")
    lo = snapshot.get("fifty_two_week_low")
    if hi and lo:
        lines.append(f"52-week range: ${lo:,.2f} – ${hi:,.2f}")

    vol = snapshot.get("volume")
    avg_vol = snapshot.get("avg_volume")
    if vol:
        vol_str = f"Volume: {_fmt_num(vol)} current"
        projected = snapshot.get("volume_projected")
        if projected:
            vol_str += f" → ~{_fmt_num(projected)} projected"
        if avg_vol:
            vol_str += f" vs {_fmt_num(avg_vol)} avg"
        ratio = snapshot.get("volume_ratio_projected") or snapshot.get("volume_ratio")
        if ratio:
            vol_str += f" ({ratio:.1f}×)"
        capture = snapshot.get("snapshot_time_et")
        if capture:
            vol_str += f" [captured {capture}]"
        lines.append(vol_str)

    parts = []
    if snapshot.get("market_cap"):
        parts.append(f"Market cap: {_fmt_money(snapshot['market_cap'])}")
    pe_t = snapshot.get("pe_trailing")
    pe_f = snapshot.get("pe_forward")
    if pe_t or pe_f:
        pe_parts = []
        if pe_t:
            pe_parts.append(f"trailing {pe_t:.1f}")
        if pe_f:
            pe_parts.append(f"forward {pe_f:.1f}")
        parts.append(f"P/E: {' | '.join(pe_parts)}")
    if parts:
        lines.append(" | ".join(parts))

    if snapshot.get("next_earnings"):
        lines.append(f"Next earnings: {snapshot['next_earnings']}")

    sector = snapshot.get("sector")
    industry = snapshot.get("industry")
    if sector:
        lines.append(f"Sector: {sector}" + (f" — {industry}" if industry else ""))

    lines.append("")
    lines.append(f"## Recent News ({len(news)} items)")
    if news:
        for item in news[:8]:
            title = item.get("title", "")
            pub = item.get("publisher", "")
            age = item.get("age", "")
            meta = ""
            if pub and age:
                meta = f" ({pub}, {age})"
            elif pub:
                meta = f" ({pub})"
            elif age:
                meta = f" ({age})"
            lines.append(f"- {title}{meta}")
    else:
        lines.append("No recent news available.")

    lines.append("")
    lines.append(f"## Prior Activity on {ticker}")

    trades = prior.get("trades", [])
    if trades:
        lines.append(f"Trades ({len(trades)}):")
        for t in trades:
            action = str(t.get("action", "")).capitalize()
            tdate = t.get("entry_date", "")
            price_val = t.get("price")
            price_s = f" @ ${price_val}" if price_val else ""
            reason = t.get("reason", "")
            lines.append(f"  {tdate} {action}{price_s} — {reason}")
    else:
        lines.append("Trades: none")

    preds = prior.get("predictions", [])
    if preds:
        lines.append(f"Predictions ({len(preds)}):")
        for p in preds:
            pdate = p.get("prediction_date", "")
            direction = str(p.get("direction", "")).upper()
            conf = p.get("confidence")
            conf_s = f" {int(float(conf) * 100)}%" if conf is not None else ""
            status = p.get("status", "")
            status_s = f" [{status}]" if status else ""
            reasoning = p.get("reasoning", "")
            lines.append(f"  {pdate} {direction}{conf_s}{status_s} — {reasoning}")
    else:
        lines.append("Predictions: none")

    prior_res = prior.get("prior_research", [])
    if prior_res:
        lines.append(f"Prior research: {', '.join(prior_res)}")
    else:
        lines.append("Prior research: none")

    return "\n".join(lines)


def _parse_ai_response(raw_text: str) -> dict[str, Any]:
    """Parse the AI JSON response into an analysis dict.

    Tries json.loads first; falls back to extracting the first {...} block
    from markdown fences or prose. Returns {"error": "..."} on failure.
    """
    required = {"setup", "bull_case", "bear_case", "key_risks",
                "direction", "confidence", "reasoning"}

    def _try_parse(text: str) -> Optional[dict[str, Any]]:
        try:
            return json.loads(text)
        except Exception:
            return None

    result = _try_parse(raw_text.strip())

    if result is None:
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if fence:
            result = _try_parse(fence.group(1))

    if result is None:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end > start:
            result = _try_parse(raw_text[start : end + 1])

    if result is None:
        return {"error": "could not parse AI response as JSON"}

    if not isinstance(result, dict):
        return {"error": "AI response was not a JSON object"}

    missing = required - result.keys()
    if missing:
        return {"error": f"AI response missing keys: {', '.join(sorted(missing))}"}

    direction = str(result.get("direction", "")).lower().strip()
    if direction not in ("up", "down", "flat"):
        return {"error": f"AI returned invalid direction: {direction!r}"}

    try:
        confidence = float(result["confidence"])
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        return {"error": f"AI returned invalid confidence: {result['confidence']!r}"}

    return {
        "setup": str(result.get("setup", "")),
        "bull_case": str(result.get("bull_case", "")),
        "bear_case": str(result.get("bear_case", "")),
        "key_risks": str(result.get("key_risks", "")),
        "direction": direction,
        "confidence": confidence,
        "reasoning": str(result.get("reasoning", "")),
    }


# ── note builder ─────────────────────────────────────────────────────────────

def _build_research_note(
    ticker: str,
    note_date: date,
    snapshot: dict[str, Any],
    news: list[dict[str, Any]],
    prior: dict[str, Any],
    ai_analysis: Optional[dict[str, Any]] = None,
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
        if vol:
            vol_str = f"**Volume:** {_fmt_num(vol)}"
            projected = snapshot.get("volume_projected")
            if projected:
                vol_str += f" → ~{_fmt_num(projected)} projected"
            if avg_vol:
                vol_str += f" vs {_fmt_num(avg_vol)} avg"
            ratio = snapshot.get("volume_ratio_projected") or snapshot.get("volume_ratio")
            if ratio:
                vol_str += f" ({ratio:.1f}×)"
            capture = snapshot.get("snapshot_time_et")
            if capture:
                vol_str += f" — {capture}"
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

    # ── Analysis sections ─────────────────────────────────────────────────────
    if ai_analysis is None:
        for section in ["Setup", "Bull case", "Bear case", "Key risks"]:
            lines += [f"## {section}", "", ""]
    elif "error" in ai_analysis:
        error_msg = f"> AI analysis unavailable: {ai_analysis['error']}"
        for section in ["Setup", "Bull case", "Bear case", "Key risks"]:
            lines += [f"## {section}", "", error_msg, ""]
    else:
        for heading, key in [
            ("Setup", "setup"),
            ("Bull case", "bull_case"),
            ("Bear case", "bear_case"),
            ("Key risks", "key_risks"),
        ]:
            lines += [f"## {heading}", "", ai_analysis[key], ""]

        lines += [
            "",
            "---",
            "",
            "## AI's suggested call",
            "",
            f"Direction: {ai_analysis['direction']}  ",
            f"Confidence: {ai_analysis['confidence']:.2f}  ",
            f"Reasoning: {ai_analysis['reasoning']}",
            "",
        ]

    # ── My call ───────────────────────────────────────────────────────────────
    if ai_analysis is not None and "error" not in ai_analysis:
        lines += [
            "## My call",
            "",
            "**My take:** ",
            "",
            "→ Log when ready:",
            "",
            "```",
            f"wf log-prediction --ticker {ticker} "
            f"--direction {ai_analysis['direction']} "
            f"--confidence {ai_analysis['confidence']:.2f} "
            f'--resolve-by YYYY-MM-DD --reasoning "..."',
            "```",
            "",
        ]
    elif ai_analysis is not None:
        lines += [
            "## My call",
            "",
            "**My take:** ",
            "",
            "→ Log when ready:",
            "",
            "```",
            f"wf log-prediction --ticker {ticker} --direction [up/down/flat] "
            f'--confidence 0.0 --resolve-by YYYY-MM-DD --reasoning "..."',
            "```",
            "",
        ]
    else:
        lines += [
            "## My call",
            "",
            "Complete your analysis above, then log a prediction:",
            "",
            "```",
            f"wf log-prediction --ticker {ticker} --direction [up/down/flat] "
            f'--timeframe "..." --confidence 0.0 --reasoning "..." --resolve-by YYYY-MM-DD',
            "```",
            "",
        ]

    return "\n".join(lines)


# ── private helpers ───────────────────────────────────────────────────────────

def _project_intraday_volume(current_vol: int, now_et: datetime) -> Optional[int]:
    """Project a partial intraday volume to an estimated full-day total.

    Uses linear extrapolation based on how much of the 9:30–4:00 ET session
    has elapsed. Returns None when the market is closed or at the open tick
    (division by zero). Works well from ~10 AM onward; early morning estimates
    are noisier due to opening-auction volume spikes.
    """
    if now_et.weekday() >= 5:
        return None
    minutes_since_midnight = now_et.hour * 60 + now_et.minute
    elapsed = minutes_since_midnight - _MARKET_OPEN_MINUTES
    if elapsed <= 0 or minutes_since_midnight >= _MARKET_CLOSE_MINUTES:
        return None
    fraction = elapsed / _TRADING_DAY_MINUTES
    return int(current_vol / fraction)

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
