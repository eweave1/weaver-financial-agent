"""Trade proposal generation for the morning briefing.

Identifies setup candidates from the watchlist, calls DeepSeek for a directional
assessment, calculates position sizing per configured risk rules, and auto-logs
agent predictions to Trading/Predictions/AI/.

Long-only. A bearish read on a ticker → "no trade." A bearish read across the
whole watchlist → "no trades today, sector looks weak." Never manufactures a setup.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import requests

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_SYSTEM_PROMPT = """\
You are a cautious financial research assistant helping a human trader evaluate \
potential long trade setups for an AI/semiconductor-focused portfolio. \
You never execute trades. You base analysis only on data provided — never invent \
prices, news, or fundamentals.

LONG-ONLY RULE: This is a long-only portfolio. Never suggest a short position. \
If the honest read on a ticker is bearish, return direction "no_trade" and explain \
briefly. If the honest read across the whole watchlist is bearish, \
"no trades today, sector looks weak" is exactly the right output. \
Do not manufacture a bullish case to have something to propose. \
A bearish read is a reason to stay out or to note that an existing position's \
thesis is weakening — not a reason to propose a short.

ANTI-BIAS RULES (follow verbatim):
- Finance-tuned language models exhibit a documented strong bullish bias, predicting \
bullish up to 80% of the time and performing near-zero on flat or consolidating periods. \
Actively counteract this.
- "No trade today" is the correct answer most days. Do not manufacture a setup.
- Do not rely on momentum indicators (RSI, MACD, Bollinger) as primary signals. \
They reliably stay bullish right through sector reversals.
- Do not propose a trade whose thesis is "the stock is down, therefore it's cheap." \
Falling knives are the most common failure mode.
- This is a proposal for a human to approve, not a decision. Frame accordingly.

Respond with a single JSON object. No markdown fences, no explanation outside the JSON.

Required schema:
{
  "direction": "long" or "no_trade",
  "reasoning": "<2-3 sentences explaining your assessment>",
  "confidence": <0.0 to 1.0>,
  "entry_price": <number, or the string "at market", or null if no_trade>,
  "stop_loss": <number, or null if no_trade>,
  "stop_condition": "<what event/price invalidates the thesis, or null if no_trade>",
  "timeframe": "day" or "swing" or "hold" or null if no_trade,
  "exit_expectation": "<rough exit target and rationale, or null if no_trade>"
}
"""


# ── Sizing ────────────────────────────────────────────────────────────────────


def get_basket_exposure(open_positions: list[dict[str, Any]]) -> float:
    """Total committed capital across open positions: entry_price × quantity."""
    total = 0.0
    for pos in open_positions:
        price = pos.get("entry_price")
        qty = pos.get("quantity")
        if price is not None and qty is not None:
            try:
                total += float(price) * float(qty)
            except (TypeError, ValueError):
                pass
    return total


def calculate_position_size(
    entry: float,
    stop: float,
    config: dict[str, Any],
    basket_exposure: float,
) -> Optional[dict[str, Any]]:
    """Apply 1% risk rule with 5% single-position and basket exposure caps.

    Returns a dict with shares, position_value, risk_amount, cap_applied, and
    a human-readable math string showing the work. Returns None if stop >= entry
    (invalid setup — shouldn't occur for long-only proposals).
    """
    if stop >= entry:
        return None

    cfg = config.get("proposals", {})
    portfolio_value = float(cfg.get("portfolio_value", 1000))
    max_risk_pct = float(cfg.get("max_risk_per_trade_pct", 1.0))
    max_pos_pct = float(cfg.get("max_single_position_pct", 5.0))
    max_basket_pct = float(cfg.get("max_basket_exposure_pct", 10.0))

    risk_budget = portfolio_value * (max_risk_pct / 100)
    max_position = portfolio_value * (max_pos_pct / 100)
    basket_cap = portfolio_value * (max_basket_pct / 100)
    basket_remaining = basket_cap - basket_exposure

    if basket_remaining <= 0:
        return {
            "shares": 0.0,
            "position_value": 0.0,
            "risk_amount": 0.0,
            "cap_applied": "basket at capacity",
            "basket_exposure": basket_exposure,
            "basket_cap": basket_cap,
            "math": (
                f"Basket at capacity: ${basket_exposure:.2f} committed of "
                f"${basket_cap:.2f} cap. No new position should be opened."
            ),
        }

    risk_per_share = entry - stop
    shares_uncapped = risk_budget / risk_per_share
    value_uncapped = shares_uncapped * entry

    cap_applied: Optional[str] = None
    position_value = value_uncapped

    if position_value > max_position:
        position_value = max_position
        cap_applied = f"5% cap (${max_position:.2f} max)"

    if position_value > basket_remaining:
        position_value = basket_remaining
        if cap_applied:
            cap_applied += f" + basket remaining ${basket_remaining:.2f}"
        else:
            cap_applied = f"basket cap (${basket_remaining:.2f} remaining of ${basket_cap:.2f})"

    shares = round(position_value / entry, 2)
    position_value_final = round(shares * entry, 2)
    actual_risk = round(shares * risk_per_share, 2)

    math_parts = [
        f"1% risk: ${risk_budget:.2f} ÷ (${entry:.2f} − ${stop:.2f}) "
        f"= {shares_uncapped:.2f} shares",
    ]
    if cap_applied:
        math_parts.append(f"→ {cap_applied}")
    math_parts.append(
        f"→ {shares} shares × ${entry:.2f} = ${position_value_final:.2f} "
        f"(risk ${actual_risk:.2f})"
    )

    return {
        "shares": shares,
        "position_value": position_value_final,
        "risk_amount": actual_risk,
        "cap_applied": cap_applied,
        "basket_exposure": basket_exposure,
        "basket_cap": basket_cap,
        "math": " ".join(math_parts),
    }


# ── Candidate identification ──────────────────────────────────────────────────


def identify_candidates(
    buckets: dict[str, list],
    open_predictions: list[dict[str, Any]],
    current_prices: dict[str, float],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return tickers that qualify for proposal evaluation.

    Priority 1: open predictions within trigger_proximity_pct of their trigger.
    Priority 2: needs_attention tickers.
    Capped at max_proposals_per_day across both sources.
    """
    cfg = config.get("proposals", {})
    max_proposals = int(cfg.get("max_proposals_per_day", 2))
    proximity_pct = float(cfg.get("trigger_proximity_pct", 2.0)) / 100

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    for pred in open_predictions:
        if len(candidates) >= max_proposals:
            break
        ticker = pred.get("ticker", "")
        trigger = pred.get("trigger_price")
        current = pred.get("current_price") or current_prices.get(ticker)
        if not trigger or not current or ticker in seen:
            continue
        dist = abs(current - trigger) / trigger
        if dist <= proximity_pct:
            sign = "+" if current >= trigger else ""
            candidates.append({
                "ticker": ticker,
                "reason": (
                    f"open prediction trigger within {dist:.1%} "
                    f"(trigger ${trigger:.2f}, current ${current:.2f} "
                    f"= {sign}{(current - trigger) / trigger:.1%})"
                ),
                "source": "prediction",
            })
            seen.add(ticker)

    for entry in buckets.get("needs_attention", []):
        if len(candidates) >= max_proposals:
            break
        ticker = entry.get("ticker", "")
        if ticker in seen:
            continue
        change = entry.get("change_pct", 0) or 0
        vol_ratio = entry.get("vol_ratio", 1.0) or 1.0
        dte = entry.get("days_to_earnings")

        parts: list[str] = [f"{change:+.1%}"]
        if vol_ratio >= 1.5:
            parts.append(f"vol {vol_ratio:.1f}×")
        if dte is not None and dte <= 7:
            parts.append(f"earnings in {dte}d")

        candidates.append({
            "ticker": ticker,
            "reason": "needs attention: " + " | ".join(parts),
            "source": "needs_attention",
        })
        seen.add(ticker)

    return candidates


# ── LLM call ──────────────────────────────────────────────────────────────────


def _build_proposal_prompt(
    candidate: dict[str, Any],
    ticker_data: dict[str, Any],
    open_positions: list[dict[str, Any]],
    open_predictions: list[dict[str, Any]],
    config: dict[str, Any],
    basket_exposure: float,
) -> str:
    ticker = candidate["ticker"]
    data = ticker_data.get(ticker, {})
    snap = data.get("snapshot", {})
    news = data.get("news", [])
    cfg = config.get("proposals", {})
    portfolio_value = float(cfg.get("portfolio_value", 1000))
    basket_cap = portfolio_value * float(cfg.get("max_basket_exposure_pct", 10.0)) / 100

    lines: list[str] = [
        f"Evaluate whether {ticker} is a good long entry today.",
        f"Flagged because: {candidate['reason']}",
        "",
        f"## {ticker} data",
    ]

    price = snap.get("current_price")
    change = snap.get("day_change_pct")
    vol_ratio = snap.get("volume_ratio_projected") or snap.get("volume_ratio")
    ne = snap.get("next_earnings")
    dte: Optional[int] = None
    if ne and str(ne) != "N/A":
        try:
            dte = (date.fromisoformat(str(ne)[:10]) - date.today()).days
            if dte < 0:
                dte = None
        except (ValueError, TypeError):
            pass

    if price is not None:
        lines.append(f"Current price: ${price:,.2f}")
    if change is not None:
        lines.append(f"Day change: {change:+.1%}")
    if vol_ratio is not None:
        lines.append(f"Volume: {float(vol_ratio):.1f}× average")
    if dte is not None:
        lines.append(f"Earnings in {dte} days")

    if news:
        lines += ["", f"## Recent {ticker} news"]
        for item in news[:3]:
            title = item.get("title", "")
            if title:
                lines.append(f"- {title}")

    if open_positions:
        lines += ["", "## Your open positions"]
        for pos in open_positions:
            ep = pos.get("entry_price")
            qty = pos.get("quantity")
            pnl = pos.get("pnl_pct")
            committed = ""
            if ep is not None and qty is not None:
                try:
                    committed = f" (${float(ep) * float(qty):,.2f} committed)"
                except (TypeError, ValueError):
                    pass
            pnl_str = f" | P&L {pnl:+.1%}" if pnl is not None else ""
            lines.append(f"{pos['ticker']}: entry ${ep}{committed}{pnl_str}")
        lines.append(
            f"Total basket: ${basket_exposure:.2f} committed of ${basket_cap:.2f} cap"
        )

    if open_predictions:
        lines += ["", "## Your open predictions"]
        for pred in open_predictions:
            dl = pred.get("days_left")
            timing = f" ({dl}d left)" if dl is not None else ""
            lines.append(f"{pred['ticker']}: {pred.get('direction', '')}{timing}")

    return "\n".join(lines)


def _call_llm(
    prompt: str,
    api_key: str,
    model: str,
    timeout: int,
) -> Optional[dict[str, Any]]:
    """POST to OpenRouter and return parsed JSON dict, or None on any failure."""
    if not api_key:
        return None
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
                "temperature": 0.2,
            },
            timeout=timeout,
        )
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    try:
        content = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

    # Strip markdown code fence if present
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\s*```\s*$", "", content, flags=re.MULTILINE)
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


# ── Concentration warning ─────────────────────────────────────────────────────


def _build_concentration_warning(
    ticker: str,
    open_positions: list[dict[str, Any]],
    open_predictions: list[dict[str, Any]],
    basket_exposure: float,
    basket_cap: float,
    sizing: Optional[dict[str, Any]],
) -> str:
    parts: list[str] = []

    held = {pos["ticker"] for pos in open_positions}
    predicted = {pred["ticker"] for pred in open_predictions}

    if ticker in held:
        parts.append(f"You already hold {ticker} in the journal.")
    if ticker in predicted:
        parts.append(f"You have an open prediction on {ticker}.")

    n_pos = len(open_positions)
    n_pred = len(open_predictions)
    if n_pos > 0 or n_pred > 0:
        exposure_parts = []
        if n_pos:
            exposure_parts.append(f"{n_pos} position{'s' if n_pos > 1 else ''}")
        if n_pred:
            exposure_parts.append(f"{n_pred} prediction{'s' if n_pred > 1 else ''}")
        parts.append(
            f"All 19 tickers are highly correlated (AI-capex thesis). "
            f"Current: {' + '.join(exposure_parts)}, ${basket_exposure:.2f} committed."
        )
    else:
        parts.append("No open positions or predictions — clean slate.")

    if sizing is not None:
        if sizing.get("cap_applied") == "basket at capacity":
            parts.append(
                f"BASKET AT CAPACITY (${basket_exposure:.2f} of ${basket_cap:.2f}). "
                "Do not open a new position."
            )
        else:
            new_total = basket_exposure + sizing.get("position_value", 0.0)
            pct = new_total / basket_cap if basket_cap > 0 else 0
            parts.append(
                f"This trade: ${sizing.get('position_value', 0):.2f} → "
                f"basket total ${new_total:.2f} of ${basket_cap:.2f} ({pct:.0%} of cap)."
            )

    return " ".join(parts)


# ── Formatting ────────────────────────────────────────────────────────────────


def format_proposal_text(proposal: dict[str, Any]) -> str:
    """Render a proposal as phone-readable plain text for the briefing and Telegram."""
    ticker = proposal["ticker"]
    direction = proposal.get("direction", "no_trade")
    reasoning = proposal.get("reasoning", "No assessment available.")
    confidence = proposal.get("confidence")
    candidate_reason = proposal.get("candidate_reason", "")
    conf_str = f"{int(confidence * 100)}%" if confidence is not None else "—"

    if direction != "long":
        return f"**{ticker} — NO TRADE** ({conf_str})\nFlagged: {candidate_reason}\n{reasoning}"

    entry = proposal.get("entry_price")
    stop = proposal.get("stop_loss")
    stop_cond = proposal.get("stop_condition") or "—"
    timeframe = proposal.get("timeframe") or "—"
    exit_exp = proposal.get("exit_expectation") or "—"
    sizing = proposal.get("sizing") or {}
    concentration = proposal.get("concentration") or "—"

    entry_str = (
        f"${float(entry):,.2f}" if isinstance(entry, (int, float))
        else str(entry or "at market")
    )
    stop_str = f"${float(stop):,.2f}" if stop is not None else "—"

    lines = [
        f"**{ticker} — LONG** ({conf_str} confidence)",
        f"Flagged: {candidate_reason}",
        "",
        f"Entry: {entry_str}  |  Stop: {stop_str}  |  {timeframe}",
        f"Stop condition: {stop_cond}",
        f"Exit: {exit_exp}",
        "",
        reasoning,
        "",
        f"Sizing: {sizing.get('math', '—')}",
        "",
        f"Concentration: {concentration}",
        "",
        f"→ To take this: place in Webull, then run",
        f"  wf log-trade --ticker {ticker}",
        "→ To skip: no action needed",
    ]
    return "\n".join(lines)


# ── Agent prediction log ──────────────────────────────────────────────────────


def log_agent_prediction(
    vault_path: Path,
    ticker: str,
    proposal: dict[str, Any],
) -> Optional[Path]:
    """Write a long proposal to Trading/Predictions/AI/ for track-record comparison.

    Only called for direction == 'long'. Returns written path or None.
    Never raises — caller wraps in try/except.
    """
    if proposal.get("direction") != "long":
        return None

    from weaver.predictions import _esc, _unique_filename

    pred_dir = vault_path / "Trading" / "Predictions" / "AI"
    pred_dir.mkdir(parents=True, exist_ok=True)

    today = date.today()
    timeframe = proposal.get("timeframe") or "swing"
    confidence = float(proposal.get("confidence") or 0.5)
    reasoning = proposal.get("reasoning") or ""
    exit_exp = proposal.get("exit_expectation") or ""
    entry = proposal.get("entry_price")
    stop = proposal.get("stop_loss")

    days_map = {"day": 1, "swing": 14, "hold": 30}
    resolve_by = today + timedelta(days=days_map.get(timeframe, 14))

    full_reasoning = reasoning
    if exit_exp:
        full_reasoning += f" Exit expectation: {exit_exp}"
    if proposal.get("candidate_reason"):
        full_reasoning += f" [Flagged: {proposal['candidate_reason']}]"

    trigger: Optional[float] = None
    if isinstance(entry, (int, float)):
        trigger = float(entry)

    confidence_pct = int(confidence * 100)
    fm_lines = [
        "---",
        f"ticker: {ticker}",
        "direction: up",
        f'timeframe: "{timeframe}"',
        f"confidence: {confidence}",
        f"resolve_by: {resolve_by}",
        f"prediction_date: {today}",
        f'reasoning: "{_esc(full_reasoning)}"',
        "status: open",
        "source: agent",
    ]
    if trigger is not None:
        fm_lines.append(f"trigger: {trigger}")
    if stop is not None:
        fm_lines.append(f"stop_loss: {stop}")
    fm_lines.append("---")

    body_lines = [
        f"# {today} {ticker} — AGENT PROPOSAL — up ({confidence_pct}%)",
        "",
        f"**Timeframe:** {timeframe}  ",
        f"**Resolve by:** {resolve_by}  ",
        f"**Confidence:** {confidence_pct}%  ",
        "**Source:** agent  ",
    ]
    if trigger is not None:
        body_lines.append(f"**Entry:** ${trigger:,.2f}  ")
    if stop is not None:
        body_lines.append(f"**Stop:** ${float(stop):,.2f}  ")
    body_lines += ["", "## Reasoning", "", full_reasoning]

    content = "\n".join(fm_lines) + "\n" + "\n".join(body_lines) + "\n"
    stem = f"{today} {ticker} (agent)"
    filepath = pred_dir / _unique_filename(pred_dir, stem)
    filepath.write_text(content, encoding="utf-8")
    return filepath


# ── Main entry point ──────────────────────────────────────────────────────────


def generate_proposals(
    ticker_data: dict[str, Any],
    buckets: dict[str, list],
    open_positions: list[dict[str, Any]],
    open_predictions: list[dict[str, Any]],
    current_prices: dict[str, float],
    vault_path: Path,
    config: dict[str, Any],
    api_key: str = "",
    model: str = "deepseek/deepseek-v4-pro",
    timeout: int = 90,
    _proposals: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Identify candidates, call the LLM, size positions, return proposal dicts.

    _proposals is an injection point for tests — if not None, returned as-is with
    text formatted, skipping all LLM calls and candidate identification.
    """
    if _proposals is not None:
        for p in _proposals:
            if "text" not in p:
                p["text"] = format_proposal_text(p)
        return _proposals

    candidates = identify_candidates(buckets, open_predictions, current_prices, config)
    if not candidates:
        return []

    basket_exposure = get_basket_exposure(open_positions)
    cfg = config.get("proposals", {})
    portfolio_value = float(cfg.get("portfolio_value", 1000))
    basket_cap = portfolio_value * float(cfg.get("max_basket_exposure_pct", 10.0)) / 100

    proposals: list[dict[str, Any]] = []

    for candidate in candidates:
        ticker = candidate["ticker"]
        prompt = _build_proposal_prompt(
            candidate=candidate,
            ticker_data=ticker_data,
            open_positions=open_positions,
            open_predictions=open_predictions,
            config=config,
            basket_exposure=basket_exposure,
        )

        llm_result = _call_llm(prompt, api_key, model, timeout)

        if llm_result is None:
            proposal: dict[str, Any] = {
                "ticker": ticker,
                "direction": "no_trade",
                "reasoning": "Proposal unavailable: LLM call failed.",
                "confidence": None,
                "candidate_reason": candidate["reason"],
                "sizing": None,
                "concentration": None,
            }
            proposal["text"] = format_proposal_text(proposal)
            proposals.append(proposal)
            continue

        direction = str(llm_result.get("direction", "no_trade")).lower()
        if direction not in ("long", "no_trade"):
            direction = "no_trade"

        entry_raw = llm_result.get("entry_price")
        stop_raw = llm_result.get("stop_loss")

        entry: Any
        entry_float: Optional[float]
        if isinstance(entry_raw, str) and entry_raw.lower() == "at market":
            entry = "at market"
            entry_float = current_prices.get(ticker)
        elif isinstance(entry_raw, (int, float)):
            entry = float(entry_raw)
            entry_float = float(entry_raw)
        else:
            entry = "at market"
            entry_float = current_prices.get(ticker)

        stop: Optional[float] = float(stop_raw) if isinstance(stop_raw, (int, float)) else None

        sizing: Optional[dict[str, Any]] = None
        if direction == "long" and entry_float is not None and stop is not None:
            sizing = calculate_position_size(
                entry=entry_float,
                stop=stop,
                config=config,
                basket_exposure=basket_exposure,
            )

        concentration = _build_concentration_warning(
            ticker=ticker,
            open_positions=open_positions,
            open_predictions=open_predictions,
            basket_exposure=basket_exposure,
            basket_cap=basket_cap,
            sizing=sizing,
        )

        proposal = {
            "ticker": ticker,
            "direction": direction,
            "candidate_reason": candidate["reason"],
            "entry_price": entry,
            "stop_loss": stop,
            "stop_condition": llm_result.get("stop_condition"),
            "timeframe": llm_result.get("timeframe"),
            "exit_expectation": llm_result.get("exit_expectation"),
            "reasoning": llm_result.get("reasoning", ""),
            "confidence": llm_result.get("confidence"),
            "sizing": sizing,
            "concentration": concentration,
        }
        proposal["text"] = format_proposal_text(proposal)

        if direction == "long":
            try:
                log_agent_prediction(vault_path, ticker, proposal)
            except Exception:
                pass

        proposals.append(proposal)

    return proposals
