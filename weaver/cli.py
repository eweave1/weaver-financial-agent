"""CLI entry points for weaver-financial-agent.

All fields on log-trade and log-prediction are optional flags. If any are
omitted the command prompts for them interactively in logical order:

  wf log-trade                        # fully interactive wizard
  wf log-trade --ticker NVDA          # ticker provided, rest prompted
  wf log-trade --ticker NVDA --action buy --quantity 10 ...  # fully scripted

Free-text fields (--reason, --stop, --reasoning) are always prompted as plain
text input — no shell quoting needed, dollar signs and arrows type freely.
Structured fields (action, horizon, direction) prompt with choices shown.
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import click

from weaver.config import get_analyze_model, get_openrouter_timeout, get_vault_path, load_config
from weaver.journal import VALID_ACTIONS, VALID_HORIZONS, log_trade
from weaver.predictions import (
    VALID_DIRECTIONS,
    VALID_OUTCOMES,
    list_pending_resolutions,
    log_prediction,
    resolve_prediction,
)


# ── group ─────────────────────────────────────────────────────────────────────

@click.group()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to config.yaml. Defaults to project root config.yaml.",
)
@click.pass_context
def main(ctx: click.Context, config_path: Optional[Path]) -> None:
    """Weaver financial research agent — notify and review, never execute."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path


def _vault(ctx: click.Context) -> Path:
    try:
        config = load_config(ctx.obj["config_path"])
        return get_vault_path(config)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


# ── log-trade ─────────────────────────────────────────────────────────────────

@main.command("log-trade")
@click.option("--ticker", default=None, help="Stock ticker (e.g. NVDA).")
@click.option(
    "--action",
    default=None,
    type=click.Choice(list(VALID_ACTIONS)),
    help="buy or sell.",
)
@click.option("--quantity", default=None, type=float, help="Number of shares.")
@click.option("--price", default=None, type=float, help="Execution price per share.")
@click.option("--reason", default=None, help="One-sentence reason for this trade.")
@click.option(
    "--stop",
    "stop_condition",
    default=None,
    help="Stop/exit condition (e.g. 'Close below $145').",
)
@click.option(
    "--horizon",
    "time_horizon",
    default=None,
    type=click.Choice(list(VALID_HORIZONS)),
    help="Time horizon: day, swing, or hold.",
)
@click.option("--target", "target_exit", type=float, default=None,
              help="Target exit price (optional).")
@click.option(
    "--date", "trade_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Trade date as YYYY-MM-DD (defaults to today).",
)
@click.option("--linked-buy", default=None,
              help="For sell entries: stem of the linked buy file "
                   "(e.g. '2026-06-30 NVDA buy').")
@click.pass_context
def log_trade_cmd(
    ctx: click.Context,
    ticker: Optional[str],
    action: Optional[str],
    quantity: Optional[float],
    price: Optional[float],
    reason: Optional[str],
    stop_condition: Optional[str],
    time_horizon: Optional[str],
    target_exit: Optional[float],
    trade_date: Optional[datetime],
    linked_buy: Optional[str],
) -> None:
    """Log a buy or sell trade to the journal.

    All fields can be passed as flags or entered interactively when omitted.
    Free-text fields (--reason, --stop) are always safe to type at a prompt —
    no shell quoting or escaping needed.
    """
    # ── structured fields: prompt with type validation if omitted ─────────────
    if ticker is None:
        ticker = click.prompt("Ticker")
    ticker = ticker.upper()

    if action is None:
        action = click.prompt("Action", type=click.Choice(list(VALID_ACTIONS)))

    if quantity is None:
        quantity = click.prompt("Quantity", type=float)

    if price is None:
        price = click.prompt("Price", type=float)

    if time_horizon is None:
        time_horizon = click.prompt(
            "Time horizon", type=click.Choice(list(VALID_HORIZONS))
        )

    # target is optional — only prompt in wizard mode (when reason+stop are also missing)
    if target_exit is None and reason is None and stop_condition is None:
        raw = click.prompt("Target exit price (Enter to skip)", default="",
                           show_default=False)
        if raw.strip():
            try:
                target_exit = float(raw.strip())
            except ValueError:
                click.echo("  Not a valid number — leaving target blank.", err=True)

    # ── free-text fields: plain prompt, type anything freely ──────────────────
    if reason is None:
        reason = click.prompt("Reason")

    if stop_condition is None:
        stop_condition = click.prompt("Stop condition")

    # ── write ─────────────────────────────────────────────────────────────────
    vault_path = _vault(ctx)
    td = trade_date.date() if trade_date else None

    try:
        filepath = log_trade(
            vault_path=vault_path,
            ticker=ticker,
            action=action,
            quantity=quantity,
            price=price,
            reason=reason,
            stop_condition=stop_condition,
            time_horizon=time_horizon,
            target_exit=target_exit,
            trade_date=td,
            linked_buy_file=linked_buy,
        )
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Unexpected error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Trade logged: {filepath}")


# ── log-prediction ────────────────────────────────────────────────────────────

@main.command("log-prediction")
@click.option("--ticker", default=None, help="Stock ticker (e.g. NVDA).")
@click.option(
    "--direction",
    default=None,
    type=click.Choice(list(VALID_DIRECTIONS)),
    help="Predicted direction: up, down, or flat.",
)
@click.option("--timeframe", default=None,
              help="Prediction timeframe (e.g. '2 weeks', '1 month').")
@click.option("--confidence", default=None, type=float,
              help="Confidence level from 0.0 to 1.0.")
@click.option("--reasoning", default=None,
              help="Your reasoning for this prediction.")
@click.option("--resolve-by", "resolve_by_str", default=None,
              help="Date to resolve the prediction by (YYYY-MM-DD).")
@click.option("--trigger", default=None, type=float,
              help="Explicit entry trigger price (e.g. 145.00). Optional — "
                   "shown in the briefing alongside current price.")
@click.option(
    "--date", "prediction_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Prediction date as YYYY-MM-DD (defaults to today).",
)
@click.pass_context
def log_prediction_cmd(
    ctx: click.Context,
    ticker: Optional[str],
    direction: Optional[str],
    timeframe: Optional[str],
    confidence: Optional[float],
    reasoning: Optional[str],
    resolve_by_str: Optional[str],
    trigger: Optional[float],
    prediction_date: Optional[datetime],
) -> None:
    """Log a directional prediction with a resolve-by date.

    All fields can be passed as flags or entered interactively when omitted.
    --reasoning is always safe to type freely at a prompt — dollar signs,
    arrows, and apostrophes all work without shell quoting.
    --trigger sets an explicit entry price level shown in the morning briefing.
    """
    # ── structured fields ─────────────────────────────────────────────────────
    if ticker is None:
        ticker = click.prompt("Ticker")
    ticker = ticker.upper()

    if direction is None:
        direction = click.prompt(
            "Direction", type=click.Choice(list(VALID_DIRECTIONS))
        )

    if timeframe is None:
        timeframe = click.prompt("Timeframe (e.g. '2 weeks')")

    # confidence: validate range whether from flag or prompt
    if confidence is None:
        confidence = click.prompt(
            "Confidence (0.0–1.0)", type=click.FloatRange(0.0, 1.0)
        )
    else:
        if not 0.0 <= confidence <= 1.0:
            click.echo(
                f"Error: --confidence must be between 0.0 and 1.0, got {confidence}",
                err=True,
            )
            sys.exit(1)

    # trigger: optional, flag-only (not prompted)
    if trigger is not None and trigger <= 0:
        click.echo(
            f"Error: --trigger must be a positive price, got {trigger}",
            err=True,
        )
        sys.exit(1)

    # resolve-by: validate future date whether from flag or prompt
    if resolve_by_str is None:
        resolve_by_date = _prompt_future_date("Resolve by (YYYY-MM-DD)")
    else:
        resolve_by_date = _parse_future_date_flag(resolve_by_str)

    # ── free-text field ───────────────────────────────────────────────────────
    if reasoning is None:
        reasoning = click.prompt("Reasoning")

    # ── write ─────────────────────────────────────────────────────────────────
    vault_path = _vault(ctx)
    pd = prediction_date.date() if prediction_date else None

    try:
        filepath = log_prediction(
            vault_path=vault_path,
            ticker=ticker,
            direction=direction,
            timeframe=timeframe,
            confidence=confidence,
            reasoning=reasoning,
            resolve_by=resolve_by_date,
            prediction_date=pd,
            trigger=trigger,
        )
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Unexpected error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Prediction logged: {filepath}")


# ── resolve-predictions ───────────────────────────────────────────────────────

@main.command("resolve-predictions")
@click.option(
    "--date", "as_of_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Treat predictions overdue as of this date (defaults to today).",
)
@click.pass_context
def resolve_predictions_cmd(
    ctx: click.Context,
    as_of_date: Optional[datetime],
) -> None:
    """Interactively resolve all open predictions past their resolve-by date.

    For each prediction, the agent fetches the actual price movement from
    yfinance and shows it to you. You then categorize the outcome:

    \b
    right       — reasoning held, call was correct
    wrong       — reasoning was tested and failed
    unforeseen  — market moved for a reason outside the original thesis
                  (tail event, macro shock, etc.)

    The 'unforeseen' category is scored separately so it doesn't penalize
    sound analysis that was simply ambushed by an unpredictable event.
    """
    vault_path = _vault(ctx)
    check_date = as_of_date.date() if as_of_date else date.today()

    pending = list_pending_resolutions(vault_path, as_of_date=check_date)
    if not pending:
        click.echo("No predictions pending resolution.")
        return

    click.echo(f"\nFound {len(pending)} prediction(s) to resolve.\n")

    for pred in pending:
        click.echo("─" * 60)
        click.echo(f"  Ticker:      {pred['ticker']}")
        click.echo(f"  Direction:   {pred['direction']}")
        click.echo(f"  Timeframe:   {pred['timeframe']}")
        confidence = pred.get("confidence")
        if confidence is not None:
            click.echo(f"  Confidence:  {int(float(confidence) * 100)}%")
        click.echo(f"  Resolve by:  {pred['resolve_by']}")
        click.echo(f"  Reasoning:   {pred['reasoning']}")
        click.echo("")

        actual = _fetch_actual_direction(
            ticker=pred["ticker"],
            prediction_date=pred.get("prediction_date"),
            resolve_by=pred["resolve_by"],
        )
        if actual:
            click.echo(f"  Actual move: {actual}")
            actual_direction = actual
        else:
            click.echo("  (Could not fetch price data automatically)")
            actual_direction = click.prompt(
                "  Actual direction",
                type=click.Choice(["up", "down", "flat"]),
            )

        outcome = click.prompt(
            "\n  Outcome",
            type=click.Choice(list(VALID_OUTCOMES)),
            show_choices=True,
        )
        notes = click.prompt("  Resolution notes")

        try:
            resolve_prediction(
                prediction_file=pred["file"],
                outcome=outcome,
                actual_direction=actual_direction,
                resolution_notes=notes,
                resolution_date=check_date,
            )
        except Exception as exc:
            click.echo(f"  Error resolving {pred['file'].name}: {exc}", err=True)
            continue

        click.echo(f"  Resolved: {pred['file'].name}\n")

    click.echo("All predictions resolved.")


# ── brief ─────────────────────────────────────────────────────────────────────

@main.command("brief")
@click.option(
    "--analyze",
    is_flag=True,
    default=False,
    help="Synthesize all data with one DeepSeek call (requires OPENROUTER_API_KEY in .env).",
)
@click.option(
    "--date", "briefing_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Briefing date as YYYY-MM-DD (defaults to today).",
)
@click.pass_context
def brief_cmd(
    ctx: click.Context,
    analyze: bool,
    briefing_date: Optional[datetime],
) -> None:
    """Scan full watchlist and write a morning briefing to Trading/Briefings/.

    Produces a prioritized three-tier briefing:
    - Macro / Market (SPY, QQQ, VIX)
    - AI-sector news (headlines appearing in 3+ ticker feeds)
    - Watchlist — Needs attention / Watching / Quiet

    Without --analyze (default): data only, no API call.
    With --analyze: one DeepSeek call synthesizes all gathered data.
    """
    import os
    from weaver.briefing import generate_briefing

    vault_path = _vault(ctx)
    config = load_config(ctx.obj["config_path"])
    bd = briefing_date.date() if briefing_date else None

    api_key: Optional[str] = None
    if analyze:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            click.echo(
                "Error: --analyze requires OPENROUTER_API_KEY to be set in .env",
                err=True,
            )
            sys.exit(1)

    watchlist = config.get("watchlist", {})
    total = sum(len(v or []) for v in watchlist.values())

    if analyze:
        click.echo(
            f"Scanning {total} tickers and analyzing with DeepSeek "
            "(this may take 1-2 minutes)..."
        )
    else:
        click.echo(f"Scanning {total} tickers...")

    def _progress(ticker: str, ok: bool) -> None:
        click.echo(f"  {ticker} {'✓' if ok else '✗'}")

    try:
        filepath, needs_tickers = generate_briefing(
            vault_path=vault_path,
            config=config,
            analyze=analyze,
            briefing_date=bd,
            api_key=api_key,
            model=get_analyze_model(config),
            timeout=get_openrouter_timeout(config),
            progress_cb=_progress,
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if needs_tickers:
        click.echo(f"Needs attention: {', '.join(needs_tickers)}")
    else:
        click.echo("Nothing flagged today.")
    click.echo(f"Briefing written: {filepath}")


# ── research ──────────────────────────────────────────────────────────────────

@main.command("research")
@click.option("--ticker", required=True, help="Stock ticker to research (e.g. NVDA).")
@click.option(
    "--date", "research_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Research date as YYYY-MM-DD (defaults to today).",
)
@click.option(
    "--analyze",
    is_flag=True,
    default=False,
    help="Fill analysis sections with DeepSeek via OpenRouter (requires OPENROUTER_API_KEY in .env).",
)
@click.pass_context
def research_cmd(
    ctx: click.Context,
    ticker: str,
    research_date: Optional[datetime],
    analyze: bool,
) -> None:
    """Fetch market data for a ticker and write a research note to Trading/Research/.

    Without --analyze (default): data sections filled, analysis sections left blank.
    With --analyze: DeepSeek fills Setup/Bull/Bear/Key risks and adds a suggested call.
    """
    import os
    from weaver.research import generate_research_note

    vault_path = _vault(ctx)
    rd = research_date.date() if research_date else None
    config = load_config(ctx.obj["config_path"])

    api_key: Optional[str] = None
    if analyze:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            click.echo(
                "Error: --analyze requires OPENROUTER_API_KEY to be set in .env",
                err=True,
            )
            sys.exit(1)

    click.echo(f"Fetching data for {ticker.upper()}...")
    if analyze:
        click.echo("Analyzing with DeepSeek (this may take up to a minute)...")

    try:
        filepath = generate_research_note(
            vault_path=vault_path,
            ticker=ticker,
            research_date=rd,
            analyze=analyze,
            analyze_model=get_analyze_model(config),
            openrouter_api_key=api_key,
            openrouter_timeout=get_openrouter_timeout(config),
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Research note written: {filepath}")


# ── private helpers ───────────────────────────────────────────────────────────

def _prompt_future_date(prompt_text: str) -> date:
    """Prompt interactively for a YYYY-MM-DD date that must be in the future.

    Re-prompts with a clear message on bad format or a past date.
    """
    while True:
        raw = click.prompt(prompt_text)
        try:
            d = datetime.strptime(raw.strip(), "%Y-%m-%d").date()
        except ValueError:
            click.echo(
                "  Enter a date in YYYY-MM-DD format (e.g. 2026-07-21).",
                err=True,
            )
            continue
        if d <= date.today():
            click.echo(
                f"  Date must be in the future (got {d}, today is {date.today()}).",
                err=True,
            )
            continue
        return d


def _parse_future_date_flag(value: str) -> date:
    """Parse and validate a --resolve-by flag value. Exits 1 on error."""
    try:
        d = datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        click.echo(
            f"Error: --resolve-by must be in YYYY-MM-DD format, got {value!r}",
            err=True,
        )
        sys.exit(1)
    if d <= date.today():
        click.echo(
            f"Error: --resolve-by must be a future date "
            f"(got {d}, today is {date.today()})",
            err=True,
        )
        sys.exit(1)
    return d


def _fetch_actual_direction(
    ticker: str,
    prediction_date: Optional[date],
    resolve_by: date,
) -> Optional[str]:
    """Fetch actual price movement from yfinance.

    Returns a human-readable string like 'up (+4.2%, $150.00 → $156.30)'
    or None if the fetch fails for any reason.
    """
    try:
        import yfinance as yf
        from datetime import timedelta

        start = prediction_date or (resolve_by - timedelta(days=90))
        end = resolve_by + timedelta(days=1)

        hist = yf.download(
            ticker,
            start=str(start),
            end=str(end),
            progress=False,
            auto_adjust=True,
        )
        if hist is None or hist.empty or len(hist) < 2:
            return None

        close = hist["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]

        entry_price = float(close.iloc[0])
        exit_price = float(close.iloc[-1])
        pct = (exit_price - entry_price) / entry_price

        if pct > 0.005:
            label = "up"
        elif pct < -0.005:
            label = "down"
        else:
            label = "flat"

        return f"{label} ({pct:+.1%}, ${entry_price:.2f} → ${exit_price:.2f})"
    except Exception:
        return None
