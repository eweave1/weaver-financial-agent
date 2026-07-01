"""CLI entry points for weaver-financial-agent.

Usage:
    wf log-trade      --ticker NVDA --action buy ...
    wf log-prediction --ticker NVDA --direction up ...
    wf resolve-predictions
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import click

from weaver.config import get_vault_path, load_config
from weaver.journal import VALID_ACTIONS, VALID_HORIZONS, log_trade
from weaver.predictions import (
    VALID_DIRECTIONS,
    VALID_OUTCOMES,
    list_pending_resolutions,
    log_prediction,
    resolve_prediction,
)


# ── group ────────────────────────────────────────────────────────────────────

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
    config = load_config(ctx.obj["config_path"])
    return get_vault_path(config)


# ── log-trade ────────────────────────────────────────────────────────────────

@main.command("log-trade")
@click.option("--ticker", required=True, help="Stock ticker (e.g. NVDA).")
@click.option(
    "--action",
    required=True,
    type=click.Choice(list(VALID_ACTIONS)),
    help="buy or sell.",
)
@click.option("--quantity", required=True, type=float, help="Number of shares.")
@click.option("--price", required=True, type=float, help="Execution price per share.")
@click.option("--reason", required=True, help="One-sentence reason for this trade.")
@click.option(
    "--stop",
    "stop_condition",
    required=True,
    help="Stop/exit condition (e.g. 'Close below 145').",
)
@click.option(
    "--horizon",
    "time_horizon",
    required=True,
    type=click.Choice(list(VALID_HORIZONS)),
    help="Time horizon: day, swing, or hold.",
)
@click.option(
    "--target",
    "target_exit",
    type=float,
    default=None,
    help="Target exit price (optional).",
)
@click.option(
    "--date",
    "trade_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Trade date as YYYY-MM-DD (defaults to today).",
)
@click.option(
    "--linked-buy",
    default=None,
    help="For sell entries: filename stem of the linked buy entry "
         "(e.g. '2026-06-30 NVDA buy').",
)
@click.pass_context
def log_trade_cmd(
    ctx: click.Context,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    reason: str,
    stop_condition: str,
    time_horizon: str,
    target_exit: Optional[float],
    trade_date: Optional[datetime],
    linked_buy: Optional[str],
) -> None:
    """Log a buy or sell trade to the journal."""
    vault_path = _vault(ctx)
    td = trade_date.date() if trade_date else None
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
    click.echo(f"Trade logged: {filepath}")


# ── log-prediction ───────────────────────────────────────────────────────────

@main.command("log-prediction")
@click.option("--ticker", required=True, help="Stock ticker (e.g. NVDA).")
@click.option(
    "--direction",
    required=True,
    type=click.Choice(list(VALID_DIRECTIONS)),
    help="Predicted direction: up, down, or flat.",
)
@click.option(
    "--timeframe",
    required=True,
    help="Prediction timeframe (e.g. '2 weeks', '1 month').",
)
@click.option(
    "--confidence",
    required=True,
    type=float,
    help="Confidence level from 0.0 to 1.0.",
)
@click.option(
    "--reasoning",
    required=True,
    help="Your reasoning for this prediction.",
)
@click.option(
    "--resolve-by",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Date to resolve the prediction by (YYYY-MM-DD).",
)
@click.option(
    "--date",
    "prediction_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    help="Prediction date as YYYY-MM-DD (defaults to today).",
)
@click.pass_context
def log_prediction_cmd(
    ctx: click.Context,
    ticker: str,
    direction: str,
    timeframe: str,
    confidence: float,
    reasoning: str,
    resolve_by: datetime,
    prediction_date: Optional[datetime],
) -> None:
    """Log a directional prediction with a resolve-by date."""
    vault_path = _vault(ctx)
    pd = prediction_date.date() if prediction_date else None
    filepath = log_prediction(
        vault_path=vault_path,
        ticker=ticker,
        direction=direction,
        timeframe=timeframe,
        confidence=confidence,
        reasoning=reasoning,
        resolve_by=resolve_by.date(),
        prediction_date=pd,
    )
    click.echo(f"Prediction logged: {filepath}")


# ── resolve-predictions ──────────────────────────────────────────────────────

@main.command("resolve-predictions")
@click.option(
    "--date",
    "as_of_date",
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

        # Best-effort: fetch actual price movement from yfinance
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

        resolve_prediction(
            prediction_file=pred["file"],
            outcome=outcome,
            actual_direction=actual_direction,
            resolution_notes=notes,
            resolution_date=check_date,
        )
        click.echo(f"  Resolved: {pred['file'].name}\n")

    click.echo("All predictions resolved.")


# ── helpers ──────────────────────────────────────────────────────────────────

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
        end = resolve_by + timedelta(days=1)  # yfinance end is exclusive

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
        # yfinance >=0.2 may return a DataFrame with MultiIndex columns for
        # single-ticker downloads in some configurations; flatten if needed.
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
