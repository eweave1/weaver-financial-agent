"""Trade journal — log buy/sell entries as Obsidian markdown files.

Each entry is a self-contained markdown file with YAML frontmatter so
Obsidian's Dataview plugin can query across all trades.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Optional

VALID_ACTIONS: tuple[str, ...] = ("buy", "sell")
VALID_HORIZONS: tuple[str, ...] = ("day", "swing", "hold")


def log_trade(
    vault_path: Path,
    ticker: str,
    action: Literal["buy", "sell"],
    quantity: float,
    price: float,
    reason: str,
    stop_condition: str,
    time_horizon: Literal["day", "swing", "hold"],
    target_exit: Optional[float] = None,
    trade_date: Optional[date] = None,
    linked_buy_file: Optional[str] = None,
) -> Path:
    """Write a trade journal entry to Trading/Journal/ in the Obsidian vault.

    For sell entries, pass linked_buy_file as the filename stem of the
    corresponding buy entry (e.g. "2026-06-30 NVDA buy") so Obsidian can
    create a wikilink between them.

    Returns the path of the written file.
    """
    if action not in VALID_ACTIONS:
        raise ValueError(f"action must be one of {VALID_ACTIONS}, got {action!r}")
    if time_horizon not in VALID_HORIZONS:
        raise ValueError(
            f"time_horizon must be one of {VALID_HORIZONS}, got {time_horizon!r}"
        )

    ticker = ticker.upper()
    entry_date = trade_date or date.today()

    journal_dir = vault_path / "Trading" / "Journal"
    journal_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{entry_date} {ticker} {action}"
    filename = _unique_filename(journal_dir, stem)
    filepath = journal_dir / filename

    content = _build_trade_note(
        ticker=ticker,
        action=action,
        quantity=quantity,
        price=price,
        reason=reason,
        target_exit=target_exit,
        stop_condition=stop_condition,
        time_horizon=time_horizon,
        entry_date=entry_date,
        linked_buy_file=linked_buy_file,
    )
    filepath.write_text(content, encoding="utf-8")
    return filepath


# ── private helpers ──────────────────────────────────────────────────────────

def _build_trade_note(
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    reason: str,
    target_exit: Optional[float],
    stop_condition: str,
    time_horizon: str,
    entry_date: date,
    linked_buy_file: Optional[str],
) -> str:
    fm: list[str] = [
        "---",
        f"ticker: {ticker}",
        f"action: {action}",
        f"quantity: {quantity}",
        f"price: {price}",
        f'reason: "{_esc(reason)}"',
        f'stop_condition: "{_esc(stop_condition)}"',
        f"time_horizon: {time_horizon}",
        f"entry_date: {entry_date}",
        "status: open",
    ]
    if target_exit is not None:
        fm.append(f"target_exit: {target_exit}")
    if linked_buy_file:
        fm.append(f'linked_buy: "[[{_esc(linked_buy_file)}]]"')
    fm.append("---")

    body: list[str] = [
        f"# {entry_date} {ticker} {action.capitalize()}",
        "",
        f"**Price:** ${price:,.2f}  ",
        f"**Quantity:** {quantity}  ",
        f"**Time horizon:** {time_horizon}  ",
        "",
        "## Reason",
        "",
        reason,
        "",
        "## Exit conditions",
        "",
        f"- **Target:** {'${:,.2f}'.format(target_exit) if target_exit is not None else 'TBD'}",
        f"- **Stop:** {stop_condition}",
    ]
    if linked_buy_file:
        body += ["", "## Linked entry", "", f"[[{linked_buy_file}]]"]

    return "\n".join(fm) + "\n" + "\n".join(body) + "\n"


def _unique_filename(directory: Path, stem: str) -> str:
    """Return a .md filename that doesn't already exist in directory."""
    candidate = f"{stem}.md"
    if not (directory / candidate).exists():
        return candidate
    n = 2
    while (directory / f"{stem} ({n}).md").exists():
        n += 1
    return f"{stem} ({n}).md"


def _esc(value: str) -> str:
    """Escape double quotes for YAML inline strings."""
    return value.replace('"', '\\"')
