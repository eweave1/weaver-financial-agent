"""Trade journal — log buy/sell entries as Obsidian markdown files.

Each entry is a self-contained markdown file with YAML frontmatter so
Obsidian's Dataview plugin can query across all trades.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Literal, Optional

from weaver.predictions import _parse_frontmatter

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
    stop_price: Optional[float] = None,
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
        stop_price=stop_price,
        time_horizon=time_horizon,
        entry_date=entry_date,
        linked_buy_file=linked_buy_file,
    )
    filepath.write_text(content, encoding="utf-8")
    return filepath


def find_open_buys(journal_dir: Path, ticker: str) -> list[Path]:
    """Return open buy file paths for ticker, sorted by filename (oldest first)."""
    if not journal_dir.exists():
        return []
    candidates = []
    for f in sorted(journal_dir.glob("*.md")):
        fm = _parse_frontmatter(f.read_text(encoding="utf-8"))
        if (
            fm.get("action") == "buy"
            and str(fm.get("ticker", "")).upper() == ticker.upper()
            and fm.get("status", "open") == "open"
        ):
            candidates.append(f)
    return candidates


def close_buy_position(
    buy_path: Path,
    sell_qty: float,
    sell_price: float,
    sell_date: date,
) -> str:
    """Update a buy file's frontmatter to record a sell. Returns 'closed' or 'partial'.

    Append-only in spirit: original quantity is preserved; quantity_remaining and
    quantity_sold are added for partial closes so the history is never rewritten.
    """
    content = buy_path.read_text(encoding="utf-8")

    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"No valid frontmatter in {buy_path}")

    fm = _parse_frontmatter(content)
    buy_qty = float(fm.get("quantity", 0) or 0)
    buy_price = float(fm.get("price", sell_price) or sell_price)
    closed_qty = min(sell_qty, buy_qty)
    realized_pnl = round((sell_price - buy_price) * closed_qty, 2)

    new_status = "closed" if sell_qty >= buy_qty else "partial"

    fm_lines = parts[1].lstrip("\n").rstrip("\n").split("\n")
    updated = []
    for line in fm_lines:
        if line.startswith("status:"):
            updated.append(f"status: {new_status}")
        else:
            updated.append(line)

    updated.append(f"exit_price: {sell_price}")
    updated.append(f"exit_date: {sell_date}")
    updated.append(f"realized_pnl: {realized_pnl}")
    if new_status == "partial":
        updated.append(f"quantity_remaining: {round(buy_qty - sell_qty, 4)}")
        updated.append(f"quantity_sold: {sell_qty}")

    body = parts[2]
    buy_path.write_text("---\n" + "\n".join(updated) + "\n---" + body, encoding="utf-8")
    return new_status


def scan_open_positions(
    vault_path: Path,
    current_prices: dict[str, float],
) -> list[dict[str, Any]]:
    """Return open buy entries from the journal.

    Reads status directly: 'closed' → excluded, 'partial' → included with
    quantity_remaining as effective quantity. Falls back to sell-reference
    inference for pre-Build1 entries that still have status: open.
    """
    journal_dir = vault_path / "Trading" / "Journal"
    if not journal_dir.exists():
        return []

    # Collect sold stems from sell files (backward compat for pre-Build1 entries)
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

        status = fm.get("status", "open")
        if status == "closed":
            continue
        # Fall back to sell-reference check for legacy open entries
        if status == "open" and f.stem in sold_stems:
            continue

        ticker = str(fm.get("ticker", "")).upper()
        entry_price = fm.get("price")
        current = current_prices.get(ticker)

        # For partial sells, use quantity_remaining as effective quantity
        raw_qty = fm.get("quantity")
        qty_remaining = fm.get("quantity_remaining")
        effective_qty = qty_remaining if qty_remaining is not None else raw_qty

        stop_price_raw = fm.get("stop_price")
        stop_price: Optional[float] = None
        if stop_price_raw is not None:
            try:
                stop_price = float(stop_price_raw)
            except (TypeError, ValueError):
                pass

        pos: dict[str, Any] = {
            "ticker": ticker,
            "entry_date": fm.get("entry_date"),
            "entry_price": entry_price,
            "current_price": current,
            "quantity": effective_qty,
            "time_horizon": fm.get("time_horizon"),
            "file_stem": f.stem,
            "status": status,
            "stop_price": stop_price,
        }
        if entry_price is not None and current is not None:
            pos["pnl_pct"] = (float(current) - float(entry_price)) / float(entry_price)

        positions.append(pos)

    return positions


# ── private helpers ──────────────────────────────────────────────────────────

def _build_trade_note(
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    reason: str,
    target_exit: Optional[float],
    stop_condition: str,
    stop_price: Optional[float],
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
    if stop_price is not None:
        fm.append(f"stop_price: {stop_price}")
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
