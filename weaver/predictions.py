"""Prediction log — record directional predictions and resolve them against outcomes.

Three resolution categories:
- right      — reasoning held and the call was correct
- wrong      — reasoning was tested and failed
- unforeseen — market moved for a reason outside the original thesis
               (tail event, macro shock, etc.); scored separately so it
               doesn't wrongly penalize sound analysis
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import yaml

VALID_DIRECTIONS: tuple[str, ...] = ("up", "down", "flat")
VALID_OUTCOMES: tuple[str, ...] = ("right", "wrong", "unforeseen")


def log_prediction(
    vault_path: Path,
    ticker: str,
    direction: Literal["up", "down", "flat"],
    timeframe: str,
    confidence: float,
    reasoning: str,
    resolve_by: date,
    prediction_date: Optional[date] = None,
    trigger: Optional[float] = None,
) -> Path:
    """Write a prediction entry to Trading/Predictions/ in the Obsidian vault.

    confidence is a float in [0.0, 1.0].
    resolve_by must be after prediction_date (or today if prediction_date is None).
    trigger is an optional explicit price level for the entry condition.

    Returns the path of the written file.
    """
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {VALID_DIRECTIONS}, got {direction!r}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")
    if trigger is not None and trigger <= 0:
        raise ValueError(f"trigger must be a positive price, got {trigger}")

    pred_date = prediction_date or date.today()
    if resolve_by <= pred_date:
        raise ValueError(
            f"resolve_by ({resolve_by}) must be in the future relative to "
            f"prediction_date ({pred_date})"
        )

    ticker = ticker.upper()
    pred_dir = vault_path / "Trading" / "Predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{pred_date} {ticker}"
    filename = _unique_filename(pred_dir, stem)
    filepath = pred_dir / filename

    content = _build_prediction_note(
        ticker=ticker,
        direction=direction,
        timeframe=timeframe,
        confidence=confidence,
        reasoning=reasoning,
        resolve_by=resolve_by,
        prediction_date=pred_date,
        trigger=trigger,
    )
    filepath.write_text(content, encoding="utf-8")
    return filepath


def list_pending_resolutions(
    vault_path: Path,
    as_of_date: Optional[date] = None,
) -> list[dict]:
    """Return all open predictions whose resolve_by date has passed.

    Each dict contains: file, ticker, direction, timeframe, confidence,
    reasoning, resolve_by, prediction_date.
    """
    check_date = as_of_date or date.today()
    pred_dir = vault_path / "Trading" / "Predictions"
    if not pred_dir.exists():
        return []

    pending: list[dict] = []
    for md_file in sorted(pred_dir.glob("*.md")):
        fm = _parse_frontmatter(md_file.read_text(encoding="utf-8"))
        if fm.get("status") != "open":
            continue
        resolve_by_raw = fm.get("resolve_by")
        if resolve_by_raw is None:
            continue
        resolve_by = _coerce_date(resolve_by_raw)
        if resolve_by is None or resolve_by > check_date:
            continue
        entry: dict = {
            "file": md_file,
            "ticker": fm.get("ticker"),
            "direction": fm.get("direction"),
            "timeframe": fm.get("timeframe"),
            "confidence": fm.get("confidence"),
            "reasoning": fm.get("reasoning"),
            "resolve_by": resolve_by,
            "prediction_date": _coerce_date(fm.get("prediction_date")),
        }
        if fm.get("trigger") is not None:
            try:
                entry["trigger"] = float(fm["trigger"])
            except (TypeError, ValueError):
                pass
        pending.append(entry)
    return pending


def resolve_prediction(
    prediction_file: Path,
    outcome: Literal["right", "wrong", "unforeseen"],
    actual_direction: str,
    resolution_notes: str,
    resolution_date: Optional[date] = None,
) -> None:
    """Update a prediction file in-place with its resolution outcome.

    Adds outcome, actual_direction, resolution_date, and resolution_notes to
    the frontmatter, changes status to 'resolved', and appends a ## Resolution
    section to the body.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"outcome must be one of {VALID_OUTCOMES}, got {outcome!r}")

    res_date = resolution_date or date.today()
    content = prediction_file.read_text(encoding="utf-8")

    # Replace "status: open" in the frontmatter with resolution fields + new status.
    # count=1 ensures we only touch the frontmatter, not any body text.
    resolution_block = (
        f"outcome: {outcome}\n"
        f"actual_direction: {actual_direction}\n"
        f"resolution_date: {res_date}\n"
        f'resolution_notes: "{_esc(resolution_notes)}"\n'
        f"status: resolved"
    )
    updated = re.sub(r"status: open", resolution_block, content, count=1)

    # Append a human-readable resolution section below the existing body.
    updated += (
        f"\n## Resolution\n\n"
        f"- **Outcome:** {outcome.capitalize()}\n"
        f"- **Actual direction:** {actual_direction}\n"
        f"- **Resolved:** {res_date}\n"
        f"- **Notes:** {resolution_notes}\n"
    )

    prediction_file.write_text(updated, encoding="utf-8")


# ── private helpers ──────────────────────────────────────────────────────────

def _build_prediction_note(
    ticker: str,
    direction: str,
    timeframe: str,
    confidence: float,
    reasoning: str,
    resolve_by: date,
    prediction_date: date,
    trigger: Optional[float] = None,
) -> str:
    confidence_pct = int(confidence * 100)
    fm: list[str] = [
        "---",
        f"ticker: {ticker}",
        f"direction: {direction}",
        f'timeframe: "{timeframe}"',
        f"confidence: {confidence}",
        f"resolve_by: {resolve_by}",
        f"prediction_date: {prediction_date}",
        f'reasoning: "{_esc(reasoning)}"',
        "status: open",
    ]
    if trigger is not None:
        fm.append(f"trigger: {trigger}")
    fm.append("---")

    body: list[str] = [
        f"# {prediction_date} {ticker} — {direction.upper()} ({confidence_pct}% confidence)",
        "",
        f"**Timeframe:** {timeframe}  ",
        f"**Resolve by:** {resolve_by}  ",
        f"**Confidence:** {confidence_pct}%  ",
    ]
    if trigger is not None:
        body.append(f"**Trigger:** ${trigger:,.2f}  ")
    body += [
        "",
        "## Reasoning",
        "",
        reasoning,
    ]

    return "\n".join(fm) + "\n" + "\n".join(body) + "\n"


def _parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from a markdown file."""
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}
    return yaml.safe_load(match.group(1)) or {}


def _coerce_date(value: object) -> Optional[date]:
    """Accept a date object, a datetime.date, or an ISO string; return a date."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        from datetime import datetime
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _unique_filename(directory: Path, stem: str) -> str:
    candidate = f"{stem}.md"
    if not (directory / candidate).exists():
        return candidate
    n = 2
    while (directory / f"{stem} ({n}).md").exists():
        n += 1
    return f"{stem} ({n}).md"


def _esc(value: str) -> str:
    return value.replace('"', '\\"')
