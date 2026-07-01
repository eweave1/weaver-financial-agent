# weaver-financial-agent

Personal financial research agent for active AI-sector stock trading. Researches,
monitors, journals, and makes logged predictions — but **never executes trades**.
Every trade decision is made and placed manually.

## Philosophy: notify and review

This agent surfaces information; I make every call. It never says "buy X" or
"target $Y." Every output requires active review before any real-world action.

This constraint is intentional and permanent. The value is in building a
disciplined, auditable research process — not in automating decisions.

## Architecture

This repo (`weaver-financial-agent`) is a **standalone Python package**. It
handles data fetching, analysis, report generation, and markdown formatting.
It writes files directly to an Obsidian vault. It has no LLM dependency and no
dependency on any delivery layer — it runs and tests cleanly on its own.

A separate layer (Hermes skills, built later) schedules the package's functions,
applies LLM reasoning via DeepSeek, and delivers outputs to Telegram. That
separation means the financial logic here survives even if the delivery layer
changes.

```
weaver-financial-agent/   ← this repo
├── weaver/               ← Python package: data, analysis, markdown
└── tests/

hermes-skills/ (separate, later)
└── financial/            ← scheduling, LLM calls, Telegram delivery
```

## Obsidian vault structure

All trading files live in a top-level `Trading/` folder in the vault:

```
Trading/
├── Journal/       ← trade entries (buy/sell)
├── Predictions/   ← prediction log with resolution tracking
├── Research/      ← pre-trade research notes (Component 3)
├── Briefings/     ← daily morning briefings (Component 1)
└── Reviews/       ← weekly/monthly reviews (later)
```

Files use YAML frontmatter throughout so Obsidian's Dataview plugin can query
across entries — filter open trades, score prediction accuracy, etc.

## Components

| Component | Status | Description |
|-----------|--------|-------------|
| 4 — Trade Journal + Prediction Log | **Built** | Log trades, log predictions, resolve predictions with three outcome categories |
| 1 — Morning Briefing | Planned | Daily macro + watchlist digest, 7 AM ET on trading days |
| 3 — Pre-Trade Research | Planned | Per-ticker deep dive with steelmanned bull/bear cases |
| 2 — Catalyst Scanner | Planned | Market-hours monitor: filings, volume spikes, moves >3% |

## Setup

**Requirements:** Python 3.11+

```bash
# Clone
git clone https://github.com/eweave1/weaver-financial-agent.git
cd weaver-financial-agent

# Create and activate virtualenv
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install package + dev dependencies
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env: set OBSIDIAN_VAULT_PATH to your vault root
```

## Usage

All commands are run via the `wf` CLI.

### Log a trade

```bash
# Buy
wf log-trade \
  --ticker NVDA \
  --action buy \
  --quantity 10 \
  --price 150.00 \
  --reason "Breakout above 150 on above-average volume, sector momentum" \
  --stop "Close below 145" \
  --horizon swing \
  --target 170.00

# Sell (link back to the buy entry)
wf log-trade \
  --ticker NVDA \
  --action sell \
  --quantity 10 \
  --price 165.00 \
  --reason "Target zone reached, starting to show distribution" \
  --stop "N/A" \
  --horizon swing \
  --linked-buy "2026-06-30 NVDA buy"
```

This writes `Trading/Journal/2026-06-30 NVDA buy.md` with YAML frontmatter
carrying all structured fields for Dataview queries.

### Log a prediction

```bash
wf log-prediction \
  --ticker NVDA \
  --direction up \
  --timeframe "2 weeks" \
  --confidence 0.70 \
  --reasoning "Earnings catalyst + AI capex cycle accelerating. Watching for hold above 150." \
  --resolve-by 2026-07-14
```

Writes `Trading/Predictions/2026-06-30 NVDA.md`.

### Resolve overdue predictions

```bash
wf resolve-predictions
```

Lists all open predictions past their resolve-by date. For each one, fetches
the actual price movement from yfinance and prompts for the outcome:

- **Right** — reasoning held and the call was correct
- **Wrong** — reasoning was tested and failed
- **Unforeseen** — market moved for a reason outside the original thesis (tail
  event, macro shock, etc.); scored separately so it doesn't wrongly penalize
  sound analysis

The distinction between *wrong* and *unforeseen* is yours to make — the agent
shows you the actual price data and asks.

### Override config or vault path

```bash
# Use a different config file
wf --config /path/to/config.yaml log-trade ...

# Override vault path without editing .env
OBSIDIAN_VAULT_PATH=/tmp/test-vault wf log-trade ...
```

## Running tests

```bash
pytest
pytest --cov=weaver --cov-report=term-missing   # with coverage
```

Tests use `tmp_path` (pytest's built-in temp directory fixture) — no vault
setup needed, no network calls.

## Watchlist and config

Edit `config.yaml` to adjust the watchlist or data settings. No code changes
needed. The vault path lives in `.env` (or `OBSIDIAN_VAULT_PATH` env var) so
it can differ between machines without touching config.

## Paper trading policy

No real capital until at least one month of consistent paper-trading
profitability across varied market conditions. The prediction resolution system
exists partly to hold that bar honestly.
