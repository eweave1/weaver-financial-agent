# weaver-financial-agent

A personal AI-powered research agent for equity analysis, focused on the AI/semiconductor sector.

It researches, monitors markets, journals trades, and tracks its own predictions against reality — but it **never executes trades**. Every decision stays with the human.

## Philosophy

Most "AI trading bot" projects optimize for autonomy: give the machine money and let it trade. This one is built on the opposite principle.

The evidence on autonomous retail trading is not encouraging, and handing capital to a self-directed agent — especially one that can be misled by injected content or a bad data feed — is a fast way to lose money. So this agent is deliberately constrained to the role where AI actually adds value: **synthesis and research, not decision-making.**

The design principle is **notify and review**. The agent surfaces information, builds steelmanned bull and bear cases, and logs predictions. The human reads, thinks, and decides. Nothing reaches a broker automatically. Ever.

The secondary goal is learning. Every prediction the agent makes — and every trade the human makes — gets logged with its reasoning and later scored against what actually happened. Over time this builds an honest track record of *both* the agent's analysis and the human's judgment, so the real question can be answered: does this system actually make better decisions, or just more confident ones?

## What it does

The agent is built from four components:

**Trade Journal & Prediction Log** — Records every trade with its reasoning, target, and stop. Separately logs predictions with a resolve-by date, then scores them against actual outcomes using three categories: *right* (the reasoning held), *wrong* (the reasoning was tested and failed), and *unforeseen* (the market moved for a reason outside the original analysis — a tail event, scored separately so it doesn't discredit sound reasoning).

**Morning Briefing** — A pre-market summary delivered daily, structured in three tiers: macro/market conditions, AI-sector news, and detailed coverage of a configurable watchlist.

**Pre-Trade Research** — On-demand deep-dive notes on a given ticker: business health, valuation context, current setup, steelmanned bull and bear cases, key risks, and a recap of the user's own prior views on that stock.

**Catalyst Scanner** — Market-hours monitoring for meaningful events (filings, unusual volume, large moves, rating changes, insider transactions) across both the watchlist and a broader universe, filtered by a priority score to cut noise.

## Architecture

The project is split into two layers by design:

- **This repository** is a standalone Python package containing all domain logic — data fetching, analysis, report generation. It calls no language models and depends on no agent framework. It can be run and tested entirely on its own.
- **The agent layer** (built separately) wraps this package with an LLM for reasoning and a messaging interface for delivery. Because the financial logic lives here independently, the agent layer can be swapped or replaced without touching the core.

This separation keeps the analytical code testable, portable, and honest about what it is: a research tool, not a black box.

## Data sources

Built on free, public data to start:

- **yfinance** — price, volume, fundamentals, news
- **SEC EDGAR** — filings and insider transactions
- **FRED** — macroeconomic indicators

No paid data feeds unless a specific, demonstrated gap justifies one.

## Model routing

The reasoning layer routes tasks by cost and difficulty: a fast, inexpensive model handles high-volume mechanical work (summaries, filtering), while a stronger model is reserved for the tasks where reasoning quality actually affects a decision (research notes, briefing synthesis).

## Status

Early and in active development. Built component by component, tested against a month of paper trading before any real capital is ever considered.

## Disclaimer

This is a personal learning project. It is not investment advice, and it produces none. Nothing in this repository recommends buying or selling any security. The author is not a financial advisor. Markets carry real risk of loss; do your own research and make your own decisions.

## License

MIT
