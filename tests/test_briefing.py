"""Tests for weaver.briefing module."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from weaver.briefing import (
    _format_trigger,
    bucket_tickers,
    days_to_earnings,
    find_sector_news,
    generate_briefing,
    scan_open_positions,
    scan_open_predictions,
    synthesize_with_ai,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "Trading" / "Briefings").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def config() -> dict[str, Any]:
    return {
        "watchlist": {
            "chips": ["NVDA", "AMD"],
            "baselines": ["SPY", "QQQ"],
        },
        "briefing": {
            "move_threshold_pct": 3.0,
            "volume_threshold": 1.5,
            "earnings_warning_days": 7,
            "watching_move_threshold_pct": 1.5,
            "watching_earnings_days": 14,
        },
    }


def _make_snapshot(
    change_pct: float = 0.002,
    vol_ratio: float = 1.0,
    next_earnings: str | None = None,
) -> dict[str, Any]:
    base_price = 100.0
    return {
        "current_price": base_price,
        "prev_close": base_price / (1 + change_pct) if change_pct != -1 else base_price,
        "day_change_pct": change_pct,
        "volume": 10_000_000,
        "avg_volume": int(10_000_000 / vol_ratio) if vol_ratio else 10_000_000,
        "volume_ratio": vol_ratio,
        **({"next_earnings": next_earnings} if next_earnings else {}),
    }


def _make_entry(
    ticker: str,
    change_pct: float = 0.002,
    vol_ratio: float = 1.0,
    news: list | None = None,
    next_earnings: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if error:
        return {"snapshot": {}, "news": [], "error": error}
    return {
        "snapshot": _make_snapshot(change_pct, vol_ratio, next_earnings),
        "news": news or [],
        "error": None,
    }


def _full_ticker_data(config: dict) -> dict[str, Any]:
    from weaver.config import get_watchlist
    return {t: _make_entry(t) for t in get_watchlist(config)}


# ── TestDaysToEarnings ────────────────────────────────────────────────────────


class TestDaysToEarnings:
    def test_future_earnings(self):
        future = str(date.today() + timedelta(days=5))
        assert days_to_earnings({"next_earnings": future}) == 5

    def test_past_earnings_returns_none(self):
        past = str(date.today() - timedelta(days=1))
        assert days_to_earnings({"next_earnings": past}) is None

    def test_missing_returns_none(self):
        assert days_to_earnings({}) is None

    def test_na_returns_none(self):
        assert days_to_earnings({"next_earnings": "N/A"}) is None

    def test_today_returns_zero(self):
        assert days_to_earnings({"next_earnings": str(date.today())}) == 0


# ── TestFindSectorNews ────────────────────────────────────────────────────────


class TestFindSectorNews:
    def test_headline_in_three_tickers_is_sector(self):
        data = {
            t: {"news": [{"title": "AI chip exports restricted"}], "snapshot": {}, "error": None}
            for t in ["NVDA", "AMD", "MSFT"]
        }
        assert "AI chip exports restricted" in find_sector_news(data)

    def test_headline_in_two_tickers_not_sector(self):
        data = {
            t: {"news": [{"title": "NVDA beats earnings"}], "snapshot": {}, "error": None}
            for t in ["NVDA", "AMD"]
        }
        assert find_sector_news(data) == []

    def test_empty_news_returns_empty(self):
        data = {"NVDA": {"news": [], "snapshot": {}, "error": None}}
        assert find_sector_news(data) == []

    def test_capped_at_five(self):
        data: dict[str, Any] = {}
        for i in range(7):
            headline = f"Big sector story {i}"
            for suffix in ["A", "B", "C"]:
                key = f"T{i}{suffix}"
                data[key] = {
                    "news": [{"title": headline}],
                    "snapshot": {},
                    "error": None,
                }
        assert len(find_sector_news(data)) <= 5


# ── TestBucketTickers ─────────────────────────────────────────────────────────


class TestBucketTickers:
    def _bucket(self, config, **kwargs) -> dict:
        ticker = kwargs.pop("ticker", "TEST")
        return bucket_tickers({ticker: _make_entry(ticker, **kwargs)}, config)

    def test_large_move_needs_attention(self, config):
        result = self._bucket(config, ticker="NVDA", change_pct=0.042)
        assert any(e["ticker"] == "NVDA" for e in result["needs_attention"])
        assert not any(e["ticker"] == "NVDA" for e in result["quiet"])

    def test_large_negative_move_needs_attention(self, config):
        result = self._bucket(config, ticker="VRT", change_pct=-0.04)
        assert any(e["ticker"] == "VRT" for e in result["needs_attention"])

    def test_small_move_quiet(self, config):
        result = self._bucket(config, ticker="ASML", change_pct=0.002, vol_ratio=1.0)
        assert any(e["ticker"] == "ASML" for e in result["quiet"])

    def test_high_volume_needs_attention(self, config):
        result = self._bucket(config, ticker="AMD", change_pct=0.001, vol_ratio=1.6)
        assert any(e["ticker"] == "AMD" for e in result["needs_attention"])

    def test_earnings_within_7_days_needs_attention(self, config):
        future = str(date.today() + timedelta(days=5))
        result = self._bucket(config, ticker="NVDA", change_pct=0.001, vol_ratio=1.0, next_earnings=future)
        assert any(e["ticker"] == "NVDA" for e in result["needs_attention"])

    def test_earnings_30_days_away_quiet(self, config):
        future = str(date.today() + timedelta(days=30))
        result = self._bucket(config, ticker="NVDA", change_pct=0.001, vol_ratio=1.0, next_earnings=future)
        assert any(e["ticker"] == "NVDA" for e in result["quiet"])

    def test_moderate_move_watching(self, config):
        # 2% move: between 1.5% watch threshold and 3% needs-attention threshold
        result = self._bucket(config, ticker="MSFT", change_pct=0.02)
        assert any(e["ticker"] == "MSFT" for e in result["watching"])

    def test_small_move_below_watch_threshold_quiet(self, config):
        # 1.4% move: below 1.5% watch threshold → Quiet (matches user's MSFT example)
        result = self._bucket(config, ticker="MSFT", change_pct=0.014)
        assert any(e["ticker"] == "MSFT" for e in result["quiet"])

    def test_earnings_8_to_14_days_watching(self, config):
        future = str(date.today() + timedelta(days=10))
        result = self._bucket(config, ticker="NVDA", change_pct=0.001, vol_ratio=1.0, next_earnings=future)
        assert any(e["ticker"] == "NVDA" for e in result["watching"])

    def test_elevated_volume_alone_goes_to_quiet(self, config):
        # Volume below the needs_attention threshold doesn't qualify for Watching
        result = self._bucket(config, ticker="CRM", change_pct=0.002, vol_ratio=1.3)
        assert any(e["ticker"] == "CRM" for e in result["quiet"])

    def test_news_alone_does_not_qualify_for_needs_attention(self, config):
        news = [{"title": "CRM beats estimates", "publisher": "Reuters", "age": "1h ago", "url": ""}]
        result = self._bucket(config, ticker="CRM", change_pct=0.002, vol_ratio=1.0, news=news)
        # News is supporting detail — not a qualifying criterion
        assert not any(e["ticker"] == "CRM" for e in result["needs_attention"])
        assert any(e["ticker"] == "CRM" for e in result["quiet"])

    def test_error_ticker_goes_to_failed(self, config):
        result = bucket_tickers({"BROKEN": _make_entry("BROKEN", error="timeout")}, config)
        assert "BROKEN" in result["failed"]
        assert not any(e["ticker"] == "BROKEN" for e in result["needs_attention"])

    def test_needs_attention_sorted_by_magnitude(self, config):
        data = {
            "NVDA": _make_entry("NVDA", change_pct=0.042),
            "AMD": _make_entry("AMD", change_pct=-0.051),
        }
        result = bucket_tickers(data, config)
        tickers = [e["ticker"] for e in result["needs_attention"]]
        # AMD has bigger absolute move, so comes first
        assert tickers.index("AMD") < tickers.index("NVDA")

    def test_exact_threshold_boundary(self, config):
        # Exactly 3.0% — should be needs_attention (>=)
        result = self._bucket(config, ticker="T", change_pct=0.03)
        assert any(e["ticker"] == "T" for e in result["needs_attention"])

    def test_just_below_threshold_watching_range(self, config):
        # 2.9% — below 3% needs-attention but above 1.5% watching threshold
        result = self._bucket(config, ticker="T", change_pct=0.029)
        assert any(e["ticker"] == "T" for e in result["watching"])


# ── TestScanOpenPositions ─────────────────────────────────────────────────────


class TestScanOpenPositions:
    def _write_journal(
        self,
        vault: Path,
        stem: str,
        action: str,
        ticker: str,
        price: float,
        linked_buy: str | None = None,
    ) -> None:
        journal_dir = vault / "Trading" / "Journal"
        journal_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "---",
            f"ticker: {ticker}",
            f"action: {action}",
            "quantity: 10",
            f"price: {price}",
            "entry_date: 2026-07-01",
            "time_horizon: swing",
            "status: open",
        ]
        if linked_buy:
            lines.append(f'linked_buy: "[[{linked_buy}]]"')
        lines.append("---")
        (journal_dir / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8")

    def test_open_buy_appears(self, vault):
        self._write_journal(vault, "2026-07-01 NVDA buy", "buy", "NVDA", 120.0)
        positions = scan_open_positions(vault, {"NVDA": 125.0})
        assert any(p["ticker"] == "NVDA" for p in positions)

    def test_linked_sell_excludes_buy(self, vault):
        self._write_journal(vault, "2026-07-01 NVDA buy", "buy", "NVDA", 120.0)
        self._write_journal(
            vault, "2026-07-05 NVDA sell", "sell", "NVDA", 130.0,
            linked_buy="2026-07-01 NVDA buy",
        )
        positions = scan_open_positions(vault, {"NVDA": 125.0})
        assert not any(p["ticker"] == "NVDA" for p in positions)

    def test_pnl_computed_correctly(self, vault):
        self._write_journal(vault, "2026-07-01 AMD buy", "buy", "AMD", 100.0)
        positions = scan_open_positions(vault, {"AMD": 110.0})
        pos = next(p for p in positions if p["ticker"] == "AMD")
        assert abs(pos["pnl_pct"] - 0.10) < 0.001

    def test_no_current_price_pnl_absent(self, vault):
        self._write_journal(vault, "2026-07-01 MSFT buy", "buy", "MSFT", 400.0)
        positions = scan_open_positions(vault, {})
        pos = next(p for p in positions if p["ticker"] == "MSFT")
        assert pos["current_price"] is None
        assert "pnl_pct" not in pos

    def test_sell_only_not_in_positions(self, vault):
        self._write_journal(vault, "2026-07-01 NVDA sell", "sell", "NVDA", 130.0)
        assert not any(
            p["ticker"] == "NVDA"
            for p in scan_open_positions(vault, {"NVDA": 125.0})
        )

    def test_empty_journal_dir(self, vault):
        (vault / "Trading" / "Journal").mkdir(parents=True, exist_ok=True)
        assert scan_open_positions(vault, {}) == []

    def test_no_journal_dir(self, vault):
        assert scan_open_positions(vault, {}) == []


# ── TestFormatTrigger ─────────────────────────────────────────────────────────


class TestFormatTrigger:
    def test_no_trigger_returns_dash(self):
        assert _format_trigger(None, 130.0, "up") == "—"

    def test_no_current_returns_price_only(self):
        assert _format_trigger(145.0, None, "up") == "$145.00"

    def test_up_triggered_when_current_at_or_above(self):
        assert "(TRIGGERED)" in _format_trigger(145.0, 145.0, "up")
        assert "(TRIGGERED)" in _format_trigger(145.0, 150.0, "up")

    def test_down_triggered_when_current_at_or_below(self):
        assert "(TRIGGERED)" in _format_trigger(145.0, 145.0, "down")
        assert "(TRIGGERED)" in _format_trigger(145.0, 140.0, "down")

    def test_up_not_triggered_shows_below(self):
        result = _format_trigger(145.0, 130.0, "up")
        assert "below" in result
        assert "TRIGGERED" not in result

    def test_down_not_triggered_shows_above(self):
        result = _format_trigger(145.0, 160.0, "down")
        assert "above" in result
        assert "TRIGGERED" not in result

    def test_no_direction_still_shows_above_or_below(self):
        result = _format_trigger(145.0, 130.0, None)
        assert "below" in result or "above" in result

    def test_distance_percentage_accurate(self):
        result = _format_trigger(200.0, 190.0, "up")
        assert "5.0%" in result


# ── TestScanOpenPredictions ───────────────────────────────────────────────────


class TestScanOpenPredictions:
    def _write_pred(
        self,
        vault: Path,
        stem: str,
        ticker: str,
        direction: str,
        resolve_by: str,
        status: str = "open",
        reasoning: str = "test reasoning",
        trigger: float | None = None,
    ) -> None:
        pred_dir = vault / "Trading" / "Predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            "---",
            f"ticker: {ticker}",
            f"direction: {direction}",
            "confidence: 0.65",
            f"resolve_by: {resolve_by}",
            "prediction_date: 2026-07-01",
            f'reasoning: "{reasoning}"',
            f"status: {status}",
        ]
        if trigger is not None:
            lines.append(f"trigger: {trigger}")
        lines.append("---")
        (pred_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_open_prediction_appears(self, vault):
        future = str(date.today() + timedelta(days=10))
        self._write_pred(vault, "2026-07-01 AMD", "AMD", "up", future)
        preds = scan_open_predictions(vault)
        assert any(p["ticker"] == "AMD" for p in preds)

    def test_resolved_prediction_excluded(self, vault):
        future = str(date.today() + timedelta(days=10))
        self._write_pred(vault, "2026-07-01 AMD", "AMD", "up", future, status="resolved")
        assert not any(p["ticker"] == "AMD" for p in scan_open_predictions(vault))

    def test_days_left_computed(self, vault):
        future = str(date.today() + timedelta(days=15))
        self._write_pred(vault, "2026-07-01 NVDA", "NVDA", "up", future)
        pred = next(p for p in scan_open_predictions(vault) if p["ticker"] == "NVDA")
        assert pred["days_left"] == 15
        assert not pred["overdue"]

    def test_overdue_prediction(self, vault):
        past = str(date.today() - timedelta(days=3))
        self._write_pred(vault, "2026-06-01 VRT", "VRT", "down", past)
        pred = next(p for p in scan_open_predictions(vault) if p["ticker"] == "VRT")
        assert pred["overdue"]
        assert pred["days_left"] == -3

    def test_no_predictions_dir_returns_empty(self, vault):
        assert scan_open_predictions(vault) == []

    def test_current_price_populated_when_provided(self, vault):
        future = str(date.today() + timedelta(days=10))
        self._write_pred(vault, "2026-07-01 NVDA", "NVDA", "up", future)
        pred = next(
            p for p in scan_open_predictions(vault, current_prices={"NVDA": 132.50})
            if p["ticker"] == "NVDA"
        )
        assert pred["current_price"] == 132.50

    def test_current_price_none_when_not_provided(self, vault):
        future = str(date.today() + timedelta(days=10))
        self._write_pred(vault, "2026-07-01 NVDA", "NVDA", "up", future)
        pred = next(p for p in scan_open_predictions(vault) if p["ticker"] == "NVDA")
        assert pred["current_price"] is None

    def test_explicit_frontmatter_trigger_read(self, vault):
        future = str(date.today() + timedelta(days=10))
        self._write_pred(
            vault, "2026-07-01 NVDA", "NVDA", "up", future, trigger=145.0,
        )
        pred = next(p for p in scan_open_predictions(vault) if p["ticker"] == "NVDA")
        assert pred["trigger_price"] == 145.0

    def test_reasoning_text_does_not_set_trigger(self, vault):
        future = str(date.today() + timedelta(days=10))
        self._write_pred(
            vault, "2026-07-01 NVDA", "NVDA", "up", future,
            reasoning="NVDA breaks above $145 as AI demand accelerates",
        )
        pred = next(p for p in scan_open_predictions(vault) if p["ticker"] == "NVDA")
        assert pred.get("trigger_price") is None

    def test_no_trigger_frontmatter_shows_dash(self, vault):
        future = str(date.today() + timedelta(days=10))
        self._write_pred(vault, "2026-07-01 AMD", "AMD", "up", future, reasoning="momentum play")
        pred = next(p for p in scan_open_predictions(vault) if p["ticker"] == "AMD")
        assert pred.get("trigger_price") is None


# ── TestGenerateBriefing ──────────────────────────────────────────────────────


class TestGenerateBriefing:
    def test_note_written_to_briefings_dir(self, vault, config):
        filepath, _ = generate_briefing(
            vault_path=vault, config=config, _ticker_data=_full_ticker_data(config), _vix=18.5,
        )
        assert filepath.exists()
        assert filepath.parent == vault / "Trading" / "Briefings"
        assert filepath.name == f"{date.today()}.md"

    def test_custom_date(self, vault, config):
        filepath, _ = generate_briefing(
            vault_path=vault,
            config=config,
            briefing_date=date(2026, 7, 4),
            _ticker_data=_full_ticker_data(config),
            _vix=None,
        )
        assert filepath.name == "2026-07-04.md"

    def test_failed_ticker_noted_note_still_written(self, vault, config):
        td = _full_ticker_data(config)
        td["AMD"] = {"snapshot": {}, "news": [], "error": "connection timeout"}
        filepath, _ = generate_briefing(
            vault_path=vault, config=config, _ticker_data=td, _vix=None,
        )
        assert filepath.exists()
        content = filepath.read_text(encoding="utf-8")
        assert "AMD" in content

    def test_no_analyze_makes_no_api_call(self, vault, config):
        with patch("weaver.briefing.requests.post") as mock_post:
            generate_briefing(
                vault_path=vault,
                config=config,
                analyze=False,
                _ticker_data=_full_ticker_data(config),
                _vix=None,
            )
        mock_post.assert_not_called()

    def test_needs_attention_tickers_returned(self, vault, config):
        td = _full_ticker_data(config)
        td["NVDA"] = _make_entry("NVDA", change_pct=0.05)
        _, needs = generate_briefing(
            vault_path=vault, config=config, _ticker_data=td, _vix=None,
        )
        assert "NVDA" in needs

    def test_note_contains_frontmatter(self, vault, config):
        filepath, _ = generate_briefing(
            vault_path=vault, config=config, _ticker_data=_full_ticker_data(config), _vix=None,
        )
        content = filepath.read_text(encoding="utf-8")
        assert "type: briefing" in content
        assert f"date: {date.today()}" in content

    def test_vix_in_note(self, vault, config):
        filepath, _ = generate_briefing(
            vault_path=vault, config=config, _ticker_data=_full_ticker_data(config), _vix=21.5,
        )
        assert "VIX: 21.5" in filepath.read_text(encoding="utf-8")

    def test_overwrite_same_day(self, vault, config):
        td = _full_ticker_data(config)
        fp1, _ = generate_briefing(vault_path=vault, config=config, _ticker_data=td, _vix=None)
        fp2, _ = generate_briefing(vault_path=vault, config=config, _ticker_data=td, _vix=None)
        assert fp1 == fp2

    def test_sector_news_appears_in_note(self, vault, config):
        td = _full_ticker_data(config)
        headline = "Chip export controls tighten"
        # Need headline in 3+ tickers; config has NVDA, AMD, SPY, QQQ
        for t in ["NVDA", "AMD", "SPY"]:
            td[t]["news"] = [{"title": headline, "publisher": "Reuters", "age": "1h ago", "url": ""}]
        filepath, _ = generate_briefing(
            vault_path=vault, config=config, _ticker_data=td, _vix=None,
        )
        assert headline in filepath.read_text(encoding="utf-8")

    def test_spy_qqq_in_market_overview_not_watchlist(self, vault, config):
        filepath, _ = generate_briefing(
            vault_path=vault, config=config, _ticker_data=_full_ticker_data(config), _vix=None,
        )
        content = filepath.read_text(encoding="utf-8")
        assert "## Market overview" in content
        # SPY/QQQ should appear in market overview section
        market_idx = content.index("## Market overview")
        watchlist_idx = content.index("## Watchlist")
        market_section = content[market_idx:watchlist_idx]
        assert "SPY" in market_section

    def test_research_note_suggests_research_command(self, vault, config):
        td = _full_ticker_data(config)
        td["NVDA"] = _make_entry("NVDA", change_pct=0.05)
        filepath, _ = generate_briefing(
            vault_path=vault, config=config, _ticker_data=td, _vix=None,
        )
        assert "wf research --ticker NVDA --analyze" in filepath.read_text(encoding="utf-8")

    def test_existing_research_shows_link(self, vault, config):
        today = date.today()
        research_dir = vault / "Trading" / "Research"
        research_dir.mkdir(parents=True, exist_ok=True)
        (research_dir / f"NVDA - {today}.md").write_text("stub", encoding="utf-8")

        td = _full_ticker_data(config)
        td["NVDA"] = _make_entry("NVDA", change_pct=0.05)
        filepath, _ = generate_briefing(
            vault_path=vault, config=config, _ticker_data=td, _vix=None,
        )
        content = filepath.read_text(encoding="utf-8")
        assert "researched today" in content
        assert "wf research --ticker NVDA" not in content


# ── TestBriefingAnalyze ───────────────────────────────────────────────────────


class TestBriefingAnalyze:
    def test_analyze_produces_synthesis_section(self, vault, config):
        filepath, _ = generate_briefing(
            vault_path=vault,
            config=config,
            analyze=True,
            _ticker_data=_full_ticker_data(config),
            _vix=None,
            _ai_synthesis="Quiet day overall. No major catalysts.",
        )
        content = filepath.read_text(encoding="utf-8")
        assert "## AI synthesis" in content
        assert "Quiet day overall" in content

    def test_no_analyze_no_synthesis_section(self, vault, config):
        filepath, _ = generate_briefing(
            vault_path=vault,
            config=config,
            analyze=False,
            _ticker_data=_full_ticker_data(config),
            _vix=None,
        )
        assert "## AI synthesis" not in filepath.read_text(encoding="utf-8")

    def test_api_failure_degrades_gracefully(self, vault, config):
        error_msg = "AI synthesis unavailable: API returned 500: Internal Server Error"
        filepath, _ = generate_briefing(
            vault_path=vault,
            config=config,
            analyze=True,
            _ticker_data=_full_ticker_data(config),
            _vix=None,
            _ai_synthesis=error_msg,
        )
        assert filepath.exists()
        assert "AI synthesis unavailable" in filepath.read_text(encoding="utf-8")

    def test_exactly_one_api_call(self, vault, config):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Market is calm today."}}]
        }
        with patch("weaver.briefing.requests.post", return_value=mock_resp) as mock_post:
            generate_briefing(
                vault_path=vault,
                config=config,
                analyze=True,
                _ticker_data=_full_ticker_data(config),
                _vix=None,
                _proposals=[],  # bypass live proposals pipeline; this test is about synthesis
                api_key="test-key",
                model="deepseek/deepseek-v4-pro",
            )
        assert mock_post.call_count == 1

    def test_synthesize_no_key_returns_error_string(self):
        result = synthesize_with_ai(
            ticker_data={},
            buckets={"needs_attention": [], "watching": [], "quiet": [], "failed": []},
            macro_tickers=frozenset({"SPY", "QQQ"}),
            vix=None,
            open_positions=[],
            open_predictions=[],
            model="deepseek/deepseek-v4-pro",
            api_key="",
        )
        assert "unavailable" in result.lower()

    def test_synthesize_network_error_returns_error_string(self):
        with patch("weaver.briefing.requests.post", side_effect=Exception("timeout")):
            result = synthesize_with_ai(
                ticker_data={},
                buckets={"needs_attention": [], "watching": [], "quiet": [], "failed": []},
                macro_tickers=frozenset({"SPY", "QQQ"}),
                vix=None,
                open_positions=[],
                open_predictions=[],
                model="deepseek/deepseek-v4-pro",
                api_key="test-key",
            )
        assert "unavailable" in result.lower()
