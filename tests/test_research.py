"""Tests for pre-trade research (weaver.research).

Network calls (fetch_snapshot, fetch_news) are bypassed via injection.
fetch_prior_views and note generation are tested with tmp_path.
"""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from weaver.journal import log_trade
from weaver.predictions import log_prediction
from weaver.research import (
    _fmt_money,
    _fmt_num,
    _format_age,
    fetch_prior_views,
    generate_research_note,
)

TODAY = date(2026, 7, 1)
FUTURE = date(2099, 1, 1)

_SNAPSHOT = {
    "name": "NVIDIA Corp",
    "current_price": 130.00,
    "prev_close": 125.00,
    "day_change_pct": 0.04,
    "fifty_two_week_high": 153.00,
    "fifty_two_week_low": 86.00,
    "volume": 45_000_000,
    "avg_volume": 38_000_000,
    "volume_ratio": 1.18,
    "market_cap": 3_200_000_000_000,
    "pe_trailing": 38.2,
    "pe_forward": 28.5,
    "next_earnings": "2026-08-15",
    "sector": "Technology",
    "industry": "Semiconductors",
    "description": "NVIDIA Corporation is a computing infrastructure company.",
}


class TestGenerateResearchNote:
    def test_creates_file_in_research_dir(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY, _snapshot=_SNAPSHOT, _news=[]
        )
        assert fp.exists()
        assert fp.parent == tmp_path / "Trading" / "Research"

    def test_filename_format(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY, _snapshot=_SNAPSHOT, _news=[]
        )
        assert fp.name == "NVDA - 2026-07-01.md"

    def test_ticker_normalized_to_uppercase(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "nvda", TODAY, _snapshot=_SNAPSHOT, _news=[]
        )
        assert fp.name == "NVDA - 2026-07-01.md"
        assert "ticker: NVDA" in fp.read_text()

    def test_research_dir_created_automatically(self, tmp_path: Path) -> None:
        generate_research_note(
            tmp_path, "NVDA", TODAY, _snapshot=_SNAPSHOT, _news=[]
        )
        assert (tmp_path / "Trading" / "Research").is_dir()

    def test_frontmatter_fields_present(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY, _snapshot=_SNAPSHOT, _news=[]
        )
        content = fp.read_text()
        assert "ticker: NVDA" in content
        assert "date: 2026-07-01" in content
        assert "type: research" in content
        assert "price: 130.00" in content

    def test_snapshot_fields_in_body(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY, _snapshot=_SNAPSHOT, _news=[]
        )
        content = fp.read_text()
        assert "$130.00" in content
        assert "$86.00" in content   # 52w low
        assert "$153.00" in content  # 52w high
        assert "$3.2T" in content    # market cap
        assert "38.2" in content     # trailing P/E

    def test_earnings_date_in_snapshot(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY, _snapshot=_SNAPSHOT, _news=[]
        )
        assert "2026-08-15" in fp.read_text()

    def test_news_items_appear_in_body(self, tmp_path: Path) -> None:
        news = [
            {"title": "NVDA beats earnings", "publisher": "Reuters",
             "age": "2h ago", "url": "https://example.com/1"},
            {"title": "AI spending surge", "publisher": "Bloomberg",
             "age": "5h ago", "url": ""},
        ]
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY, _snapshot=_SNAPSHOT, _news=news
        )
        content = fp.read_text()
        assert "NVDA beats earnings" in content
        assert "2h ago" in content
        assert "AI spending surge" in content

    def test_no_news_shows_placeholder(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY, _snapshot={}, _news=[]
        )
        assert "No news available" in fp.read_text()

    def test_analysis_sections_present(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY, _snapshot=_SNAPSHOT, _news=[]
        )
        content = fp.read_text()
        for section in ("## Setup", "## Bull case", "## Bear case", "## Key risks"):
            assert section in content

    def test_my_call_section_present(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY, _snapshot=_SNAPSHOT, _news=[]
        )
        content = fp.read_text()
        assert "## My call" in content
        assert "wf log-prediction" in content
        assert "--ticker NVDA" in content

    def test_error_snapshot_handled_gracefully(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY,
            _snapshot={"error": "Rate limited"},
            _news=[],
        )
        content = fp.read_text()
        assert "Rate limited" in content
        assert "## My call" in content   # note still completes


class TestFetchPriorViews:
    def test_empty_vault_returns_empty_lists(self, tmp_path: Path) -> None:
        prior = fetch_prior_views(tmp_path, "NVDA")
        assert prior["trades"] == []
        assert prior["predictions"] == []
        assert prior["prior_research"] == []

    def test_nonexistent_subdirs_handled_gracefully(self, tmp_path: Path) -> None:
        # No Trading/ folder at all
        prior = fetch_prior_views(tmp_path / "missing", "NVDA")
        assert prior["trades"] == []
        assert prior["predictions"] == []
        assert prior["prior_research"] == []

    def test_finds_matching_journal_entry(self, tmp_path: Path) -> None:
        log_trade(
            vault_path=tmp_path, ticker="NVDA", action="buy",
            quantity=10, price=130.0,
            reason="Breakout on volume", stop_condition="Below 125",
            time_horizon="swing", trade_date=TODAY,
        )
        prior = fetch_prior_views(tmp_path, "NVDA")
        assert len(prior["trades"]) == 1
        assert prior["trades"][0]["action"] == "buy"
        assert prior["trades"][0]["reason"] == "Breakout on volume"

    def test_ignores_different_ticker_in_journal(self, tmp_path: Path) -> None:
        log_trade(
            vault_path=tmp_path, ticker="AMD", action="buy",
            quantity=5, price=120.0,
            reason="AMD trade", stop_condition="Below 115",
            time_horizon="day", trade_date=TODAY,
        )
        prior = fetch_prior_views(tmp_path, "NVDA")
        assert prior["trades"] == []

    def test_finds_matching_prediction(self, tmp_path: Path) -> None:
        log_prediction(
            vault_path=tmp_path, ticker="NVDA", direction="up",
            timeframe="2 weeks", confidence=0.7,
            reasoning="Earnings catalyst", resolve_by=FUTURE,
            prediction_date=TODAY,
        )
        prior = fetch_prior_views(tmp_path, "NVDA")
        assert len(prior["predictions"]) == 1
        assert prior["predictions"][0]["direction"] == "up"

    def test_ignores_different_ticker_in_predictions(self, tmp_path: Path) -> None:
        log_prediction(
            vault_path=tmp_path, ticker="AMD", direction="down",
            timeframe="1 week", confidence=0.5,
            reasoning="Overbought", resolve_by=FUTURE,
            prediction_date=TODAY,
        )
        prior = fetch_prior_views(tmp_path, "NVDA")
        assert prior["predictions"] == []

    def test_ticker_matching_is_case_insensitive(self, tmp_path: Path) -> None:
        log_trade(
            vault_path=tmp_path, ticker="NVDA", action="buy",
            quantity=10, price=130.0,
            reason="Test", stop_condition="Below 125",
            time_horizon="swing", trade_date=TODAY,
        )
        prior_lower = fetch_prior_views(tmp_path, "nvda")
        prior_upper = fetch_prior_views(tmp_path, "NVDA")
        assert len(prior_lower["trades"]) == 1
        assert len(prior_upper["trades"]) == 1

    def test_finds_prior_research_notes(self, tmp_path: Path) -> None:
        research_dir = tmp_path / "Trading" / "Research"
        research_dir.mkdir(parents=True)
        (research_dir / "NVDA - 2026-06-15.md").write_text("---\nticker: NVDA\n---\n")
        prior = fetch_prior_views(tmp_path, "NVDA")
        assert "NVDA - 2026-06-15" in prior["prior_research"]

    def test_prior_research_excludes_other_tickers(self, tmp_path: Path) -> None:
        research_dir = tmp_path / "Trading" / "Research"
        research_dir.mkdir(parents=True)
        (research_dir / "AMD - 2026-06-15.md").write_text("---\nticker: AMD\n---\n")
        prior = fetch_prior_views(tmp_path, "NVDA")
        assert prior["prior_research"] == []

    def test_multiple_entries_all_returned(self, tmp_path: Path) -> None:
        for action, tdate in [("buy", date(2026, 6, 1)), ("sell", date(2026, 6, 15))]:
            log_trade(
                vault_path=tmp_path, ticker="NVDA", action=action,
                quantity=10, price=130.0,
                reason=f"{action} reason", stop_condition="Below 125",
                time_horizon="swing", trade_date=tdate,
            )
        prior = fetch_prior_views(tmp_path, "NVDA")
        assert len(prior["trades"]) == 2


class TestFormatHelpers:
    def test_fmt_num_thousands(self) -> None:
        assert _fmt_num(1_500) == "1.5K"

    def test_fmt_num_millions(self) -> None:
        assert _fmt_num(1_200_000) == "1.2M"

    def test_fmt_num_billions(self) -> None:
        assert _fmt_num(3_400_000_000) == "3.4B"

    def test_fmt_num_trillions(self) -> None:
        assert _fmt_num(3_200_000_000_000) == "3.2T"

    def test_fmt_money_prefixes_dollar(self) -> None:
        assert _fmt_money(1_200_000) == "$1.2M"

    def test_format_age_minutes(self) -> None:
        now = datetime.now(tz=timezone.utc)
        from datetime import timedelta
        dt = now - timedelta(minutes=25)
        assert _format_age(dt) == "25m ago"

    def test_format_age_hours(self) -> None:
        from datetime import timedelta
        now = datetime.now(tz=timezone.utc)
        dt = now - timedelta(hours=3)
        assert _format_age(dt) == "3h ago"

    def test_format_age_days(self) -> None:
        from datetime import timedelta
        now = datetime.now(tz=timezone.utc)
        dt = now - timedelta(days=2)
        assert _format_age(dt) == "2d ago"
