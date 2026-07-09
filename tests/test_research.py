"""Tests for pre-trade research (weaver.research).

Network calls (fetch_snapshot, fetch_news) are bypassed via injection.
fetch_prior_views and note generation are tested with tmp_path.
OpenRouter API calls are mocked — no real network or cost in CI.
"""

import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from weaver.journal import log_trade
from weaver.predictions import log_prediction
from zoneinfo import ZoneInfo

from weaver.research import (
    _fmt_money,
    _fmt_num,
    _format_age,
    _is_relevant_to_ticker,
    _parse_ai_response,
    _parse_news_item,
    _project_intraday_volume,
    analyze_with_ai,
    fetch_news,
    fetch_prior_views,
    generate_research_note,
)

_ET = ZoneInfo("America/New_York")

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


# Captured from a live yfinance call on 2026-07-01 — reflects the current
# nested-content structure where all fields live under item["content"].
_REAL_NEWS_ITEM = {
    "id": "af0da905-cc0e-4368-934e-cc9ff4234458",
    "content": {
        "id": "af0da905-cc0e-4368-934e-cc9ff4234458",
        "contentType": "VIDEO",
        "title": "What's next for AI and Big Tech in the second half of 2026?",
        "pubDate": "2026-07-01T14:19:12Z",
        "provider": {
            "displayName": "Yahoo Finance Video",
            "url": "https://finance.yahoo.com/",
            "sourceId": "video.yahoofinance.com",
        },
        "canonicalUrl": {
            "url": "https://finance.yahoo.com/video/whats-next-for-ai-141912104.html",
        },
        "clickThroughUrl": {
            "url": "https://finance.yahoo.com/video/whats-next-for-ai-141912104.html",
        },
    },
}

_LEGACY_NEWS_ITEM = {
    "title": "NVDA hits record high",
    "publisher": "Reuters",
    "link": "https://reuters.com/nvda-record",
    "providerPublishTime": 1751385600,
}


class TestParseNewsItem:
    def test_extracts_title_from_content(self) -> None:
        parsed = _parse_news_item(_REAL_NEWS_ITEM)
        assert parsed["title"] == "What's next for AI and Big Tech in the second half of 2026?"

    def test_extracts_publisher_from_provider(self) -> None:
        parsed = _parse_news_item(_REAL_NEWS_ITEM)
        assert parsed["publisher"] == "Yahoo Finance Video"

    def test_extracts_url_from_canonical_url(self) -> None:
        parsed = _parse_news_item(_REAL_NEWS_ITEM)
        assert parsed["url"] == "https://finance.yahoo.com/video/whats-next-for-ai-141912104.html"

    def test_parses_iso_pub_date_to_age(self) -> None:
        parsed = _parse_news_item(_REAL_NEWS_ITEM)
        # pubDate is in the past relative to today (2026-07-01), so age is non-None
        assert parsed["age"] is not None
        assert "ago" in parsed["age"]

    def test_legacy_flat_structure_still_parses(self) -> None:
        parsed = _parse_news_item(_LEGACY_NEWS_ITEM)
        assert parsed["title"] == "NVDA hits record high"
        assert parsed["publisher"] == "Reuters"
        assert parsed["url"] == "https://reuters.com/nvda-record"

    def test_legacy_unix_timestamp_produces_age(self) -> None:
        parsed = _parse_news_item(_LEGACY_NEWS_ITEM)
        assert parsed["age"] is not None
        assert "ago" in parsed["age"]

    def test_empty_item_returns_safe_defaults(self) -> None:
        parsed = _parse_news_item({})
        assert parsed["title"] == ""
        assert parsed["publisher"] == ""
        assert parsed["url"] == ""
        assert parsed["age"] is None

    def test_prefers_canonical_url_over_click_through(self) -> None:
        item = {
            "content": {
                "title": "Test",
                "canonicalUrl": {"url": "https://canonical.example.com"},
                "clickThroughUrl": {"url": "https://clickthrough.example.com"},
            }
        }
        parsed = _parse_news_item(item)
        assert parsed["url"] == "https://canonical.example.com"


# ── intraday volume projection ────────────────────────────────────────────────

_SNAPSHOT_WITH_PROJECTION = {
    **_SNAPSHOT,
    "volume_projected": 52_000_000,
    "volume_ratio_projected": 1.37,
    "snapshot_time_et": "11:00 ET",
}


class TestProjectIntradayVolume:
    def _et(self, year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
        return datetime(year, month, day, hour, minute, tzinfo=_ET)

    def test_projects_at_11am_weekday(self) -> None:
        # Wednesday 11:00 AM ET — 90 min of 390 elapsed = 23.08%
        now = self._et(2026, 7, 8, 11, 0)
        result = _project_intraday_volume(12_000_000, now)
        assert result is not None
        # 12M / (90/390) ≈ 52M
        assert 50_000_000 < result < 55_000_000

    def test_projects_at_2pm(self) -> None:
        # 2:00 PM ET — 270 min of 390 elapsed = 69.2%
        now = self._et(2026, 7, 8, 14, 0)
        result = _project_intraday_volume(30_000_000, now)
        assert result is not None
        # 30M / (270/390) ≈ 43.3M
        assert 42_000_000 < result < 45_000_000

    def test_returns_none_before_market_open(self) -> None:
        now = self._et(2026, 7, 8, 9, 29)
        assert _project_intraday_volume(1_000_000, now) is None

    def test_returns_none_at_market_open(self) -> None:
        # elapsed == 0 → division by zero guard
        now = self._et(2026, 7, 8, 9, 30)
        assert _project_intraday_volume(1_000_000, now) is None

    def test_returns_none_at_or_after_close(self) -> None:
        now = self._et(2026, 7, 8, 16, 0)
        assert _project_intraday_volume(40_000_000, now) is None

    def test_returns_none_on_saturday(self) -> None:
        now = self._et(2026, 7, 11, 11, 0)  # Saturday
        assert _project_intraday_volume(1_000_000, now) is None

    def test_returns_none_on_sunday(self) -> None:
        now = self._et(2026, 7, 12, 11, 0)  # Sunday
        assert _project_intraday_volume(1_000_000, now) is None

    def test_projected_volume_in_note(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY,
            _snapshot=_SNAPSHOT_WITH_PROJECTION, _news=[],
        )
        content = fp.read_text()
        assert "projected" in content
        assert "11:00 ET" in content

    def test_projected_volume_in_ai_prompt(self) -> None:
        from weaver.research import _build_analysis_prompt
        prompt = _build_analysis_prompt(
            "NVDA", _SNAPSHOT_WITH_PROJECTION, [],
            {"trades": [], "predictions": [], "prior_research": []},
        )
        assert "projected" in prompt
        assert "11:00 ET" in prompt

    def test_no_projection_when_market_closed(self, tmp_path: Path) -> None:
        # Snapshot without volume_projected (market closed at fetch time)
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY,
            _snapshot=_SNAPSHOT, _news=[],
        )
        content = fp.read_text()
        # Should still show volume, just without projection
        assert "Volume" in content
        assert "projected" not in content


# ── news relevance filtering ──────────────────────────────────────────────────

class TestIsRelevantToTicker:
    def test_ticker_symbol_in_title_passes(self) -> None:
        assert _is_relevant_to_ticker("NVDA earnings beat estimates", "NVDA") is True

    def test_ticker_lowercase_in_title_passes(self) -> None:
        assert _is_relevant_to_ticker("Why nvda is the AI trade", "NVDA") is True

    def test_company_name_word_in_title_passes(self) -> None:
        assert _is_relevant_to_ticker("NVIDIA announces new Blackwell chip", "NVDA", "NVIDIA Corp") is True

    def test_multi_word_company_name_partial_match(self) -> None:
        assert _is_relevant_to_ticker("Microsoft expands Azure AI", "MSFT", "Microsoft Corp") is True

    def test_unrelated_headline_rejected(self) -> None:
        assert _is_relevant_to_ticker("McCormick raises dividend outlook", "NVDA", "NVIDIA Corp") is False

    def test_unrelated_headline_no_company_name(self) -> None:
        assert _is_relevant_to_ticker("Netflix beats subscriber forecast", "NVDA") is False

    def test_empty_title_rejected(self) -> None:
        assert _is_relevant_to_ticker("", "NVDA", "NVIDIA Corp") is False

    def test_ticker_word_boundary_not_partial_match(self) -> None:
        # "METADATA" should not match ticker "META"
        assert _is_relevant_to_ticker("Metadata standards evolve", "META", "Meta Platforms") is False

    def test_ticker_possessive_still_matches(self) -> None:
        # "NVDA's" should match — apostrophe is a word boundary
        assert _is_relevant_to_ticker("NVDA's options activity spikes", "NVDA") is True

    def test_generic_company_words_not_matched(self) -> None:
        # "Corp" and "Trust" are in the generic-words list — shouldn't cause false positives
        assert _is_relevant_to_ticker("Corporate trust issues in banking", "NVDA", "NVIDIA Corp") is False

    def test_general_ai_headline_without_ticker_rejected(self) -> None:
        # Reproduces the real-world noise case: general AI article in NVDA feed
        assert _is_relevant_to_ticker(
            "What's next for AI and Big Tech in the second half of 2026?",
            "NVDA", "NVIDIA Corp",
        ) is False

    def test_rivian_headline_in_nvda_feed_rejected(self) -> None:
        assert _is_relevant_to_ticker("Rivian stock slides after recall", "NVDA", "NVIDIA Corp") is False


class TestFetchNewsFiltering:
    """fetch_news should return only ticker-relevant articles."""

    def _make_raw_item(self, title: str) -> dict:
        return {"content": {"title": title, "pubDate": "2026-07-07T12:00:00Z",
                             "provider": {"displayName": "Reuters"},
                             "canonicalUrl": {"url": "https://example.com"}}}

    def test_unrelated_articles_filtered_out(self) -> None:
        raw = [
            self._make_raw_item("NVDA rallies on strong earnings"),
            self._make_raw_item("McCormick raises dividend outlook"),
            self._make_raw_item("Netflix beats subscriber forecast"),
            self._make_raw_item("NVIDIA unveils next-gen data-center GPU"),
        ]
        mock_ticker = MagicMock()
        mock_ticker.news = raw
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = fetch_news("NVDA", company_name="NVIDIA Corp")
        titles = [item["title"] for item in result]
        assert "NVDA rallies on strong earnings" in titles
        assert "NVIDIA unveils next-gen data-center GPU" in titles
        assert "McCormick raises dividend outlook" not in titles
        assert "Netflix beats subscriber forecast" not in titles

    def test_limit_applied_after_filtering(self) -> None:
        raw = [self._make_raw_item(f"NVDA item {i}") for i in range(20)]
        mock_ticker = MagicMock()
        mock_ticker.news = raw
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = fetch_news("NVDA", limit=5, company_name="NVIDIA Corp")
        assert len(result) == 5

    def test_all_filtered_returns_empty_list(self) -> None:
        raw = [
            self._make_raw_item("McCormick raises dividend"),
            self._make_raw_item("Rivian recall announced"),
        ]
        mock_ticker = MagicMock()
        mock_ticker.news = raw
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = fetch_news("NVDA", company_name="NVIDIA Corp")
        assert result == []


# ── AI analysis ───────────────────────────────────────────────────────────────

_GOOD_AI = {
    "setup": "NVDA is trading near the top of its 52-week range with elevated valuation.",
    "bull_case": "AI infrastructure spending remains robust; NVDA holds pricing power in data-center GPUs.",
    "bear_case": "Valuation is stretched and any guidance miss into earnings could trigger a sharp sell-off.",
    "key_risks": "Next earnings on 2026-08-15 is the critical binary. Consensus expects continued GPU demand strength.",
    "direction": "up",
    "confidence": 0.6,
    "reasoning": "Near-term AI tailwinds support the bull case, but the setup into earnings is high-risk.",
}


class TestAnalyzeWithAI:
    def test_happy_path_fills_sections(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY,
            _snapshot=_SNAPSHOT, _news=[],
            analyze=True, _ai_analysis=_GOOD_AI,
        )
        content = fp.read_text()
        assert "AI infrastructure spending" in content
        assert "Valuation is stretched" in content
        assert "Next earnings on 2026-08-15" in content

    def test_suggested_call_section_present(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY,
            _snapshot=_SNAPSHOT, _news=[],
            analyze=True, _ai_analysis=_GOOD_AI,
        )
        content = fp.read_text()
        assert "## AI's suggested call" in content
        assert "Direction: up" in content
        assert "Confidence: 0.60" in content
        assert "Near-term AI tailwinds" in content

    def test_my_take_line_present_on_success(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY,
            _snapshot=_SNAPSHOT, _news=[],
            analyze=True, _ai_analysis=_GOOD_AI,
        )
        assert "**My take:**" in fp.read_text()

    def test_log_prediction_template_prefilled(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY,
            _snapshot=_SNAPSHOT, _news=[],
            analyze=True, _ai_analysis=_GOOD_AI,
        )
        content = fp.read_text()
        assert "--direction up" in content
        assert "--confidence 0.60" in content

    def test_api_error_writes_unavailable_in_sections(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY,
            _snapshot=_SNAPSHOT, _news=[],
            analyze=True, _ai_analysis={"error": "network timeout"},
        )
        content = fp.read_text()
        assert "AI analysis unavailable" in content
        assert "network timeout" in content
        assert "## Setup" in content
        assert "## Bull case" in content

    def test_api_error_still_has_my_take_line(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY,
            _snapshot=_SNAPSHOT, _news=[],
            analyze=True, _ai_analysis={"error": "API 429"},
        )
        assert "**My take:**" in fp.read_text()

    def test_api_error_no_suggested_call_section(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY,
            _snapshot=_SNAPSHOT, _news=[],
            analyze=True, _ai_analysis={"error": "rate limited"},
        )
        assert "## AI's suggested call" not in fp.read_text()

    def test_analyze_false_blank_sections(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY, _snapshot=_SNAPSHOT, _news=[],
        )
        content = fp.read_text()
        assert "## Setup" in content
        assert "## AI's suggested call" not in content
        assert "**My take:**" not in content
        assert "AI analysis unavailable" not in content

    def test_analyze_false_preserves_existing_my_call(self, tmp_path: Path) -> None:
        fp = generate_research_note(
            tmp_path, "NVDA", TODAY, _snapshot=_SNAPSHOT, _news=[],
        )
        content = fp.read_text()
        assert "Complete your analysis above" in content
        assert "wf log-prediction" in content

    # ── _parse_ai_response ────────────────────────────────────────────────────

    def test_parse_clean_json(self) -> None:
        result = _parse_ai_response(json.dumps(_GOOD_AI))
        assert result["direction"] == "up"
        assert result["confidence"] == 0.6
        assert "error" not in result

    def test_parse_json_in_markdown_fence(self) -> None:
        raw = f"Sure!\n```json\n{json.dumps(_GOOD_AI)}\n```"
        result = _parse_ai_response(raw)
        assert result["direction"] == "up"
        assert "error" not in result

    def test_parse_json_embedded_in_prose(self) -> None:
        raw = f"Here is my analysis: {json.dumps(_GOOD_AI)} Hope that helps."
        result = _parse_ai_response(raw)
        assert result["direction"] == "up"
        assert "error" not in result

    def test_parse_bad_input_returns_error(self) -> None:
        result = _parse_ai_response("This is not JSON at all.")
        assert "error" in result

    def test_parse_missing_keys_returns_error(self) -> None:
        result = _parse_ai_response(json.dumps({"direction": "up", "confidence": 0.5}))
        assert "error" in result
        assert "missing keys" in result["error"]

    def test_parse_invalid_direction_returns_error(self) -> None:
        bad = {**_GOOD_AI, "direction": "sideways"}
        result = _parse_ai_response(json.dumps(bad))
        assert "error" in result
        assert "direction" in result["error"]

    def test_parse_confidence_clamped_to_range(self) -> None:
        over = {**_GOOD_AI, "confidence": 1.5}
        result = _parse_ai_response(json.dumps(over))
        assert result["confidence"] == 1.0

    # ── analyze_with_ai (mocked) ──────────────────────────────────────────────

    def test_network_error_returns_error_dict(self) -> None:
        import requests as req_module
        with patch("weaver.research.requests.post",
                   side_effect=req_module.exceptions.ConnectionError("refused")):
            result = analyze_with_ai("NVDA", {}, [], {}, "deepseek/deepseek-v4-pro", "fake-key")
        assert "error" in result
        assert "network" in result["error"].lower()

    def test_non_200_status_returns_error_dict(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        with patch("weaver.research.requests.post", return_value=mock_resp):
            result = analyze_with_ai("NVDA", {}, [], {}, "deepseek/deepseek-v4-pro", "fake-key")
        assert "error" in result
        assert "429" in result["error"]

    def test_bad_json_in_response_returns_error_dict(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Sorry, cannot help with that."}}]
        }
        with patch("weaver.research.requests.post", return_value=mock_resp):
            result = analyze_with_ai("NVDA", {}, [], {}, "deepseek/deepseek-v4-pro", "fake-key")
        assert "error" in result

    def test_missing_api_key_returns_error_dict(self) -> None:
        result = analyze_with_ai("NVDA", {}, [], {}, "deepseek/deepseek-v4-pro", api_key="")
        assert "error" in result
        assert "OPENROUTER_API_KEY" in result["error"]

    def test_successful_api_call_returns_parsed_analysis(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps(_GOOD_AI)}}]
        }
        with patch("weaver.research.requests.post", return_value=mock_resp):
            result = analyze_with_ai("NVDA", _SNAPSHOT, [], {}, "deepseek/deepseek-v4-pro", "fake-key")
        assert result["direction"] == "up"
        assert result["confidence"] == 0.6
        assert "error" not in result
