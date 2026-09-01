"""Tests for weaver.regime and its integration with proposals/briefing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from weaver.regime import get_regime


# ── helpers ───────────────────────────────────────────────────────────────────

def _rows(n: int, price: float) -> list[dict]:
    """n identical close prices."""
    return [{"close": price}] * n


def _trending_rows(n: int, base: float, step: float) -> list[dict]:
    """n rows starting at base, incrementing by step each day."""
    return [{"close": base + i * step} for i in range(n)]


_CFG_WARN = {"regime": {"mode": "warn", "ma_period": 20}}
_CFG_HALT = {"regime": {"mode": "halt", "ma_period": 20}}
_CFG_OFF  = {"regime": {"mode": "off",  "ma_period": 20}}


# ── get_regime ────────────────────────────────────────────────────────────────

class TestGetRegime:
    def test_uptrend_when_price_above_ma(self):
        # Last 20 closes at $100, final close jumps to $110 → above MA
        rows = _rows(19, 100.0) + [{"close": 110.0}]
        result = get_regime(_CFG_WARN, _qqq_data=rows)
        assert result["regime"] == "uptrend"
        assert result["qqq_price"] == pytest.approx(110.0)
        assert result["pct_from_ma"] > 0

    def test_downtrend_when_price_below_ma(self):
        # Last 20 closes at $100, final close drops to $90 → below MA
        rows = _rows(19, 100.0) + [{"close": 90.0}]
        result = get_regime(_CFG_WARN, _qqq_data=rows)
        assert result["regime"] == "downtrend"
        assert result["qqq_price"] == pytest.approx(90.0)
        assert result["pct_from_ma"] < 0

    def test_uptrend_when_price_exactly_at_ma(self):
        rows = _rows(20, 100.0)  # MA == price
        result = get_regime(_CFG_WARN, _qqq_data=rows)
        assert result["regime"] == "uptrend"
        assert result["pct_from_ma"] == pytest.approx(0.0)

    def test_unknown_when_data_is_none(self):
        result = get_regime(_CFG_WARN, _qqq_data=None)
        # _qqq_data=None triggers live fetch, which we mock to fail
        with patch("weaver.regime._fetch_qqq_ohlcv", return_value=None):
            result = get_regime(_CFG_WARN)
        assert result["regime"] == "unknown"
        assert result["qqq_price"] is None
        assert result["qqq_ma20"] is None

    def test_unknown_when_fewer_than_ma_period_rows(self):
        rows = _rows(10, 100.0)  # only 10 rows, need 20
        result = get_regime(_CFG_WARN, _qqq_data=rows)
        assert result["regime"] == "unknown"

    def test_pct_from_ma_magnitude(self):
        # MA = 100, price = 95 → -5% below
        rows = _rows(19, 100.0) + [{"close": 95.0}]
        result = get_regime(_CFG_WARN, _qqq_data=rows)
        # MA of 19×100 + 95 = (1900+95)/20 = 99.75; price=95 → pct ≈ -4.76%
        assert result["pct_from_ma"] == pytest.approx((95.0 - 99.75) / 99.75, rel=1e-3)
        assert result["pct_from_ma"] < 0

    def test_ma_uses_last_n_rows_only(self):
        # 25 rows: first 5 are $200 (should be ignored), last 20 are $100
        rows = _rows(5, 200.0) + _rows(19, 100.0) + [{"close": 90.0}]
        result = get_regime(_CFG_WARN, _qqq_data=rows)
        # MA should be ~99.5 (not skewed by the $200 rows)
        assert result["qqq_ma20"] == pytest.approx(99.5, rel=1e-3)


# ── proposals integration ─────────────────────────────────────────────────────

class TestProposalsRegime:
    def _make_downtrend(self) -> dict:
        return {"regime": "downtrend", "qqq_price": 440.0, "qqq_ma20": 455.0, "pct_from_ma": -0.033}

    def _make_uptrend(self) -> dict:
        return {"regime": "uptrend", "qqq_price": 465.0, "qqq_ma20": 455.0, "pct_from_ma": 0.022}

    def test_halt_mode_downtrend_returns_empty(self, tmp_path):
        from weaver.proposals import generate_proposals
        result = generate_proposals(
            ticker_data={}, buckets={}, open_positions=[], open_predictions=[],
            current_prices={}, vault_path=tmp_path,
            config=_CFG_HALT,
            _regime=self._make_downtrend(),
        )
        assert result == []

    def test_halt_mode_uptrend_does_not_block(self, tmp_path, capsys):
        from weaver.proposals import generate_proposals
        # With uptrend, halt mode should NOT return early — candidates are empty so still []
        # but the early-return path was not taken (no stderr message)
        result = generate_proposals(
            ticker_data={}, buckets={}, open_positions=[], open_predictions=[],
            current_prices={}, vault_path=tmp_path,
            config=_CFG_HALT,
            _regime=self._make_uptrend(),
        )
        err = capsys.readouterr().err
        assert "halted" not in err

    def test_warn_mode_downtrend_proceeds_with_warning_in_prompt(self, tmp_path):
        from weaver.proposals import generate_proposals
        captured_prompts = []

        def mock_call_llm(prompt, api_key, model, timeout, _system_prompt=None):
            captured_prompts.append(_system_prompt or "")
            return None  # no trade result

        with patch("weaver.proposals._call_llm", side_effect=mock_call_llm):
            with patch("weaver.proposals._fetch_ohlcv", return_value=None):
                generate_proposals(
                    ticker_data={"NVDA": {"snapshot": {"current_price": 900}, "news": []}},
                    buckets={"needs_attention": [{"ticker": "NVDA", "snapshot": {"current_price": 900}, "news": [], "change_pct": 0.05, "vol_ratio": 2.0, "days_to_earnings": None}]},
                    open_positions=[], open_predictions=[],
                    current_prices={"NVDA": 900.0},
                    vault_path=tmp_path,
                    config={**_CFG_WARN, "proposals": {"portfolio_value": 1000, "max_risk_per_trade_pct": 1.0, "max_single_position_pct": 5.0, "max_basket_exposure_pct": 10.0, "max_proposals_per_day": 2, "trigger_proximity_pct": 2.0}},
                    _regime=self._make_downtrend(),
                )

        assert len(captured_prompts) >= 1
        assert "MARKET REGIME ALERT" in captured_prompts[0]
        assert "downtrend" in captured_prompts[0].lower()

    def test_off_mode_skips_regime_check(self, tmp_path, capsys):
        from weaver.proposals import generate_proposals
        with patch("weaver.regime.get_regime") as mock_get:
            generate_proposals(
                ticker_data={}, buckets={}, open_positions=[], open_predictions=[],
                current_prices={}, vault_path=tmp_path,
                config=_CFG_OFF,
            )
        mock_get.assert_not_called()


# ── briefing integration ──────────────────────────────────────────────────────

class TestBriefingRegime:
    def _minimal_ticker_data(self) -> dict:
        return {
            "SPY": {"snapshot": {"current_price": 550.0, "day_change_pct": 0.005}, "news": [], "error": None},
            "QQQ": {"snapshot": {"current_price": 440.0, "day_change_pct": -0.003}, "news": [], "error": None},
        }

    def _make_config(self, mode: str = "warn") -> dict:
        return {
            "regime": {"mode": mode, "ma_period": 20},
            "briefing": {},
            "proposals": {"portfolio_value": 1000, "max_risk_per_trade_pct": 1.0, "max_single_position_pct": 5.0, "max_basket_exposure_pct": 10.0, "max_proposals_per_day": 2, "trigger_proximity_pct": 2.0},
        }

    def test_regime_line_always_in_macro_section(self, tmp_path):
        from weaver.briefing import build_briefing_note
        from datetime import date
        regime = {"regime": "uptrend", "qqq_price": 465.0, "qqq_ma20": 455.0, "pct_from_ma": 0.022}
        note = build_briefing_note(
            briefing_date=date(2026, 8, 4),
            ticker_data=self._minimal_ticker_data(),
            macro_tickers=frozenset({"SPY", "QQQ"}),
            vix=15.0,
            sector_headlines=[],
            buckets={"needs_attention": [], "watching": [], "quiet": [], "failed": []},
            open_positions=[], open_predictions=[],
            synthesis=None, vault_path=tmp_path,
            regime_state=regime, regime_mode="warn",
        )
        assert "QQQ Regime: UPTREND" in note
        assert "above 20MA" in note

    def test_downtrend_block_rendered_in_warn_mode(self, tmp_path):
        from weaver.briefing import build_briefing_note
        from datetime import date
        regime = {"regime": "downtrend", "qqq_price": 440.0, "qqq_ma20": 455.0, "pct_from_ma": -0.033}
        note = build_briefing_note(
            briefing_date=date(2026, 8, 4),
            ticker_data=self._minimal_ticker_data(),
            macro_tickers=frozenset({"SPY", "QQQ"}),
            vix=22.0,
            sector_headlines=[],
            buckets={"needs_attention": [], "watching": [], "quiet": [], "failed": []},
            open_positions=[], open_predictions=[],
            synthesis=None, vault_path=tmp_path,
            regime_state=regime, regime_mode="warn",
        )
        assert "## Market Regime" in note
        assert "DOWNTREND" in note
        assert "Proposals warned" in note

    def test_downtrend_block_suppressed_in_off_mode(self, tmp_path):
        from weaver.briefing import build_briefing_note
        from datetime import date
        regime = {"regime": "downtrend", "qqq_price": 440.0, "qqq_ma20": 455.0, "pct_from_ma": -0.033}
        note = build_briefing_note(
            briefing_date=date(2026, 8, 4),
            ticker_data=self._minimal_ticker_data(),
            macro_tickers=frozenset({"SPY", "QQQ"}),
            vix=22.0,
            sector_headlines=[],
            buckets={"needs_attention": [], "watching": [], "quiet": [], "failed": []},
            open_positions=[], open_predictions=[],
            synthesis=None, vault_path=tmp_path,
            regime_state=regime, regime_mode="off",
        )
        assert "## Market Regime" not in note

    def test_uptrend_block_suppressed(self, tmp_path):
        from weaver.briefing import build_briefing_note
        from datetime import date
        regime = {"regime": "uptrend", "qqq_price": 465.0, "qqq_ma20": 455.0, "pct_from_ma": 0.022}
        note = build_briefing_note(
            briefing_date=date(2026, 8, 4),
            ticker_data=self._minimal_ticker_data(),
            macro_tickers=frozenset({"SPY", "QQQ"}),
            vix=15.0,
            sector_headlines=[],
            buckets={"needs_attention": [], "watching": [], "quiet": [], "failed": []},
            open_positions=[], open_predictions=[],
            synthesis=None, vault_path=tmp_path,
            regime_state=regime, regime_mode="warn",
        )
        assert "## Market Regime" not in note

    def test_halt_mode_downtrend_block_says_halted(self, tmp_path):
        from weaver.briefing import build_briefing_note
        from datetime import date
        regime = {"regime": "downtrend", "qqq_price": 440.0, "qqq_ma20": 455.0, "pct_from_ma": -0.033}
        note = build_briefing_note(
            briefing_date=date(2026, 8, 4),
            ticker_data=self._minimal_ticker_data(),
            macro_tickers=frozenset({"SPY", "QQQ"}),
            vix=22.0,
            sector_headlines=[],
            buckets={"needs_attention": [], "watching": [], "quiet": [], "failed": []},
            open_positions=[], open_predictions=[],
            synthesis=None, vault_path=tmp_path,
            regime_state=regime, regime_mode="halt",
        )
        assert "Proposals halted" in note
        assert "Proposals warned" not in note
