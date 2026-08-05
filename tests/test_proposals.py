"""Tests for weaver.proposals module."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from weaver.proposals import (
    calculate_position_size,
    format_proposal_text,
    generate_proposals,
    get_basket_exposure,
    identify_candidates,
    log_agent_prediction,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config() -> dict[str, Any]:
    return {
        "proposals": {
            "portfolio_value": 1000,
            "max_risk_per_trade_pct": 1.0,
            "max_single_position_pct": 5.0,
            "max_basket_exposure_pct": 10.0,
            "max_proposals_per_day": 2,
            "trigger_proximity_pct": 2.0,
        }
    }


def _make_position(ticker: str, entry: float, qty: float, current: float | None = None) -> dict:
    p: dict[str, Any] = {"ticker": ticker, "entry_price": entry, "quantity": qty}
    if current is not None:
        p["current_price"] = current
        p["pnl_pct"] = (current - entry) / entry
    return p


def _make_prediction(
    ticker: str,
    direction: str = "up",
    days_left: int = 10,
    trigger: float | None = None,
    current: float | None = None,
) -> dict:
    return {
        "ticker": ticker,
        "direction": direction,
        "days_left": days_left,
        "overdue": days_left < 0,
        "trigger_price": trigger,
        "current_price": current,
        "confidence": 0.65,
        "resolve_by": date.today() + timedelta(days=days_left),
    }


def _make_bucket_entry(
    ticker: str,
    change_pct: float = 0.05,
    vol_ratio: float = 1.0,
    dte: int | None = None,
) -> dict:
    return {
        "ticker": ticker,
        "change_pct": change_pct,
        "vol_ratio": vol_ratio,
        "days_to_earnings": dte,
        "snapshot": {},
        "news": [],
    }


# ── TestGetBasketExposure ─────────────────────────────────────────────────────


class TestGetBasketExposure:
    def test_empty_returns_zero(self):
        assert get_basket_exposure([]) == 0.0

    def test_sums_entry_price_times_quantity(self):
        positions = [
            _make_position("NVDA", 200.0, 2),
            _make_position("AMD", 100.0, 5),
        ]
        assert get_basket_exposure(positions) == pytest.approx(900.0)

    def test_missing_price_skipped(self):
        positions = [{"ticker": "NVDA", "quantity": 5}]
        assert get_basket_exposure(positions) == 0.0

    def test_missing_quantity_skipped(self):
        positions = [{"ticker": "NVDA", "entry_price": 200.0}]
        assert get_basket_exposure(positions) == 0.0


# ── TestCalculatePositionSize ─────────────────────────────────────────────────


class TestCalculatePositionSize:
    def test_1pct_rule_uncapped(self, config):
        # entry $100, stop $90, risk/share $10, risk_budget $10 → 1 share → $100
        # but 5% cap = $50, so it gets capped
        result = calculate_position_size(100.0, 90.0, config, basket_exposure=0.0)
        assert result is not None
        assert result["cap_applied"] is not None  # 5% cap binds
        assert result["position_value"] == pytest.approx(50.0)
        assert result["shares"] == pytest.approx(0.5)

    def test_5pct_single_position_cap_applied(self, config):
        # Large uncapped position → 5% cap (=$50) binds
        result = calculate_position_size(100.0, 99.0, config, basket_exposure=0.0)
        assert result is not None
        assert result["position_value"] <= 50.0
        assert "5% cap" in (result["cap_applied"] or "")

    def test_1pct_rule_is_binding_when_stop_is_far(self, config):
        # entry $10, stop $0 — risk/share $10, risk_budget $10 → 1 share → $10
        # 5% cap = $50 doesn't bind; 1% rule gives $10
        result = calculate_position_size(10.0, 0.01, config, basket_exposure=0.0)
        assert result is not None
        assert result["position_value"] <= 50.0
        assert result["shares"] == pytest.approx(round(10.0 / 10.0, 2), rel=0.01)

    def test_basket_remaining_limits_size(self, config):
        # basket_exposure = $95, cap = $100, remaining = $5
        result = calculate_position_size(100.0, 90.0, config, basket_exposure=95.0)
        assert result is not None
        assert result["position_value"] <= 5.0
        assert "basket" in (result["cap_applied"] or "").lower()

    def test_basket_at_capacity_returns_zero_shares(self, config):
        # basket fully used
        result = calculate_position_size(100.0, 90.0, config, basket_exposure=100.0)
        assert result is not None
        assert result["shares"] == 0.0
        assert result["cap_applied"] == "basket at capacity"

    def test_stop_above_entry_returns_none(self, config):
        assert calculate_position_size(100.0, 110.0, config, basket_exposure=0.0) is None

    def test_stop_equal_entry_returns_none(self, config):
        assert calculate_position_size(100.0, 100.0, config, basket_exposure=0.0) is None

    def test_math_string_present(self, config):
        result = calculate_position_size(100.0, 90.0, config, basket_exposure=0.0)
        assert result is not None
        assert "1% risk" in result["math"]
        assert "$" in result["math"]

    def test_risk_amount_is_shares_times_risk_per_share(self, config):
        result = calculate_position_size(50.0, 45.0, config, basket_exposure=0.0)
        assert result is not None
        expected_risk = result["shares"] * (50.0 - 45.0)
        assert result["risk_amount"] == pytest.approx(expected_risk, rel=0.01)


# ── TestIdentifyCandidates ────────────────────────────────────────────────────


class TestIdentifyCandidates:
    def test_needs_attention_ticker_qualifies(self, config):
        buckets = {"needs_attention": [_make_bucket_entry("NVDA", change_pct=0.05)]}
        result = identify_candidates(buckets, [], {}, config)
        assert any(c["ticker"] == "NVDA" for c in result)

    def test_prediction_near_trigger_qualifies(self, config):
        pred = _make_prediction("NVDA", trigger=100.0, current=101.0)  # 1% above
        result = identify_candidates({"needs_attention": []}, [pred], {"NVDA": 101.0}, config)
        assert any(c["ticker"] == "NVDA" for c in result)

    def test_prediction_far_from_trigger_excluded(self, config):
        pred = _make_prediction("NVDA", trigger=100.0, current=110.0)  # 10% away
        result = identify_candidates({"needs_attention": []}, [pred], {"NVDA": 110.0}, config)
        assert not any(c["ticker"] == "NVDA" for c in result)

    def test_capped_at_max_proposals(self, config):
        buckets = {
            "needs_attention": [
                _make_bucket_entry("NVDA"),
                _make_bucket_entry("AMD"),
                _make_bucket_entry("MSFT"),
            ]
        }
        result = identify_candidates(buckets, [], {}, config)
        assert len(result) <= 2

    def test_no_candidates_when_nothing_qualifies(self, config):
        result = identify_candidates({"needs_attention": [], "watching": [], "quiet": []}, [], {}, config)
        assert result == []

    def test_no_duplicate_tickers(self, config):
        # NVDA appears in both prediction (near trigger) and needs_attention — only one entry
        pred = _make_prediction("NVDA", trigger=100.0, current=101.0)
        buckets = {"needs_attention": [_make_bucket_entry("NVDA")]}
        result = identify_candidates(buckets, [pred], {"NVDA": 101.0}, config)
        nvda_entries = [c for c in result if c["ticker"] == "NVDA"]
        assert len(nvda_entries) == 1

    def test_prediction_has_priority_over_needs_attention(self, config):
        # With max_proposals=1, a near-trigger prediction should be picked over needs_attention
        pred = _make_prediction("AMD", trigger=100.0, current=101.0)
        buckets = {"needs_attention": [_make_bucket_entry("NVDA"), _make_bucket_entry("AMD")]}
        one_proposal_config = {**config, "proposals": {**config["proposals"], "max_proposals_per_day": 1}}
        result = identify_candidates(buckets, [pred], {"AMD": 101.0}, one_proposal_config)
        assert len(result) == 1
        assert result[0]["ticker"] == "AMD"
        assert result[0]["source"] == "prediction"


# ── TestGenerateProposals ─────────────────────────────────────────────────────


class TestGenerateProposals:
    def _call(self, config, vault, injected=None, buckets=None, positions=None, predictions=None):
        return generate_proposals(
            ticker_data={},
            buckets=buckets or {"needs_attention": []},
            open_positions=positions or [],
            open_predictions=predictions or [],
            current_prices={},
            vault_path=vault,
            config=config,
            _proposals=injected,
        )

    def test_no_candidates_returns_empty(self, config, tmp_path):
        result = self._call(config, tmp_path)
        assert result == []

    def test_injected_proposals_returned(self, config, tmp_path):
        injected = [{"ticker": "NVDA", "direction": "no_trade", "reasoning": "test"}]
        result = self._call(config, tmp_path, injected=injected)
        assert len(result) == 1
        assert result[0]["ticker"] == "NVDA"

    def test_injected_proposals_get_text_field(self, config, tmp_path):
        injected = [{"ticker": "NVDA", "direction": "no_trade", "reasoning": "test", "candidate_reason": "test", "confidence": 0.5}]
        result = self._call(config, tmp_path, injected=injected)
        assert "text" in result[0]

    def test_llm_failure_produces_no_trade_entry(self, config, tmp_path):
        buckets = {"needs_attention": [_make_bucket_entry("NVDA")]}
        with patch("weaver.proposals._call_llm", return_value=None):
            result = generate_proposals(
                ticker_data={"NVDA": {"snapshot": {}, "news": []}},
                buckets=buckets,
                open_positions=[],
                open_predictions=[],
                current_prices={"NVDA": 200.0},
                vault_path=tmp_path,
                config=config,
                api_key="test-key",
            )
        assert len(result) == 1
        assert result[0]["direction"] == "no_trade"
        assert "unavailable" in result[0]["reasoning"].lower()

    def test_basket_full_reflected_in_sizing(self, config, tmp_path):
        # $100 of exposure = exactly at cap
        positions = [_make_position("NVDA", 100.0, 1)]  # $100 = full basket
        long_response = {
            "direction": "long", "entry_price": 200.0, "stop_loss": 190.0,
            "stop_condition": "break below $190", "timeframe": "swing",
            "exit_expectation": "target $220", "reasoning": "strong setup",
            "confidence": 0.7,
        }
        buckets = {"needs_attention": [_make_bucket_entry("AMD")]}
        with patch("weaver.proposals._call_llm", return_value=long_response):
            result = generate_proposals(
                ticker_data={"AMD": {"snapshot": {"current_price": 200.0}, "news": []}},
                buckets=buckets,
                open_positions=positions,
                open_predictions=[],
                current_prices={"AMD": 200.0},
                vault_path=tmp_path,
                config=config,
                api_key="test-key",
            )
        assert len(result) == 1
        sizing = result[0].get("sizing")
        assert sizing is not None
        assert sizing["cap_applied"] == "basket at capacity"
        assert sizing["shares"] == 0.0


# ── TestLogAgentPrediction ────────────────────────────────────────────────────


class TestLogAgentPrediction:
    def _long_proposal(self, **kwargs) -> dict:
        base = {
            "direction": "long",
            "entry_price": 145.0,
            "stop_loss": 135.0,
            "stop_condition": "closes below $135",
            "timeframe": "swing",
            "exit_expectation": "target $160",
            "reasoning": "strong breakout",
            "confidence": 0.70,
            "candidate_reason": "needs attention: +5.0%",
        }
        base.update(kwargs)
        return base

    def test_writes_to_ai_subfolder(self, tmp_path):
        proposal = self._long_proposal()
        path = log_agent_prediction(tmp_path, "NVDA", proposal)
        assert path is not None
        assert "AI" in str(path)
        assert path.exists()

    def test_no_trade_not_logged(self, tmp_path):
        proposal = {"direction": "no_trade", "reasoning": "bearish"}
        result = log_agent_prediction(tmp_path, "NVDA", proposal)
        assert result is None

    def test_frontmatter_has_source_agent(self, tmp_path):
        path = log_agent_prediction(tmp_path, "NVDA", self._long_proposal())
        content = path.read_text(encoding="utf-8")
        assert "source: agent" in content

    def test_trigger_written_when_entry_is_float(self, tmp_path):
        path = log_agent_prediction(tmp_path, "NVDA", self._long_proposal(entry_price=145.0))
        content = path.read_text(encoding="utf-8")
        assert "trigger: 145.0" in content

    def test_at_market_entry_has_no_trigger(self, tmp_path):
        path = log_agent_prediction(tmp_path, "NVDA", self._long_proposal(entry_price="at market"))
        content = path.read_text(encoding="utf-8")
        assert "trigger:" not in content

    def test_unique_filename_on_same_day(self, tmp_path):
        p1 = log_agent_prediction(tmp_path, "NVDA", self._long_proposal())
        p2 = log_agent_prediction(tmp_path, "NVDA", self._long_proposal())
        assert p1 != p2


# ── TestFormatProposalText ────────────────────────────────────────────────────


class TestFormatProposalText:
    def test_long_includes_ticker_and_webull_instruction(self):
        proposal = {
            "ticker": "NVDA",
            "direction": "long",
            "candidate_reason": "needs attention: +5.0%",
            "entry_price": 200.0,
            "stop_loss": 190.0,
            "stop_condition": "breaks below $190",
            "timeframe": "swing",
            "exit_expectation": "target $220",
            "reasoning": "strong setup on earnings beat",
            "confidence": 0.70,
            "sizing": {"math": "Risk $10 ÷ $10 = 1 share → $200"},
            "concentration": "Clean slate.",
        }
        text = format_proposal_text(proposal)
        assert "NVDA" in text
        assert "LONG" in text
        assert "wf log-trade --ticker NVDA" in text
        assert "Webull" in text

    def test_no_trade_shows_no_trade(self):
        proposal = {
            "ticker": "AMD",
            "direction": "no_trade",
            "reasoning": "sector looks weak",
            "confidence": 0.8,
            "candidate_reason": "needs attention",
        }
        text = format_proposal_text(proposal)
        assert "NO TRADE" in text
        assert "sector looks weak" in text
        assert "wf log-trade" not in text

    def test_no_trade_on_llm_failure(self):
        proposal = {
            "ticker": "MSFT",
            "direction": "no_trade",
            "reasoning": "Proposal unavailable: LLM call failed.",
            "confidence": None,
            "candidate_reason": "needs attention",
        }
        text = format_proposal_text(proposal)
        assert "NO TRADE" in text
