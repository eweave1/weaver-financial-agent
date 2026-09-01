"""Tests for weaver.exits — exit condition scanner."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from weaver.exits import deliver_alerts, format_exit_message, scan_exits


# ── helpers ───────────────────────────────────────────────────────────────────

def _pos(ticker: str, entry: float, stop: float | None, qty: float = 10.0) -> dict:
    return {
        "ticker": ticker,
        "entry_price": entry,
        "stop_price": stop,
        "quantity": qty,
        "entry_date": "2026-01-01",
        "time_horizon": "swing",
        "file_stem": f"2026-01-01 {ticker} buy",
        "status": "open",
    }


_CFG = {"exits": {"approaching_stop_pct": 2.0}}


# ── scan_exits ────────────────────────────────────────────────────────────────

class TestScanExits:
    def test_stop_breached_at_stop(self):
        alerts = scan_exits(
            Path("/fake"), _CFG,
            _positions=[_pos("NVDA", 100.0, 90.0)],
            _prices={"NVDA": 90.0},
        )
        assert len(alerts) == 1
        assert alerts[0]["condition"] == "stop_breached"
        assert alerts[0]["ticker"] == "NVDA"

    def test_stop_breached_below_stop(self):
        alerts = scan_exits(
            Path("/fake"), _CFG,
            _positions=[_pos("NVDA", 100.0, 90.0)],
            _prices={"NVDA": 85.0},
        )
        assert alerts[0]["condition"] == "stop_breached"

    def test_approaching_stop_within_pct(self):
        # stop=$90, 2% above → $91.80; current=$91.0 (1.1% above) → alert
        alerts = scan_exits(
            Path("/fake"), _CFG,
            _positions=[_pos("NVDA", 100.0, 90.0)],
            _prices={"NVDA": 91.0},
        )
        assert len(alerts) == 1
        assert alerts[0]["condition"] == "approaching_stop"

    def test_no_alert_outside_approaching_pct(self):
        # stop=$90, 2% above → $91.80; current=$95.0 (5.6% above) → no alert
        alerts = scan_exits(
            Path("/fake"), _CFG,
            _positions=[_pos("NVDA", 100.0, 90.0)],
            _prices={"NVDA": 95.0},
        )
        assert alerts == []

    def test_2r_profit_triggered(self):
        # entry=$100, stop=$90, risk=$10 → 2R target=$120; current=$125
        alerts = scan_exits(
            Path("/fake"), _CFG,
            _positions=[_pos("NVDA", 100.0, 90.0)],
            _prices={"NVDA": 125.0},
        )
        assert len(alerts) == 1
        assert alerts[0]["condition"] == "2r_profit"
        assert alerts[0]["target_2r"] == pytest.approx(120.0)

    def test_no_stop_price_skipped(self):
        alerts = scan_exits(
            Path("/fake"), _CFG,
            _positions=[_pos("NVDA", 100.0, None)],
            _prices={"NVDA": 85.0},
        )
        assert alerts == []

    def test_invalid_stop_above_entry_skipped(self):
        # risk = entry - stop = 100 - 110 = -10 → skip
        alerts = scan_exits(
            Path("/fake"), _CFG,
            _positions=[_pos("NVDA", 100.0, 110.0)],
            _prices={"NVDA": 95.0},
        )
        assert alerts == []

    def test_missing_current_price_skipped(self):
        alerts = scan_exits(
            Path("/fake"), _CFG,
            _positions=[_pos("NVDA", 100.0, 90.0)],
            _prices={},
        )
        assert alerts == []

    def test_sort_order_stop_breach_first(self):
        positions = [
            _pos("AMD", 100.0, 90.0),   # prices → approaching ($91)
            _pos("NVDA", 100.0, 90.0),  # prices → stop_breached ($85)
        ]
        prices = {"AMD": 91.0, "NVDA": 85.0}
        alerts = scan_exits(Path("/fake"), _CFG, _positions=positions, _prices=prices)
        conditions = [a["condition"] for a in alerts]
        assert conditions[0] == "stop_breached"
        assert conditions[1] == "approaching_stop"

    def test_no_alerts_when_price_in_middle(self):
        # current=$98: above $91.80 (approaching threshold) and below $120 (2R)
        alerts = scan_exits(
            Path("/fake"), _CFG,
            _positions=[_pos("NVDA", 100.0, 90.0)],
            _prices={"NVDA": 98.0},
        )
        assert alerts == []

    def test_empty_positions(self):
        alerts = scan_exits(Path("/fake"), _CFG, _positions=[], _prices={})
        assert alerts == []


# ── format_exit_message ───────────────────────────────────────────────────────

class TestFormatExitMessage:
    def test_returns_none_when_empty(self):
        assert format_exit_message([], date(2026, 8, 4)) is None

    def test_stop_breached_format(self):
        alert = {
            "ticker": "NVDA", "condition": "stop_breached",
            "current": 88.50, "entry": 100.0, "stop": 90.0,
            "risk": 10.0, "pnl_pct": -0.115,
        }
        msg = format_exit_message([alert], date(2026, 8, 4))
        assert msg is not None
        assert "STOP BREACHED" in msg
        assert "NVDA" in msg
        assert "$88.50" in msg
        assert "$90.00" in msg
        assert "Exit Alerts" in msg
        assert "2026-08-04" in msg

    def test_approaching_stop_format(self):
        alert = {
            "ticker": "AMD", "condition": "approaching_stop",
            "current": 91.0, "entry": 100.0, "stop": 90.0,
            "risk": 10.0, "pct_above_stop": 0.011, "pnl_pct": -0.09,
        }
        msg = format_exit_message([alert], date(2026, 8, 4))
        assert msg is not None
        assert "APPROACHING STOP" in msg
        assert "AMD" in msg
        assert "$90.00" in msg

    def test_2r_profit_format(self):
        alert = {
            "ticker": "NVDA", "condition": "2r_profit",
            "current": 125.0, "entry": 100.0, "stop": 90.0,
            "risk": 10.0, "target_2r": 120.0, "pnl_pct": 0.25,
        }
        msg = format_exit_message([alert], date(2026, 8, 4))
        assert msg is not None
        assert "2R PROFIT" in msg
        assert "$120.00" in msg
        assert "+25.0%" in msg


# ── deliver_alerts ────────────────────────────────────────────────────────────

class TestDeliverAlerts:
    def test_returns_true_on_success(self):
        with patch("weaver.exits.send_telegram", return_value=True):
            result = deliver_alerts("msg", "tok", "chat", has_stop_breach=False)
        assert result is True

    def test_stop_breach_retried_once_on_failure(self):
        with patch("weaver.exits.send_telegram", side_effect=[False, True]) as mock_send:
            with patch("time.sleep"):
                result = deliver_alerts("msg", "tok", "chat", has_stop_breach=True)
        assert mock_send.call_count == 2
        assert result is True

    def test_non_stop_breach_not_retried(self):
        with patch("weaver.exits.send_telegram", return_value=False) as mock_send:
            deliver_alerts("msg", "tok", "chat", has_stop_breach=False)
        assert mock_send.call_count == 1

    def test_stop_breach_failure_logs_failed_marker(self, capsys):
        with patch("weaver.exits.send_telegram", return_value=False):
            with patch("time.sleep"):
                deliver_alerts("msg", "tok", "chat", has_stop_breach=True)
        err = capsys.readouterr().err
        assert "TELEGRAM DELIVERY FAILED" in err
        assert "STOP BREACH ALERT NOT SENT" in err

    def test_non_stop_breach_failure_no_failed_marker(self, capsys):
        with patch("weaver.exits.send_telegram", return_value=False):
            deliver_alerts("msg", "tok", "chat", has_stop_breach=False)
        err = capsys.readouterr().err
        assert "TELEGRAM DELIVERY FAILED" not in err
