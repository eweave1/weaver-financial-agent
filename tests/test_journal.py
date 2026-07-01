"""Tests for trade journal logging (weaver.journal)."""

from datetime import date
from pathlib import Path

import pytest

from weaver.journal import log_trade

# Shared test date — avoids relying on date.today() in assertions
TRADE_DATE = date(2026, 6, 30)

# Minimal valid kwargs shared across tests
_BASE = dict(
    ticker="NVDA",
    action="buy",
    quantity=10,
    price=150.0,
    reason="Breakout above resistance on volume",
    stop_condition="Close below 145",
    time_horizon="swing",
    trade_date=TRADE_DATE,
)


class TestLogTrade:
    def test_creates_file_in_journal_dir(self, tmp_path: Path) -> None:
        fp = log_trade(vault_path=tmp_path, **_BASE)
        assert fp.exists()
        assert fp.parent == tmp_path / "Trading" / "Journal"

    def test_filename_format(self, tmp_path: Path) -> None:
        fp = log_trade(vault_path=tmp_path, **_BASE)
        assert fp.name == "2026-06-30 NVDA buy.md"

    def test_ticker_normalized_to_uppercase(self, tmp_path: Path) -> None:
        fp = log_trade(vault_path=tmp_path, **{**_BASE, "ticker": "nvda"})
        assert "NVDA" in fp.name

    def test_journal_dir_created_automatically(self, tmp_path: Path) -> None:
        log_trade(vault_path=tmp_path, **_BASE)
        assert (tmp_path / "Trading" / "Journal").is_dir()

    def test_frontmatter_contains_all_required_fields(self, tmp_path: Path) -> None:
        fp = log_trade(vault_path=tmp_path, **{**_BASE, "target_exit": 170.0})
        content = fp.read_text()
        assert "ticker: NVDA" in content
        assert "action: buy" in content
        assert "quantity: 10" in content
        assert "price: 150.0" in content
        assert "time_horizon: swing" in content
        assert "target_exit: 170.0" in content
        assert "entry_date: 2026-06-30" in content
        assert "status: open" in content

    def test_target_exit_omitted_when_not_provided(self, tmp_path: Path) -> None:
        fp = log_trade(vault_path=tmp_path, **_BASE)
        assert "target_exit" not in fp.read_text()

    def test_sell_entry_includes_linked_buy(self, tmp_path: Path) -> None:
        fp = log_trade(
            vault_path=tmp_path,
            **{
                **_BASE,
                "action": "sell",
                "price": 165.0,
                "trade_date": date(2026, 7, 5),
                "linked_buy_file": "2026-06-30 NVDA buy",
            },
        )
        content = fp.read_text()
        assert "linked_buy" in content
        assert "2026-06-30 NVDA buy" in content

    def test_duplicate_date_creates_numbered_file(self, tmp_path: Path) -> None:
        fp1 = log_trade(vault_path=tmp_path, **_BASE)
        fp2 = log_trade(vault_path=tmp_path, **{**_BASE, "reason": "Second entry"})
        assert fp1.name == "2026-06-30 NVDA buy.md"
        assert fp2.name == "2026-06-30 NVDA buy (2).md"

    def test_invalid_action_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="action"):
            log_trade(vault_path=tmp_path, **{**_BASE, "action": "hold"})

    def test_invalid_horizon_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="time_horizon"):
            log_trade(vault_path=tmp_path, **{**_BASE, "time_horizon": "week"})

    def test_body_contains_reason_and_stop(self, tmp_path: Path) -> None:
        fp = log_trade(vault_path=tmp_path, **_BASE)
        content = fp.read_text()
        assert "Breakout above resistance on volume" in content
        assert "Close below 145" in content

    def test_different_tickers_produce_separate_files(self, tmp_path: Path) -> None:
        fp_nvda = log_trade(vault_path=tmp_path, **_BASE)
        fp_amd = log_trade(
            vault_path=tmp_path, **{**_BASE, "ticker": "AMD", "reason": "AMD trade"}
        )
        assert fp_nvda != fp_amd
        assert fp_nvda.exists()
        assert fp_amd.exists()
