"""Tests for trade journal logging (weaver.journal)."""

from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner

from weaver.journal import (
    close_buy_position,
    find_open_buys,
    log_trade,
    scan_open_positions,
)
from weaver.cli import main

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


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_buy(
    journal_dir: Path,
    ticker: str = "NVDA",
    qty: float = 10.0,
    price: float = 130.0,
    status: str = "open",
    date_str: str = "2026-08-01",
) -> Path:
    content = (
        f"---\n"
        f"ticker: {ticker}\n"
        f"action: buy\n"
        f"quantity: {qty}\n"
        f"price: {price}\n"
        f'reason: "test"\n'
        f'stop_condition: "below 125"\n'
        f"time_horizon: swing\n"
        f"entry_date: {date_str}\n"
        f"status: {status}\n"
        f"---\n"
        f"# {date_str} {ticker} Buy\n"
    )
    path = journal_dir / f"{date_str} {ticker} buy.md"
    path.write_text(content, encoding="utf-8")
    return path


def _env(tmp_path: Path) -> dict[str, str]:
    return {"OBSIDIAN_VAULT_PATH": str(tmp_path)}


# ── TestCloseBuyPosition ──────────────────────────────────────────────────────

class TestCloseBuyPosition:
    def test_full_sell_sets_status_closed(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        buy_path = _write_buy(journal_dir)

        close_buy_position(buy_path, sell_qty=10.0, sell_price=145.0, sell_date=date(2026, 8, 10))

        content = buy_path.read_text()
        assert "status: closed" in content
        assert "exit_price: 145.0" in content
        assert "exit_date: 2026-08-10" in content

    def test_full_sell_preserves_original_quantity(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        buy_path = _write_buy(journal_dir, qty=10.0)

        close_buy_position(buy_path, sell_qty=10.0, sell_price=145.0, sell_date=date(2026, 8, 10))

        content = buy_path.read_text()
        assert "quantity: 10.0" in content
        assert "quantity_remaining" not in content
        assert "quantity_sold" not in content

    def test_partial_sell_sets_status_partial(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        buy_path = _write_buy(journal_dir, qty=10.0)

        result = close_buy_position(buy_path, sell_qty=5.0, sell_price=145.0, sell_date=date(2026, 8, 10))

        assert result == "partial"
        content = buy_path.read_text()
        assert "status: partial" in content

    def test_partial_sell_appends_remaining_and_sold(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        buy_path = _write_buy(journal_dir, qty=10.0)

        close_buy_position(buy_path, sell_qty=3.0, sell_price=145.0, sell_date=date(2026, 8, 10))

        content = buy_path.read_text()
        assert "quantity_remaining: 7.0" in content
        assert "quantity_sold: 3.0" in content
        assert "quantity: 10.0" in content  # original preserved

    def test_realized_pnl_computed_correctly(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        buy_path = _write_buy(journal_dir, qty=10.0, price=130.0)

        close_buy_position(buy_path, sell_qty=10.0, sell_price=145.0, sell_date=date(2026, 8, 10))

        content = buy_path.read_text()
        # pnl = (145 - 130) * 10 = 150.0
        assert "realized_pnl: 150.0" in content

    def test_returns_closed_on_full_sell(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        buy_path = _write_buy(journal_dir)
        result = close_buy_position(buy_path, 10.0, 145.0, date(2026, 8, 10))
        assert result == "closed"

    def test_sell_more_than_bought_still_closes(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        buy_path = _write_buy(journal_dir, qty=5.0)
        result = close_buy_position(buy_path, sell_qty=10.0, sell_price=145.0, sell_date=date(2026, 8, 10))
        assert result == "closed"
        assert "status: closed" in buy_path.read_text()


# ── TestFindOpenBuys ──────────────────────────────────────────────────────────

class TestFindOpenBuys:
    def test_returns_open_buy(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        _write_buy(journal_dir, ticker="NVDA")
        result = find_open_buys(journal_dir, "NVDA")
        assert len(result) == 1

    def test_excludes_closed_buy(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        _write_buy(journal_dir, ticker="NVDA", status="closed")
        result = find_open_buys(journal_dir, "NVDA")
        assert result == []

    def test_excludes_other_ticker(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        _write_buy(journal_dir, ticker="AMD")
        result = find_open_buys(journal_dir, "NVDA")
        assert result == []

    def test_multiple_open_buys_returned(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        _write_buy(journal_dir, date_str="2026-08-01")
        _write_buy(journal_dir, date_str="2026-08-05")
        result = find_open_buys(journal_dir, "NVDA")
        assert len(result) == 2

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        assert find_open_buys(journal_dir, "NVDA") == []


# ── TestLogTradeSellClosure (CLI integration) ─────────────────────────────────

class TestLogTradeSellClosure:
    def test_sell_with_linked_buy_closes_buy(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        buy_path = _write_buy(journal_dir)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "log-trade", "--ticker", "NVDA", "--action", "sell",
                "--quantity", "10", "--price", "145.00", "--horizon", "swing",
                "--reason", "Target hit", "--stop", "N/A",
                "--linked-buy", "2026-08-01 NVDA buy",
            ],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "status: closed" in buy_path.read_text()
        assert "Buy closed" in result.output

    def test_sell_with_missing_linked_buy_warns(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "log-trade", "--ticker", "NVDA", "--action", "sell",
                "--quantity", "5", "--price", "145.00", "--horizon", "swing",
                "--reason", "Exit", "--stop", "N/A",
                "--linked-buy", "nonexistent file",
            ],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0  # sell still written
        assert "not found" in result.output.lower()
        # sell file should exist
        journal_dir = tmp_path / "Trading" / "Journal"
        assert any(f.name.endswith("sell.md") for f in journal_dir.glob("*.md"))

    def test_auto_match_single_open_buy(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        buy_path = _write_buy(journal_dir)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "log-trade", "--ticker", "NVDA", "--action", "sell",
                "--quantity", "10", "--price", "150.00", "--horizon", "swing",
                "--reason", "Exit", "--stop", "N/A",
            ],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "status: closed" in buy_path.read_text()
        assert "Matched open buy" in result.output

    def test_auto_match_multiple_open_buys_warns_and_does_not_pick(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        buy1 = _write_buy(journal_dir, date_str="2026-08-01")
        buy2 = _write_buy(journal_dir, date_str="2026-08-05")

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "log-trade", "--ticker", "NVDA", "--action", "sell",
                "--quantity", "10", "--price", "150.00", "--horizon", "swing",
                "--reason", "Exit", "--stop", "N/A",
            ],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0
        # Neither buy should be closed
        assert "status: open" in buy1.read_text()
        assert "status: open" in buy2.read_text()
        assert "multiple" in result.output.lower()
        assert "--linked-buy" in result.output

    def test_partial_sell_reflected_in_buy_file(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        buy_path = _write_buy(journal_dir)

        runner = CliRunner()
        runner.invoke(
            main,
            [
                "log-trade", "--ticker", "NVDA", "--action", "sell",
                "--quantity", "5", "--price", "145.00", "--horizon", "swing",
                "--reason", "Trim", "--stop", "N/A",
                "--linked-buy", "2026-08-01 NVDA buy",
            ],
            env=_env(tmp_path),
        )
        content = buy_path.read_text()
        assert "status: partial" in content
        assert "quantity_remaining: 5.0" in content


# ── TestScanOpenPositions ─────────────────────────────────────────────────────

class TestScanOpenPositions:
    def test_open_buy_returned(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        _write_buy(journal_dir)
        positions = scan_open_positions(tmp_path, {})
        assert any(p["ticker"] == "NVDA" for p in positions)

    def test_closed_buy_excluded(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        _write_buy(journal_dir, status="closed")
        assert scan_open_positions(tmp_path, {}) == []

    def test_partial_buy_included_with_remaining_qty(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        path = _write_buy(journal_dir, qty=10.0, status="partial")
        # Add quantity_remaining to frontmatter
        content = path.read_text()
        path.write_text(
            content.replace("status: partial\n", "status: partial\nquantity_remaining: 5.0\n"),
            encoding="utf-8",
        )
        positions = scan_open_positions(tmp_path, {})
        pos = next(p for p in positions if p["ticker"] == "NVDA")
        assert float(pos["quantity"]) == 5.0  # uses remaining, not original 10

    def test_legacy_sell_reference_excluded(self, tmp_path: Path) -> None:
        # Pre-Build1: buy has status: open but a sell file references it
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        _write_buy(journal_dir)
        sell_content = (
            "---\n"
            "ticker: NVDA\n"
            "action: sell\n"
            "quantity: 10.0\n"
            "price: 150.0\n"
            'reason: "exit"\n'
            'stop_condition: "N/A"\n'
            "time_horizon: swing\n"
            "entry_date: 2026-08-10\n"
            "status: open\n"
            'linked_buy: "[[2026-08-01 NVDA buy]]"\n'
            "---\n"
        )
        (journal_dir / "2026-08-10 NVDA sell.md").write_text(sell_content, encoding="utf-8")
        positions = scan_open_positions(tmp_path, {})
        assert not any(p["ticker"] == "NVDA" for p in positions)

    def test_no_journal_dir_returns_empty(self, tmp_path: Path) -> None:
        assert scan_open_positions(tmp_path, {}) == []


# ── TestPositionsCommand ──────────────────────────────────────────────────────

class TestPositionsCommand:
    def test_no_positions_message(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["positions"], env=_env(tmp_path))
        assert result.exit_code == 0, result.output
        assert "No open positions" in result.output

    def test_shows_open_position(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        _write_buy(journal_dir, ticker="NVDA", qty=5.0, price=130.0)

        runner = CliRunner()
        result = runner.invoke(main, ["positions"], env=_env(tmp_path))
        assert result.exit_code == 0, result.output
        assert "NVDA" in result.output
        assert "130" in result.output

    def test_shows_basket_exposure(self, tmp_path: Path) -> None:
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        _write_buy(journal_dir, ticker="NVDA", qty=1.0, price=50.0)

        runner = CliRunner()
        result = runner.invoke(main, ["positions"], env=_env(tmp_path))
        assert "Basket exposure" in result.output

    def test_over_cap_warning_shown(self, tmp_path: Path) -> None:
        # With portfolio_value=1000 and cap=10%, basket_cap=$100
        # A $200 position is over cap
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        _write_buy(journal_dir, ticker="NVDA", qty=2.0, price=100.0)  # $200 committed

        runner = CliRunner()
        result = runner.invoke(main, ["positions"], env=_env(tmp_path))
        assert "WARNING" in result.output
        assert "OVER CAP" in result.output

    def test_no_over_cap_warning_when_within_limits(self, tmp_path: Path) -> None:
        # $50 committed, cap=$100 — should be fine
        journal_dir = tmp_path / "Trading" / "Journal"
        journal_dir.mkdir(parents=True)
        _write_buy(journal_dir, ticker="NVDA", qty=1.0, price=50.0)

        runner = CliRunner()
        result = runner.invoke(main, ["positions"], env=_env(tmp_path))
        assert "WARNING" not in result.output
