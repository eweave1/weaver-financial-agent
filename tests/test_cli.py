"""CLI integration tests using Click's CliRunner.

CliRunner captures stdin/stdout and patches os.environ, so these tests run
without a real terminal and without touching the actual vault at C:\\brain.

The `input` parameter simulates keystrokes: each line maps to one prompt answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from weaver.cli import main


def _env(tmp_path: Path) -> dict[str, str]:
    """Environment with vault path pointing at a temp directory."""
    return {"OBSIDIAN_VAULT_PATH": str(tmp_path)}


# ── log-trade: fully interactive ──────────────────────────────────────────────

class TestLogTradeInteractive:
    def test_fully_interactive_no_flags(self, tmp_path: Path) -> None:
        """Running wf log-trade with zero flags prompts for every field."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-trade"],
            input="\n".join([
                "NVDA",     # ticker
                "buy",      # action
                "10",       # quantity
                "130.00",   # price
                "swing",    # horizon
                "",         # target (skip)
                "Breakout above resistance on volume",  # reason
                "Close below 125",                      # stop
            ]) + "\n",
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "Trade logged" in result.output
        assert (tmp_path / "Trading" / "Journal").exists()

    def test_ticker_provided_rest_prompted(self, tmp_path: Path) -> None:
        """Providing --ticker skips the ticker prompt; everything else is prompted."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-trade", "--ticker", "AMD"],
            input="\n".join([
                "sell",
                "5",
                "120.00",
                "day",
                "",
                "Target reached",
                "N/A",
            ]) + "\n",
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "AMD" in result.output

    def test_all_flags_no_prompts(self, tmp_path: Path) -> None:
        """All flags provided — no interactive prompts triggered."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "log-trade",
                "--ticker", "NVDA",
                "--action", "buy",
                "--quantity", "10",
                "--price", "130.00",
                "--horizon", "swing",
                "--reason", "Test reason",
                "--stop", "Below 125",
            ],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "Trade logged" in result.output
        # No prompts means output should just be the confirmation line
        assert "Ticker" not in result.output

    def test_reason_with_dollar_signs_interactive(self, tmp_path: Path) -> None:
        """Dollar signs in interactively entered reason are preserved correctly."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-trade", "--ticker", "NVDA", "--action", "buy",
             "--quantity", "10", "--price", "130.00", "--horizon", "swing",
             "--stop", "Below $125"],
            input="Target $190 from $130, risk $348->$297\n",  # reason with $ and ->
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        files = list((tmp_path / "Trading" / "Journal").glob("*.md"))
        assert len(files) == 1
        content = files[0].read_text()
        assert "$190" in content
        assert "$348->$297" in content

    def test_target_skipped_when_enter_pressed(self, tmp_path: Path) -> None:
        """Pressing Enter at the target prompt leaves target_exit as None."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-trade", "--ticker", "NVDA", "--action", "buy",
             "--quantity", "10", "--price", "130.00", "--horizon", "swing"],
            input="\n".join(["", "Test reason", "Below 125"]) + "\n",
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        files = list((tmp_path / "Trading" / "Journal").glob("*.md"))
        content = files[0].read_text()
        assert "target_exit" not in content

    def test_target_accepted_when_number_entered(self, tmp_path: Path) -> None:
        """Entering a number at the target prompt sets target_exit."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-trade", "--ticker", "NVDA", "--action", "buy",
             "--quantity", "10", "--price", "130.00", "--horizon", "swing"],
            input="\n".join(["150.00", "Test reason", "Below 125"]) + "\n",
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        content = list((tmp_path / "Trading" / "Journal").glob("*.md"))[0].read_text()
        assert "target_exit: 150.0" in content

    def test_success_message_shows_full_path(self, tmp_path: Path) -> None:
        """Success message contains the full vault path so there's no ambiguity."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-trade", "--ticker", "NVDA", "--action", "buy",
             "--quantity", "10", "--price", "130.00", "--horizon", "swing",
             "--reason", "Test", "--stop", "Below 125"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0
        # Full path in message means the vault root is visible
        assert str(tmp_path) in result.output


class TestLogTradeValidation:
    def test_invalid_action_flag_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-trade", "--ticker", "NVDA", "--action", "hold",
             "--quantity", "10", "--price", "130.00", "--horizon", "swing",
             "--reason", "Test", "--stop", "Below 125"],
            env=_env(tmp_path),
        )
        assert result.exit_code != 0

    def test_invalid_horizon_flag_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-trade", "--ticker", "NVDA", "--action", "buy",
             "--quantity", "10", "--price", "130.00", "--horizon", "week",
             "--reason", "Test", "--stop", "Below 125"],
            env=_env(tmp_path),
        )
        assert result.exit_code != 0


# ── log-prediction: fully interactive ────────────────────────────────────────

class TestLogPredictionInteractive:
    def test_fully_interactive_no_flags(self, tmp_path: Path) -> None:
        """Running wf log-prediction with zero flags prompts for every field."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-prediction"],
            input="\n".join([
                "NVDA",
                "up",
                "2 weeks",
                "0.7",
                "2026-09-01",
                "Earnings catalyst, target $190 from $130",
            ]) + "\n",
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "Prediction logged" in result.output

    def test_all_flags_no_prompts(self, tmp_path: Path) -> None:
        """All flags provided — no interactive prompts triggered."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "log-prediction",
                "--ticker", "NVDA",
                "--direction", "up",
                "--timeframe", "2 weeks",
                "--confidence", "0.7",
                "--resolve-by", "2026-09-01",
                "--reasoning", "Test reasoning",
            ],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "Prediction logged" in result.output
        assert "Ticker" not in result.output

    def test_reasoning_with_dollar_signs_and_arrows(self, tmp_path: Path) -> None:
        """Dollar signs and arrows in reasoning survive the interactive prompt."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-prediction", "--ticker", "NVDA", "--direction", "up",
             "--timeframe", "2 weeks", "--confidence", "0.7",
             "--resolve-by", "2026-09-01"],
            input="Support at $130, resistance $190, range $348->$297\n",
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        files = list((tmp_path / "Trading" / "Predictions").glob("*.md"))
        content = files[0].read_text()
        assert "$190" in content
        assert "$348->$297" in content

    def test_success_message_shows_full_path(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-prediction", "--ticker", "NVDA", "--direction", "up",
             "--timeframe", "2 weeks", "--confidence", "0.7",
             "--resolve-by", "2026-09-01", "--reasoning", "Test"],
            env=_env(tmp_path),
        )
        assert result.exit_code == 0
        assert str(tmp_path) in result.output


class TestLogPredictionValidation:
    def test_confidence_above_1_flag_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-prediction", "--ticker", "NVDA", "--direction", "up",
             "--timeframe", "2 weeks", "--confidence", "1.5",
             "--resolve-by", "2026-09-01", "--reasoning", "Test"],
            env=_env(tmp_path),
        )
        assert result.exit_code != 0
        assert "confidence" in result.output.lower()

    def test_confidence_below_0_flag_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-prediction", "--ticker", "NVDA", "--direction", "up",
             "--timeframe", "2 weeks", "--confidence", "-0.1",
             "--resolve-by", "2026-09-01", "--reasoning", "Test"],
            env=_env(tmp_path),
        )
        assert result.exit_code != 0

    def test_past_resolve_by_flag_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-prediction", "--ticker", "NVDA", "--direction", "up",
             "--timeframe", "2 weeks", "--confidence", "0.7",
             "--resolve-by", "2020-01-01", "--reasoning", "Test"],
            env=_env(tmp_path),
        )
        assert result.exit_code != 0
        assert "future" in result.output.lower()

    def test_malformed_resolve_by_flag_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-prediction", "--ticker", "NVDA", "--direction", "up",
             "--timeframe", "2 weeks", "--confidence", "0.7",
             "--resolve-by", "not-a-date", "--reasoning", "Test"],
            env=_env(tmp_path),
        )
        assert result.exit_code != 0
        assert "yyyy-mm-dd" in result.output.lower() or "format" in result.output.lower()

    def test_invalid_direction_flag_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-prediction", "--ticker", "NVDA", "--direction", "sideways",
             "--timeframe", "2 weeks", "--confidence", "0.7",
             "--resolve-by", "2026-09-01", "--reasoning", "Test"],
            env=_env(tmp_path),
        )
        assert result.exit_code != 0

    def test_interactive_past_date_reprompts(self, tmp_path: Path) -> None:
        """Entering a past date interactively shows an error and re-prompts."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-prediction"],
            input="\n".join([
                "NVDA",
                "up",
                "2 weeks",
                "0.7",
                "2020-01-01",    # past — rejected
                "2026-09-01",    # future — accepted
                "Test reasoning",
            ]) + "\n",
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        assert "future" in result.output.lower()
        # File should have been written with the second (valid) date
        files = list((tmp_path / "Trading" / "Predictions").glob("*.md"))
        assert len(files) == 1

    def test_interactive_confidence_reprompts_on_out_of_range(
        self, tmp_path: Path
    ) -> None:
        """Entering confidence > 1.0 interactively re-prompts until valid."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["log-prediction"],
            input="\n".join([
                "NVDA",
                "up",
                "2 weeks",
                "1.5",       # out of range — Click re-prompts automatically
                "0.7",       # valid
                "2026-09-01",
                "Test reasoning",
            ]) + "\n",
            env=_env(tmp_path),
        )
        assert result.exit_code == 0, result.output
        files = list((tmp_path / "Trading" / "Predictions").glob("*.md"))
        assert len(files) == 1
