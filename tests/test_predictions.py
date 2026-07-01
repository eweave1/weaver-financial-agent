"""Tests for prediction logging and resolution (weaver.predictions)."""

from datetime import date
from pathlib import Path

import pytest

from weaver.predictions import (
    list_pending_resolutions,
    log_prediction,
    resolve_prediction,
)

# Dates used across tests
TODAY = date(2026, 6, 30)
FUTURE = date(2099, 1, 1)   # safely in the future for all creation calls
PAST = date(2020, 1, 1)     # clearly in the past

# Minimal valid kwargs
_BASE = dict(
    ticker="NVDA",
    direction="up",
    timeframe="2 weeks",
    confidence=0.70,
    reasoning="Earnings catalyst and AI capex cycle accelerating",
    resolve_by=FUTURE,
    prediction_date=TODAY,
)


class TestLogPrediction:
    def test_creates_file_in_predictions_dir(self, tmp_path: Path) -> None:
        fp = log_prediction(vault_path=tmp_path, **_BASE)
        assert fp.exists()
        assert fp.parent == tmp_path / "Trading" / "Predictions"

    def test_filename_format(self, tmp_path: Path) -> None:
        fp = log_prediction(vault_path=tmp_path, **_BASE)
        assert fp.name == f"{TODAY} NVDA.md"

    def test_ticker_normalized_to_uppercase(self, tmp_path: Path) -> None:
        fp = log_prediction(vault_path=tmp_path, **{**_BASE, "ticker": "nvda"})
        assert "NVDA" in fp.name

    def test_predictions_dir_created_automatically(self, tmp_path: Path) -> None:
        log_prediction(vault_path=tmp_path, **_BASE)
        assert (tmp_path / "Trading" / "Predictions").is_dir()

    def test_frontmatter_contains_all_fields(self, tmp_path: Path) -> None:
        fp = log_prediction(vault_path=tmp_path, **_BASE)
        content = fp.read_text()
        assert "ticker: NVDA" in content
        assert "direction: up" in content
        assert "confidence: 0.7" in content
        assert f"resolve_by: {FUTURE}" in content
        assert f"prediction_date: {TODAY}" in content
        assert "status: open" in content

    def test_duplicate_date_creates_numbered_file(self, tmp_path: Path) -> None:
        fp1 = log_prediction(vault_path=tmp_path, **_BASE)
        fp2 = log_prediction(
            vault_path=tmp_path, **{**_BASE, "reasoning": "Second call"}
        )
        assert fp1.name == f"{TODAY} NVDA.md"
        assert fp2.name == f"{TODAY} NVDA (2).md"

    def test_invalid_direction_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="direction"):
            log_prediction(vault_path=tmp_path, **{**_BASE, "direction": "sideways"})

    def test_confidence_above_1_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="confidence"):
            log_prediction(vault_path=tmp_path, **{**_BASE, "confidence": 1.5})

    def test_confidence_below_0_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="confidence"):
            log_prediction(vault_path=tmp_path, **{**_BASE, "confidence": -0.1})

    def test_resolve_by_same_as_prediction_date_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="future"):
            log_prediction(
                vault_path=tmp_path, **{**_BASE, "resolve_by": TODAY}
            )

    def test_resolve_by_before_prediction_date_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="future"):
            log_prediction(
                vault_path=tmp_path, **{**_BASE, "resolve_by": PAST}
            )

    def test_body_contains_reasoning(self, tmp_path: Path) -> None:
        fp = log_prediction(vault_path=tmp_path, **_BASE)
        assert "Earnings catalyst" in fp.read_text()


class TestListPendingResolutions:
    def _plant(
        self,
        tmp_path: Path,
        ticker: str,
        resolve_by: date,
        status: str = "open",
    ) -> Path:
        """Create a prediction file, then patch its resolve_by and status."""
        fp = log_prediction(
            vault_path=tmp_path,
            ticker=ticker,
            direction="up",
            timeframe="1 week",
            confidence=0.6,
            reasoning="Test",
            resolve_by=FUTURE,       # must pass validation
            prediction_date=TODAY,
        )
        content = fp.read_text()
        content = content.replace(f"resolve_by: {FUTURE}", f"resolve_by: {resolve_by}")
        if status != "open":
            content = content.replace("status: open", f"status: {status}")
        fp.write_text(content)
        return fp

    def test_returns_overdue_open_predictions(self, tmp_path: Path) -> None:
        self._plant(tmp_path, "NVDA", date(2026, 6, 1))
        self._plant(tmp_path, "AMD", date(2026, 6, 1))
        pending = list_pending_resolutions(tmp_path, as_of_date=TODAY)
        assert len(pending) == 2

    def test_excludes_future_predictions(self, tmp_path: Path) -> None:
        self._plant(tmp_path, "NVDA", date(2026, 6, 1))   # overdue
        self._plant(tmp_path, "AMD", date(2099, 1, 1))    # future
        pending = list_pending_resolutions(tmp_path, as_of_date=TODAY)
        assert len(pending) == 1
        assert pending[0]["ticker"] == "NVDA"

    def test_excludes_already_resolved_predictions(self, tmp_path: Path) -> None:
        self._plant(tmp_path, "NVDA", date(2026, 6, 1), status="resolved")
        pending = list_pending_resolutions(tmp_path, as_of_date=TODAY)
        assert pending == []

    def test_nonexistent_vault_returns_empty(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        assert list_pending_resolutions(missing, as_of_date=TODAY) == []

    def test_empty_predictions_dir_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "Trading" / "Predictions").mkdir(parents=True)
        assert list_pending_resolutions(tmp_path, as_of_date=TODAY) == []

    def test_result_contains_expected_keys(self, tmp_path: Path) -> None:
        self._plant(tmp_path, "NVDA", date(2026, 6, 1))
        pending = list_pending_resolutions(tmp_path, as_of_date=TODAY)
        assert len(pending) == 1
        record = pending[0]
        for key in ("file", "ticker", "direction", "timeframe", "confidence",
                    "reasoning", "resolve_by", "prediction_date"):
            assert key in record, f"missing key: {key}"


class TestResolvePrediction:
    def _open_prediction(self, tmp_path: Path, ticker: str = "NVDA") -> Path:
        return log_prediction(vault_path=tmp_path, **{**_BASE, "ticker": ticker})

    def test_status_changes_to_resolved(self, tmp_path: Path) -> None:
        fp = self._open_prediction(tmp_path)
        resolve_prediction(fp, "right", "up", "Stock moved as expected", TODAY)
        assert "status: resolved" in fp.read_text()
        assert "status: open" not in fp.read_text()

    def test_outcome_written_to_frontmatter(self, tmp_path: Path) -> None:
        fp = self._open_prediction(tmp_path)
        resolve_prediction(fp, "right", "up", "Notes here", TODAY)
        assert "outcome: right" in fp.read_text()

    def test_actual_direction_written_to_frontmatter(self, tmp_path: Path) -> None:
        fp = self._open_prediction(tmp_path)
        resolve_prediction(fp, "wrong", "down", "Reversed sharply", TODAY)
        assert "actual_direction: down" in fp.read_text()

    def test_resolution_date_written_to_frontmatter(self, tmp_path: Path) -> None:
        fp = self._open_prediction(tmp_path)
        resolve_prediction(fp, "right", "up", "On target", TODAY)
        assert f"resolution_date: {TODAY}" in fp.read_text()

    def test_resolution_section_appended_to_body(self, tmp_path: Path) -> None:
        fp = self._open_prediction(tmp_path)
        resolve_prediction(fp, "unforeseen", "up", "Musk tweet caused 20% spike", TODAY)
        content = fp.read_text()
        assert "## Resolution" in content
        assert "Unforeseen" in content

    def test_unforeseen_outcome_recorded(self, tmp_path: Path) -> None:
        fp = self._open_prediction(tmp_path)
        resolve_prediction(fp, "unforeseen", "up", "Macro shock", TODAY)
        assert "outcome: unforeseen" in fp.read_text()

    def test_wrong_outcome_recorded(self, tmp_path: Path) -> None:
        fp = self._open_prediction(tmp_path)
        resolve_prediction(fp, "wrong", "down", "Thesis failed", TODAY)
        assert "outcome: wrong" in fp.read_text()

    def test_invalid_outcome_raises(self, tmp_path: Path) -> None:
        fp = self._open_prediction(tmp_path)
        with pytest.raises(ValueError, match="outcome"):
            resolve_prediction(fp, "maybe", "up", "Notes")  # type: ignore[arg-type]

    def test_resolution_notes_preserved(self, tmp_path: Path) -> None:
        fp = self._open_prediction(tmp_path)
        notes = "Stock moved up 8% driven by strong guidance"
        resolve_prediction(fp, "right", "up", notes, TODAY)
        assert notes in fp.read_text()
