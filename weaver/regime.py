"""Market regime detection — QQQ 20-day MA circuit breaker.

Returns "uptrend", "downtrend", or "unknown".
"unknown" always fails open: callers proceed as if there is no downtrend.
"""

from __future__ import annotations

from typing import Any, Optional


def get_regime(
    config: dict[str, Any],
    _qqq_data: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Compute market regime from QQQ vs its N-day MA.

    _qqq_data: injection point for tests — list of dicts containing at least
    a "close" key. When omitted, fetches live OHLCV from yfinance.

    Returns:
        regime      "uptrend" | "downtrend" | "unknown"
        qqq_price   float | None
        qqq_ma20    float | None    (named ma20 regardless of ma_period setting)
        pct_from_ma float | None    (positive = above MA, negative = below)
    """
    cfg = config.get("regime", {})
    ma_period = int(cfg.get("ma_period", 20))

    rows = _qqq_data if _qqq_data is not None else _fetch_qqq_ohlcv()

    if not rows or len(rows) < ma_period:
        return {"regime": "unknown", "qqq_price": None, "qqq_ma20": None, "pct_from_ma": None}

    closes = [float(r["close"]) for r in rows]
    ma = sum(closes[-ma_period:]) / ma_period
    qqq_price = closes[-1]
    pct_from_ma = (qqq_price - ma) / ma

    return {
        "regime": "uptrend" if qqq_price >= ma else "downtrend",
        "qqq_price": qqq_price,
        "qqq_ma20": ma,
        "pct_from_ma": pct_from_ma,
    }


def _fetch_qqq_ohlcv() -> Optional[list[dict[str, Any]]]:
    """Fetch QQQ daily closes from yfinance (3-month window). Returns None on failure."""
    try:
        import yfinance as yf

        hist = yf.download("QQQ", period="3mo", progress=False, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        close = hist["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        rows = [{"close": float(v)} for v in close.dropna().values]
        return rows or None
    except Exception:
        return None
