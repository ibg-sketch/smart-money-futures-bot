import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from signals.formatting import format_signal_telegram


def _base_signal(verdict: str = "NO_TRADE", cvd: float = 0.0, cvd_active: bool = False):
    return {
        "symbol": "TESTUSDT",
        "verdict": verdict,
        "confidence": 0.5,
        "regime": "neutral",
        "last_close": 100.0,
        "vwap_ref": 0,
        "volume": {"last": 0, "median": 0, "spike": False},
        "liq_summary": {"long_count": 0, "short_count": 0, "long_usd": 0, "short_usd": 0},
        "components": {"CVD_pos": cvd_active, "CVD_neg": False},
        "cvd": cvd,
        "oi_change": 0,
        "rsi": None,
        "ema_short": None,
        "ema_long": None,
        "atr": None,
    }


def test_cvd_na_rendering_when_missing():
    signal = _base_signal(cvd=0.0)
    text = format_signal_telegram(signal)
    assert "CVD: N/A" in text


def test_cvd_rendering_when_present():
    signal = _base_signal(cvd=150.0, cvd_active=True)
    text = format_signal_telegram(signal)
    assert "CVD: N/A" not in text
    assert "CVD:" in text
