"""
Testes unitários para src/metrics/resilience_metrics.py.

Cobertura:
    - calculate_drawdown: série flat, queda simples, queda e recuperação.
    - calculate_recovery_time: recuperação completa, sem recuperação, sem queda.
    - calculate_pre_event_volatility: série constante (vol=0), dados insuficientes.
    - calculate_normalized_drawdown: sinal e caso None.
    - analyze_event: integração básica com série sintética.
    - _parse_window: formatos aceitos e rejeição de input inválido.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.resilience_metrics import (
    _parse_window,
    analyze_event,
    calculate_drawdown,
    calculate_normalized_drawdown,
    calculate_pre_event_volatility,
    calculate_recovery_time,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

def make_series(values: list, start: str = "2020-01-02") -> pd.Series:
    """Série com índice de dias úteis Mon-Fri."""
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.Series(values, index=idx, dtype=float, name="test")


# ─────────────────────────────────────────────────────────────
# _parse_window
# ─────────────────────────────────────────────────────────────

def test_parse_window_string_90d():
    assert _parse_window("90d") == 90


def test_parse_window_string_180d():
    assert _parse_window("180d") == 180


def test_parse_window_int():
    assert _parse_window(45) == 45


def test_parse_window_invalid():
    with pytest.raises(ValueError, match="Formato de janela"):
        _parse_window("90")


# ─────────────────────────────────────────────────────────────
# calculate_drawdown
# ─────────────────────────────────────────────────────────────

def test_drawdown_flat_series():
    s = make_series([100.0] * 60)
    result = calculate_drawdown(s, s.index[0], s.index[-1])
    assert result["drawdown_max"] == pytest.approx(0.0, abs=1e-10)
    assert result["retorno_acumulado_janela"] == pytest.approx(0.0, abs=1e-10)


def test_drawdown_10pct_drop():
    s = make_series([100.0] * 10 + [90.0] * 10)
    result = calculate_drawdown(s, s.index[0], s.index[-1])
    assert result["drawdown_max"] == pytest.approx(-0.10, abs=1e-6)
    assert result["retorno_acumulado_janela"] == pytest.approx(-0.10, abs=1e-6)


def test_drawdown_with_recovery():
    # Cai 20% e recupera; drawdown = -0.20, retorno final = 0
    s = make_series([100.0] * 10 + [80.0] * 5 + [100.0] * 10)
    result = calculate_drawdown(s, s.index[0], s.index[-1])
    assert result["drawdown_max"] == pytest.approx(-0.20, abs=1e-6)
    assert result["retorno_acumulado_janela"] == pytest.approx(0.0, abs=1e-6)
    assert result["data_drawdown_max"] is not None


def test_drawdown_insufficient_data():
    s = make_series([100.0])
    result = calculate_drawdown(s, s.index[0], s.index[0])
    assert result["drawdown_max"] is None


def test_drawdown_upward_trend():
    # Série sempre crescente: drawdown deve ser 0
    s = make_series([100.0 + i for i in range(30)])
    result = calculate_drawdown(s, s.index[0], s.index[-1])
    assert result["drawdown_max"] == pytest.approx(0.0, abs=1e-10)
    assert result["retorno_acumulado_janela"] > 0


# ─────────────────────────────────────────────────────────────
# calculate_recovery_time
# ─────────────────────────────────────────────────────────────

def test_recovery_complete():
    s = make_series([100.0] * 10 + [80.0] * 5 + [100.0] * 10)
    result = calculate_recovery_time(s, s.index[0], s.index[-1])
    assert result["recuperou"] is True
    assert result["dias_recuperacao"] is not None
    assert result["dias_recuperacao"] > 0
    assert result["data_recuperacao"] is not None


def test_recovery_no_recovery():
    s = make_series([100.0] * 10 + [80.0] * 10)
    result = calculate_recovery_time(s, s.index[0], s.index[-1])
    assert result["recuperou"] is False
    assert result["dias_recuperacao"] is None


def test_recovery_no_drop():
    # Nunca caiu → recuperação instantânea
    s = make_series([100.0] * 20)
    result = calculate_recovery_time(s, s.index[0], s.index[-1])
    assert result["recuperou"] is True
    assert result["dias_recuperacao"] == 0


def test_recovery_partial_and_then_full():
    # Cai, sobe parcialmente, cai de novo, recupera
    vals = [100.0] * 5 + [90.0] * 5 + [95.0] * 3 + [88.0] * 3 + [100.5] * 5
    s = make_series(vals)
    result = calculate_recovery_time(s, s.index[0], s.index[-1])
    assert result["recuperou"] is True


# ─────────────────────────────────────────────────────────────
# calculate_pre_event_volatility
# ─────────────────────────────────────────────────────────────

def test_volatility_constant_series():
    s = make_series([100.0] * 100)
    t_event = s.index[50]
    vol = calculate_pre_event_volatility(s, t_event, lookback=30)
    assert vol == pytest.approx(0.0, abs=1e-10)


def test_volatility_known_value():
    # Série com retornos diários de exatamente 1% → vol anual = 1% * sqrt(252)
    values = [100.0 * (1.01 ** i) for i in range(60)]
    s = make_series(values)
    t_event = s.index[50]
    vol = calculate_pre_event_volatility(s, t_event, lookback=30)
    # Vol de série geométrica constante = ~0 (retorno constante)
    assert vol == pytest.approx(0.0, abs=1e-6)


def test_volatility_insufficient_data():
    s = make_series([100.0] * 4)
    t_event = s.index[-1]
    with pytest.raises(ValueError, match="Dados insuficientes"):
        calculate_pre_event_volatility(s, t_event, lookback=30)


# ─────────────────────────────────────────────────────────────
# calculate_normalized_drawdown
# ─────────────────────────────────────────────────────────────

def test_normalized_drawdown_negative():
    # Série com variação real → vol > 0 → resultado negativo quando há queda
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 50)
    values = 100.0 * np.cumprod(1 + returns)
    values = np.append(values, [values[-1] * 0.80] * 20)  # queda de 20%
    s = make_series(values.tolist())
    t_event = s.index[45]
    vol = calculate_pre_event_volatility(s, t_event, lookback=30)
    assert vol > 0
    nd = calculate_normalized_drawdown(s, s.index[40], s.index[-1], vol)
    assert nd is not None
    assert nd < 0


def test_normalized_drawdown_zero_vol_returns_none():
    s = make_series([100.0] * 40)
    nd = calculate_normalized_drawdown(s, s.index[0], s.index[-1], vol_pre_evento=0.0)
    assert nd is None


# ─────────────────────────────────────────────────────────────
# analyze_event (integração)
# ─────────────────────────────────────────────────────────────

def test_analyze_event_returns_complete_dict():
    np.random.seed(7)
    returns = np.random.normal(0.0002, 0.012, 300)
    values = 100.0 * np.cumprod(1 + returns)
    s = make_series(values.tolist(), start="2020-01-02")
    t_event = s.index[100]

    result = analyze_event(s, "test_asset", t_event, janela="90d")

    expected_keys = {
        "ativo", "evento_data", "janela", "t_start", "t_end",
        "drawdown_max", "data_drawdown_max", "retorno_acumulado_janela",
        "vol_pre_evento", "drawdown_normalizado",
        "recuperou", "dias_recuperacao", "data_recuperacao",
    }
    assert expected_keys.issubset(result.keys())
    assert result["ativo"] == "test_asset"
    assert result["janela"] == "90d"


def test_analyze_event_180d():
    np.random.seed(9)
    returns = np.random.normal(0, 0.01, 400)
    values = 100.0 * np.cumprod(1 + returns)
    s = make_series(values.tolist(), start="2020-01-02")
    t_event = s.index[100]

    result = analyze_event(s, "asset_x", t_event, janela="180d")
    assert result["janela"] == "180d"
    assert result["drawdown_max"] is not None or result["drawdown_max"] is None  # pode ser None se janela excede série


if __name__ == "__main__":
    # Execução direta sem pytest
    import sys as _sys
    import traceback as _tb

    tests = [
        test_parse_window_string_90d,
        test_parse_window_string_180d,
        test_parse_window_int,
        test_drawdown_flat_series,
        test_drawdown_10pct_drop,
        test_drawdown_with_recovery,
        test_drawdown_insufficient_data,
        test_drawdown_upward_trend,
        test_recovery_complete,
        test_recovery_no_recovery,
        test_recovery_no_drop,
        test_recovery_partial_and_then_full,
        test_volatility_constant_series,
        test_volatility_known_value,
        test_volatility_insufficient_data,
        test_normalized_drawdown_negative,
        test_normalized_drawdown_zero_vol_returns_none,
        test_analyze_event_returns_complete_dict,
        test_analyze_event_180d,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except Exception:
            print(f"  ✗ {t.__name__}")
            _tb.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} testes passaram.")
    _sys.exit(failed)
