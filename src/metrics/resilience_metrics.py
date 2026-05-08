"""
Métricas de resiliência para análise de ativos brasileiros em períodos de crise.

Definição operacional de resiliência (utilizada neste projeto):
    Capacidade de um ativo de preservar valor (drawdown baixo) E de recuperar
    ao nível pré-evento em prazo razoável (tempo de recuperação curto).
    Nenhuma dessas dimensões, isolada, caracteriza resiliência completa.

    Referência de janela:
        T-5: 5 dias úteis antes do evento (base da normalização)
        T+0: data do evento
        T+90: 90 dias úteis após o evento (janela curta)
        T+180: 180 dias úteis após o evento (janela longa)

Limitações conhecidas do módulo:
    - "Dias úteis" contados via calendário Mon-Fri (pd.bdate_range);
      feriados brasileiros não são excluídos — a contagem é aproximada.
    - A janela T-5 assume que o ativo não estava em tendência de queda
      antecipada ao evento; em eventos com antecipação de mercado isso
      pode subestimar o drawdown real.
    - Volatilidade pré-evento estimada em 30 dias úteis; sensível a regimes
      de baixa volatilidade artificialmente comprimida.
"""

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PRE_EVENT_LOOKBACK: int = 30
BUSINESS_DAYS_PER_YEAR: int = 252
_T_MINUS_5: int = 5


def _parse_window(janela: Union[str, int]) -> int:
    """Converte janela '90d' / '180d' / int → número de dias úteis."""
    if isinstance(janela, int):
        return janela
    if isinstance(janela, str) and janela.endswith("d"):
        return int(janela[:-1])
    raise ValueError(
        f"Formato de janela não reconhecido: {janela!r}. "
        "Use '90d', '180d' ou um inteiro."
    )


def _window_dates(
    t_event: pd.Timestamp, janela_dias: int
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Retorna (t_start=T-5, t_end=T+janela_dias) em dias úteis."""
    t_start = t_event - pd.tseries.offsets.BusinessDay(_T_MINUS_5)
    t_end = t_event + pd.tseries.offsets.BusinessDay(janela_dias)
    return t_start, t_end


# ─────────────────────────────────────────────────────────────
# Funções de cálculo individuais
# ─────────────────────────────────────────────────────────────

def calculate_drawdown(
    serie: pd.Series,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> dict:
    """
    Calcula o drawdown máximo e o retorno acumulado na janela [t_start, t_end].

    O drawdown máximo mede a maior queda pico-a-vale observada dentro da
    janela, relativa ao pico imediatamente anterior. O retorno acumulado
    mede o resultado total do ativo no período completo da janela.

    Args:
        serie: Série de preços/cotas com DatetimeIndex (sem fusos).
        t_start: Data inicial da janela (tipicamente T-5 dias úteis).
        t_end: Data final da janela (tipicamente T+90 ou T+180 dias úteis).

    Returns:
        dict com:
            drawdown_max (float | None): Drawdown máximo (negativo; ex: -0.15 = -15%).
            data_drawdown_max (Timestamp | None): Data do drawdown máximo.
            retorno_acumulado_janela (float | None): Retorno total do período.

    Limitações conhecidas:
        - Usa preços de fechamento; não captura volatilidade intraday.
        - Não distingue se o drawdown ocorreu antes ou após T+0.
    """
    window = serie.loc[t_start:t_end].dropna()

    if len(window) < 2:
        logger.warning(
            "Janela insuficiente (%d obs) entre %s e %s",
            len(window), t_start.date(), t_end.date(),
        )
        return {
            "drawdown_max": None,
            "data_drawdown_max": None,
            "retorno_acumulado_janela": None,
        }

    base = window.iloc[0]
    if base == 0:
        raise ValueError(
            "Valor inicial da série é zero; normalização impossível. "
            "Verifique os dados para datas próximas a T-5."
        )

    normalized = window / base
    rolling_max = normalized.cummax()
    dd_series = (normalized - rolling_max) / rolling_max

    return {
        "drawdown_max": float(dd_series.min()),
        "data_drawdown_max": dd_series.idxmin(),
        "retorno_acumulado_janela": float(normalized.iloc[-1] - 1.0),
    }


def calculate_recovery_time(
    serie: pd.Series,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> dict:
    """
    Calcula se e quando o ativo recuperou o nível de T-5 dentro da janela.

    A "recuperação" é definida como o retorno ao nível do primeiro dia da
    janela (T-5). O número de dias úteis é contado do ponto de drawdown
    máximo até a data de primeiro retorno ao nível de referência.

    Args:
        serie: Série de preços/cotas com DatetimeIndex.
        t_start: Data de referência T-5 (nível-alvo de recuperação).
        t_end: Limite da janela de observação.

    Returns:
        dict com:
            recuperou (bool): True se voltou ao nível T-5 dentro da janela.
            dias_recuperacao (int | None): Dias úteis do pico de drawdown
                à recuperação (None se não recuperou).
            data_recuperacao (Timestamp | None): Data de recuperação.

    Limitações conhecidas:
        - "Recuperação" usa o nível T-5 como referência, não o pico histórico.
        - Dias úteis contados via pd.bdate_range (Mon-Fri, sem feriados BR).
        - Uma recuperação seguida de nova queda ainda conta como "recuperou".
    """
    window = serie.loc[t_start:t_end].dropna()

    if len(window) < 2:
        return {"recuperou": False, "dias_recuperacao": None, "data_recuperacao": None}

    base_value = window.iloc[0]
    normalized = window / base_value

    rolling_max = normalized.cummax()
    dd_series = (normalized - rolling_max) / rolling_max

    if dd_series.min() >= -1e-8:
        # Nunca houve queda significativa; não precisa recuperar
        return {
            "recuperou": True,
            "dias_recuperacao": 0,
            "data_recuperacao": window.index[0],
        }

    idx_max_dd = dd_series.idxmin()

    # Buscar o primeiro ponto após o drawdown máximo em que normalized >= 1.0
    post_dd = normalized.loc[idx_max_dd:]
    recovery_mask = post_dd >= (1.0 - 1e-8)

    if not recovery_mask.any():
        return {"recuperou": False, "dias_recuperacao": None, "data_recuperacao": None}

    data_recuperacao = post_dd[recovery_mask].index[0]
    dias = max(0, len(pd.bdate_range(idx_max_dd, data_recuperacao)) - 1)

    return {
        "recuperou": True,
        "dias_recuperacao": int(dias),
        "data_recuperacao": data_recuperacao,
    }


def calculate_pre_event_volatility(
    serie: pd.Series,
    t_event: pd.Timestamp,
    lookback: int = PRE_EVENT_LOOKBACK,
) -> float:
    """
    Calcula a volatilidade anualizada dos N dias úteis anteriores ao evento.

    A volatilidade é estimada como desvio padrão dos retornos diários
    multiplicado por sqrt(252). Usa os últimos `lookback` dias com dados
    disponíveis antes de t_event.

    Args:
        serie: Série de preços/cotas.
        t_event: Data do evento (T+0).
        lookback: Número de dias para estimativa (padrão: 30).

    Returns:
        Volatilidade anualizada (ex: 0.20 = 20% ao ano).

    Raises:
        ValueError: Se não houver dados suficientes antes de t_event.

    Limitações conhecidas:
        - 30 dias úteis podem não ser representativos de regimes estruturais.
        - Sensível a outliers pontuais na janela de estimativa.
    """
    pre = serie[serie.index < t_event].tail(lookback)

    if len(pre) < 5:
        raise ValueError(
            f"Dados insuficientes antes de {t_event.date()} para estimar "
            f"volatilidade. Mínimo: 5 obs; encontradas: {len(pre)}."
        )

    returns = pre.pct_change().dropna()
    vol = returns.std() * np.sqrt(BUSINESS_DAYS_PER_YEAR)
    return float(vol)


def calculate_normalized_drawdown(
    serie: pd.Series,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    vol_pre_evento: float,
) -> Optional[float]:
    """
    Drawdown máximo normalizado pela volatilidade anualizada pré-evento.

    Interpreta a severidade da queda em relação ao regime de risco habitual
    do ativo. Um valor de -2.0 indica que a queda máxima observada
    equivale a 2 vezes a volatilidade anual estimada pré-evento.

    Args:
        serie: Série de preços/cotas.
        t_start: Início da janela de análise.
        t_end: Fim da janela de análise.
        vol_pre_evento: Volatilidade anualizada pré-evento.

    Returns:
        Razão drawdown_max / vol_pre_evento, ou None se indisponível.

    Limitações conhecidas:
        - Divide drawdown cumulativo por vol anualizada — não é um z-score
          estatístico estrito; é uma métrica de intensidade relativa.
        - Para ativos com baixíssima vol (ex: CDI), o z-score pode ser
          distorcido por pequenas variações numéricas.
    """
    dd_result = calculate_drawdown(serie, t_start, t_end)
    dd_max = dd_result.get("drawdown_max")

    if dd_max is None:
        return None
    if vol_pre_evento == 0 or vol_pre_evento is None:
        return None

    return float(dd_max / vol_pre_evento)


# ─────────────────────────────────────────────────────────────
# Função wrapper principal
# ─────────────────────────────────────────────────────────────

def analyze_event(
    serie: pd.Series,
    ativo_nome: str,
    t_event: pd.Timestamp,
    janela: Union[str, int] = "90d",
) -> dict:
    """
    Calcula todas as métricas de resiliência para um par ativo × evento.

    Função wrapper que aplica calculate_drawdown, calculate_recovery_time,
    calculate_pre_event_volatility e calculate_normalized_drawdown de forma
    coordenada para uma janela T-5 a T+N.

    Args:
        serie: Série de preços/cotas com DatetimeIndex (sem fusos).
        ativo_nome: Identificador do ativo (ex: 'ibov', 'ima_b5', 'ifmm').
        t_event: Data do evento (T+0).
        janela: Tamanho da janela pós-evento ('90d', '180d' ou int).

    Returns:
        dict com chaves:
            ativo, evento_data, janela, t_start, t_end,
            drawdown_max, data_drawdown_max, retorno_acumulado_janela,
            vol_pre_evento, drawdown_normalizado,
            recuperou, dias_recuperacao, data_recuperacao.
    """
    janela_dias = _parse_window(janela)
    t_start, t_end = _window_dates(t_event, janela_dias)

    # Clipar t_end ao último dado disponível
    if not serie.empty:
        t_end = min(t_end, serie.index[-1])

    logger.debug(
        "[%s] evento=%s janela=%s → %s a %s",
        ativo_nome, t_event.date(), janela, t_start.date(), t_end.date(),
    )

    try:
        vol_pre = calculate_pre_event_volatility(serie, t_event)
    except ValueError as exc:
        logger.warning("[%s] Vol pré-evento: %s", ativo_nome, exc)
        vol_pre = None

    dd = calculate_drawdown(serie, t_start, t_end)
    rec = calculate_recovery_time(serie, t_start, t_end)

    dd_norm = (
        calculate_normalized_drawdown(serie, t_start, t_end, vol_pre)
        if vol_pre is not None
        else None
    )

    return {
        "ativo": ativo_nome,
        "evento_data": t_event,
        "janela": str(janela),
        "t_start": t_start,
        "t_end": t_end,
        **dd,
        "vol_pre_evento": vol_pre,
        "drawdown_normalizado": dd_norm,
        **rec,
    }


# ─────────────────────────────────────────────────────────────
# Testes inline (assert em casos determinísticos)
# ─────────────────────────────────────────────────────────────

def _run_inline_tests() -> None:
    """Executa asserts em casos óbvios para validação rápida do módulo."""

    def _make(values: list, start: str = "2020-01-02") -> pd.Series:
        idx = pd.bdate_range(start=start, periods=len(values))
        return pd.Series(values, index=idx, dtype=float)

    # 1. Série flat → drawdown zero
    s_flat = _make([100.0] * 60)
    dd = calculate_drawdown(s_flat, s_flat.index[0], s_flat.index[-1])
    assert dd["drawdown_max"] == 0.0, f"Esperado 0.0, obtido {dd['drawdown_max']}"
    assert dd["retorno_acumulado_janela"] == 0.0

    # 2. Queda de 10% sem recuperação
    s_drop = _make([100.0] * 10 + [90.0] * 10)
    dd2 = calculate_drawdown(s_drop, s_drop.index[0], s_drop.index[-1])
    assert abs(dd2["drawdown_max"] - (-0.10)) < 1e-6, f"Esperado -0.10, obtido {dd2['drawdown_max']}"

    # 3. Recuperação completa
    s_rec = _make([100.0] * 10 + [80.0] * 5 + [100.0] * 10)
    rec = calculate_recovery_time(s_rec, s_rec.index[0], s_rec.index[-1])
    assert rec["recuperou"] is True
    assert rec["dias_recuperacao"] is not None and rec["dias_recuperacao"] > 0

    # 4. Sem recuperação
    s_no_rec = _make([100.0] * 10 + [80.0] * 10)
    rec2 = calculate_recovery_time(s_no_rec, s_no_rec.index[0], s_no_rec.index[-1])
    assert rec2["recuperou"] is False

    # 5. Volatilidade série constante ≈ 0
    s_const = _make([100.0] * 100)
    t_ev = s_const.index[50]
    vol = calculate_pre_event_volatility(s_const, t_ev, lookback=30)
    assert vol == 0.0, f"Esperado 0.0, obtido {vol}"

    # 6. Drawdown normalizado negativo quando há queda
    s_fall = _make([100.0 + i * 0.1 for i in range(40)] + [75.0] * 20)
    t_ev2 = s_fall.index[35]
    t_s = s_fall.index[30]
    t_e = s_fall.index[-1]
    vol2 = calculate_pre_event_volatility(s_fall, t_ev2, lookback=25)
    nd = calculate_normalized_drawdown(s_fall, t_s, t_e, vol2)
    assert nd is not None and nd < 0, f"Esperado negativo, obtido {nd}"

    print("✓ Todos os testes inline passaram.")


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.WARNING)
    _run_inline_tests()
