"""
Download de dados do Ibovespa via yfinance.

Ticker utilizado: ^BVSP (Yahoo Finance)
Dados: preços de fechamento ajustados (auto_adjust=True).

Limitações conhecidas:
    - Yahoo Finance pode apresentar dados faltantes ou ajustes retroativos
      sem aviso. Recomendável validar contra fonte oficial (B3) para fins
      de publicação.
    - Preços ajustados incorporam eventos corporativos dos componentes do
      índice; o Ibovespa em si não paga dividendos, mas a metodologia
      de ajuste do Yahoo pode diferir da B3.
    - Feriados brasileiros geram lacunas na série.
"""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

IBOV_TICKER = "^BVSP"


def fetch_ibovespa(
    start_date: str,
    end_date: str,
    ticker: str = IBOV_TICKER,
) -> pd.Series:
    """
    Faz download do Ibovespa via yfinance (fechamento ajustado).

    Args:
        start_date: Data inicial no formato 'YYYY-MM-DD'.
        end_date: Data final no formato 'YYYY-MM-DD' (exclusivo no yfinance).
        ticker: Ticker do Yahoo Finance (padrão: '^BVSP').

    Returns:
        pd.Series com preços de fechamento ajustados, DatetimeIndex tz-naive.

    Raises:
        ValueError: Se o download retornar DataFrame vazio ou sem a coluna 'Close'.
        RuntimeError: Se o yfinance levantar erro inesperado.

    Limitações conhecidas:
        - yfinance não garante série completa retroativa; comparar com B3 se necessário.
        - Endpoint público sem autenticação; sujeito a throttling.
    """
    logger.info("Ibovespa (%s) | %s → %s", ticker, start_date, end_date)

    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Falha no download do Ibovespa ({ticker}) via yfinance: {exc}"
        ) from exc

    if raw is None or raw.empty:
        raise ValueError(
            f"yfinance retornou DataFrame vazio para {ticker} "
            f"no período {start_date}–{end_date}."
        )

    if "Close" not in raw.columns:
        raise ValueError(
            f"Coluna 'Close' não encontrada. Colunas disponíveis: {raw.columns.tolist()}"
        )

    serie = raw["Close"].squeeze()
    serie = serie.rename("ibov")

    # Garantir que o índice seja tz-naive (yfinance às vezes retorna tz-aware)
    if serie.index.tz is not None:
        serie.index = serie.index.tz_localize(None)

    serie = serie.sort_index().dropna()

    logger.info(
        "  → %d obs | %s a %s | último fechamento: %.0f pts",
        len(serie),
        serie.index[0].date(),
        serie.index[-1].date(),
        serie.iloc[-1],
    )
    return serie
