"""
Download de dados do Banco Central do Brasil via API pública SGS.

Série utilizada:
    - Série 12: Taxa CDI (% ao dia)

CDI é tratado como referência de custo de oportunidade — representa o
retorno "sem risco" do mercado brasileiro. Não é uma classe de ativo
comparável com fundos ou índices de renda variável; é a linha de base
contra a qual outros ativos são avaliados.

Limitações conhecidas:
    - A API SGS pode apresentar instabilidade pontual.
    - Feriados nacionais geram lacunas na série (sem interpolação aqui).
    - O endpoint não exige autenticação, mas pode ter rate-limiting.
"""

import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_BCB_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{sid}/dados"
CDI_SERIE_ID = 12
BUSINESS_DAYS_PER_YEAR = 252


def fetch_bcb_series(
    series_id: int,
    start_date: str,
    end_date: str,
    timeout: int = 30,
) -> pd.Series:
    """
    Faz download de uma série temporal do SGS/BCB.

    Args:
        series_id: Código SGS (ex: 12 para CDI).
        start_date: Data inicial no formato 'DD/MM/YYYY'.
        end_date: Data final no formato 'DD/MM/YYYY'.
        timeout: Timeout HTTP em segundos.

    Returns:
        pd.Series com DatetimeIndex e valores numéricos da série.

    Raises:
        requests.HTTPError: Se a API retornar status de erro.
        ValueError: Se os dados retornados estiverem em formato inesperado
                    ou se o período não retornar observações.

    Limitações conhecidas:
        - Formato de data aceito pela API é brasileiro (DD/MM/YYYY).
        - Não há paginação; períodos muito longos podem retornar timeout.
    """
    # Construir URL com slashes não codificados — a API BCB pode rejeitar %2F
    url = (
        _BCB_URL.format(sid=series_id)
        + f"?formato=json&dataInicial={start_date}&dataFinal={end_date}"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    }

    logger.info("BCB SGS série %d | %s → %s", series_id, start_date, end_date)
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    data = resp.json()
    if not data:
        raise ValueError(
            f"BCB série {series_id}: nenhum dado retornado para "
            f"{start_date}–{end_date}. Verifique o período solicitado."
        )

    df = pd.DataFrame(data)
    _required = {"data", "valor"}
    if not _required.issubset(df.columns):
        raise ValueError(
            f"Formato inesperado da API BCB. "
            f"Colunas esperadas: {_required}; obtidas: {set(df.columns)}."
        )

    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

    serie = df.set_index("data")["valor"].rename(f"bcb_serie_{series_id}")
    serie = serie.sort_index().dropna()

    logger.info(
        "  → %d obs | %s a %s",
        len(serie),
        serie.index[0].date(),
        serie.index[-1].date(),
    )
    return serie


def fetch_cdi(start_date: str, end_date: str) -> pd.Series:
    """
    Faz download da taxa CDI diária (BCB SGS série 12).

    A taxa retornada está em % ao dia (ex: 0.0507 = 0,0507% ao dia).
    CDI é utilizado como referência de custo de oportunidade; não possui
    drawdown relevante e não é comparável a ativos de risco.

    Args:
        start_date: Data inicial no formato 'DD/MM/YYYY'.
        end_date: Data final no formato 'DD/MM/YYYY'.

    Returns:
        pd.Series com taxa CDI diária em % ao dia.
    """
    return fetch_bcb_series(CDI_SERIE_ID, start_date, end_date)


def build_cdi_index(
    cdi_rates: pd.Series,
    base_date: pd.Timestamp,
) -> pd.Series:
    """
    Constrói índice acumulado CDI com base 100 a partir de base_date.

    Cada taxa diária (% ao dia) é convertida em fator de crescimento
    diário e acumulada via produto cumulativo (juros compostos diários).
    O resultado representa o crescimento de R$ 100 investidos ao CDI
    sem considerar impostos ou spread.

    Args:
        cdi_rates: Série com taxa CDI diária em % ao dia.
        base_date: Data de referência para base 100 (tipicamente T-5).

    Returns:
        pd.Series com o índice CDI acumulado, base 100 em base_date.

    Raises:
        ValueError: Se não houver dados a partir de base_date.

    Limitações conhecidas:
        - Ignora feriados (gaps na série geram acumulação apenas em dias úteis).
        - Não considera come-cotas ou IR sobre fundos DI.
    """
    rates = cdi_rates[cdi_rates.index >= base_date].copy()
    if rates.empty:
        raise ValueError(
            f"Não há dados de CDI a partir de {base_date.date()}. "
            "Verifique o período de download."
        )

    factors = 1.0 + rates / 100.0
    cumulative = factors.cumprod()
    index = 100.0 * cumulative / cumulative.iloc[0]
    index.name = "cdi_acumulado"
    return index
