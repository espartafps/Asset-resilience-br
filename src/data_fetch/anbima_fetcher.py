"""
Download e parsing de séries da ANBIMA: IFMM e IMA-B 5.

Estratégia de download (tentada nesta ordem):
1. Download direto das URLs conhecidas com headers simulando browser.
2. Leitura de arquivo local em data/raw/ (fallback manual).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSTRUÇÕES PARA FALLBACK MANUAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Se o download automático falhar:

IMA-B 5:
  1. Acesse: https://www.anbima.com.br/pt_br/informar/estatisticas/
             precos-e-indices/ima/ima.htm
  2. Procure "Séries Históricas" ou "Download" e baixe o arquivo Excel.
  3. Salve como: data/raw/ima_geral.xlsx  (ou .xls)
  O parser procura automaticamente colunas com "IMA-B 5" no cabeçalho.

IFMM:
  1. Acesse: https://www.anbima.com.br/pt_br/informar/estatisticas/
             fundos-de-investimento/ifmm.htm
  2. Procure o link de séries históricas ou planilha de cotas.
  3. Salve como: data/raw/ifmm.xlsx  (ou .csv)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Limitações conhecidas:
    - URLs da ANBIMA são instáveis e mudam com atualizações do site.
    - Os arquivos Excel costumam ter linhas de metadados antes dos dados;
      o parser usa heurística para localizar o cabeçalho real.
    - O IFMM tem cobertura diária (dias úteis); séries anteriores a 2010
      podem ter disponibilidade reduzida.
    - Valores numéricos nos XLS da ANBIMA podem usar ponto ou vírgula
      como decimal dependendo da versão; o parser tenta ambos.
"""

import io
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DATA_RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
    "Referer": "https://www.anbima.com.br/",
}

# URLs candidatas — em ordem de tentativa. Atualizar caso o site da ANBIMA
# mude a localização dos arquivos.
_IMA_URLS: list[str] = [
    "https://www.anbima.com.br/data/files/8A/A0/D1/4A/E4D748104BC17066E053DE0AE50A9F09/IMA_Geral.xls",
    "https://www.anbima.com.br/informacoes/ima/arq/ima_completo.xls",
    "https://www.anbima.com.br/data/files/8A/A0/D1/4A/E4D748104BC17066E053DE0AE50A9F09/ima_completo.xlsx",
]

_IFMM_URLS: list[str] = [
    "https://www.anbima.com.br/data/files/8A/A0/D1/4A/E4D748104BC17066E053DE0AE50A9F09/IFMM_historico.xlsx",
    "https://www.anbima.com.br/data/files/8A/A0/D1/4A/E4D748104BC17066E053DE0AE50A9F09/IFMM.xlsx",
]

_IMA_LOCAL_NAMES = ["ima_geral.xlsx", "ima_geral.xls", "IMA_Geral.xls", "ima_completo.xlsx"]
_IFMM_LOCAL_NAMES = ["ifmm.xlsx", "ifmm.xls", "IFMM.xlsx", "IFMM_historico.xlsx"]

# Variantes de nome de coluna para identificar o índice correto
_IMA_B5_COL_VARIANTS = ["IMA-B 5", "IMAB5", "IMA-B5", "IMA_B5", "imab5"]
_IFMM_COL_VARIANTS = ["IFMM", "Índice FMM", "ifmm", "indice_fmm", "Número Índice"]
_DATE_COL_VARIANTS = ["Data", "DATA", "data", "Dt Referência", "Dt_Referencia", "Date"]


# ─────────────────────────────────────────────────────────────
# Funções internas de I/O
# ─────────────────────────────────────────────────────────────

def _try_download(urls: list[str], timeout: int = 20) -> Optional[bytes]:
    """Tenta cada URL da lista; retorna conteúdo bytes do primeiro sucesso."""
    for url in urls:
        try:
            logger.debug("  Tentando: %s", url)
            resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=timeout)
            resp.raise_for_status()
            logger.info("  Download bem-sucedido: %s (%d bytes)", url, len(resp.content))
            return resp.content
        except requests.RequestException as exc:
            logger.debug("  Falha em %s: %s", url, exc)
    return None


def _try_local(local_names: list[str]) -> Optional[Path]:
    """Procura pelos nomes de arquivo na pasta data/raw/; retorna o primeiro encontrado."""
    for name in local_names:
        candidate = DATA_RAW_DIR / name
        if candidate.exists():
            logger.info("  Arquivo local encontrado: %s", candidate)
            return candidate
    return None


# ─────────────────────────────────────────────────────────────
# Parser de Excel da ANBIMA
# ─────────────────────────────────────────────────────────────

def _find_header_row(df_raw: pd.DataFrame) -> int:
    """
    Localiza a linha que contém cabeçalhos reais (com variante de 'Data').
    Os arquivos da ANBIMA costumam ter 3–10 linhas de metadados antes dos dados.
    """
    date_variants_lower = {v.lower() for v in _DATE_COL_VARIANTS}
    for i, row in df_raw.iterrows():
        row_strs = [str(v).strip().lower() for v in row.values if pd.notna(v)]
        if any(s in date_variants_lower for s in row_strs):
            return int(i)
    return 0


def _parse_anbima_excel(
    source: bytes | Path,
    target_col_variants: list[str],
    sheet_index: int = 0,
) -> pd.Series:
    """
    Faz parse de um Excel da ANBIMA e extrai a coluna alvo como pd.Series.

    O Excel pode ter N linhas de metadados antes do cabeçalho real.
    O parser usa heurística para localizar o início dos dados.

    Args:
        source: bytes do arquivo baixado, ou Path para arquivo local.
        target_col_variants: Variantes de nome da coluna de interesse.
        sheet_index: Índice da aba do Excel (padrão: primeira).

    Returns:
        pd.Series com DatetimeIndex e valores numéricos do índice.

    Raises:
        ValueError: Se coluna-alvo ou coluna de data não forem encontradas.
    """
    read_kwargs = dict(header=None, sheet_name=sheet_index)
    if isinstance(source, bytes):
        df_raw = pd.read_excel(io.BytesIO(source), **read_kwargs)
    else:
        df_raw = pd.read_excel(source, **read_kwargs)

    header_row = _find_header_row(df_raw)
    df = df_raw.iloc[header_row:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.iloc[0]]
    df = df.iloc[1:].reset_index(drop=True)

    # Localizar coluna de data
    date_col = None
    for variant in _DATE_COL_VARIANTS:
        matches = [c for c in df.columns if variant.lower() in c.lower()]
        if matches:
            date_col = matches[0]
            break
    if date_col is None:
        raise ValueError(
            f"Coluna de data não encontrada. Colunas disponíveis: {df.columns.tolist()}"
        )

    # Localizar coluna do índice alvo
    target_col = None
    for variant in target_col_variants:
        matches = [c for c in df.columns if variant.lower() in c.lower()]
        if matches:
            target_col = matches[0]
            break
    if target_col is None:
        raise ValueError(
            f"Nenhuma coluna compatível com {target_col_variants} encontrada. "
            f"Colunas disponíveis: {df.columns.tolist()}"
        )

    df = df[[date_col, target_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df[target_col] = (
        df[target_col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

    df = df.dropna(subset=[date_col, target_col])
    serie = df.set_index(date_col)[target_col].sort_index()
    serie.index.name = "data"
    return serie


def _parse_anbima_csv(source: Path, target_col_variants: list[str]) -> pd.Series:
    """
    Parse de CSV da ANBIMA. Tenta separadores ; e , automaticamente.
    """
    for sep in [";", ","]:
        try:
            df = pd.read_csv(source, sep=sep, encoding="latin1")
            date_col = next(
                (c for c in df.columns for v in _DATE_COL_VARIANTS if v.lower() in c.lower()),
                None,
            )
            target_col = next(
                (c for c in df.columns for v in target_col_variants if v.lower() in c.lower()),
                None,
            )
            if date_col and target_col:
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
                df[target_col] = (
                    df[target_col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .pipe(pd.to_numeric, errors="coerce")
                )
                df = df.dropna(subset=[date_col, target_col])
                return df.set_index(date_col)[target_col].sort_index()
        except Exception:
            continue
    raise ValueError(
        f"Não foi possível parsear CSV {source} com variantes {target_col_variants}."
    )


# ─────────────────────────────────────────────────────────────
# Funções públicas
# ─────────────────────────────────────────────────────────────

def fetch_ima_b5(
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Faz download ou leitura local do IMA-B 5 (ANBIMA).

    O IMA-B 5 é o índice ANBIMA de NTN-Bs com prazo de até 5 anos.
    Representa a rentabilidade de uma carteira teórica de títulos públicos
    indexados ao IPCA de curta duration — historicamente mais estável
    que o IMA-B total em períodos de stress de taxa de juro.

    Estratégia:
        1. Tenta URLs conhecidas com headers de browser.
        2. Se falhar, busca arquivo local em data/raw/.
        3. Se nenhum disponível, levanta AnbimaDataUnavailableError com instruções.

    Args:
        start_date: Data inicial 'YYYY-MM-DD'.
        end_date: Data final 'YYYY-MM-DD'.

    Returns:
        pd.Series com DatetimeIndex e valores do IMA-B 5, filtrado pelo período.

    Raises:
        AnbimaDataUnavailableError: Se nenhuma fonte estiver disponível.

    Limitações conhecidas:
        - Parser heurístico pode falhar se a ANBIMA alterar o layout do Excel.
        - URLs são sujeitas a mudança; verificar periodicamente.
    """
    logger.info("IMA-B 5 | %s → %s", start_date, end_date)
    serie = _load_anbima_series(
        remote_urls=_IMA_URLS,
        local_names=_IMA_LOCAL_NAMES,
        col_variants=_IMA_B5_COL_VARIANTS,
        series_name="ima_b5",
    )
    return _filter_and_rename(serie, start_date, end_date, "ima_b5")


def fetch_ifmm(
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Faz download ou leitura local do IFMM (Índice de Fundos Multimercado, ANBIMA).

    O IFMM mede o desempenho médio de fundos multimercado livres, ponderado
    por patrimônio. É publicado diariamente em dias úteis pela ANBIMA.

    Estratégia: idêntica ao fetch_ima_b5 (ver docstring).

    Args:
        start_date: Data inicial 'YYYY-MM-DD'.
        end_date: Data final 'YYYY-MM-DD'.

    Returns:
        pd.Series com DatetimeIndex e valores do IFMM, filtrado pelo período.

    Raises:
        AnbimaDataUnavailableError: Se nenhuma fonte estiver disponível.

    Limitações conhecidas:
        - IFMM representa média ponderada de fundos; desempenho de fundos
          individuais pode divergir significativamente.
        - Dados anteriores a 2008 têm cobertura reduzida.
    """
    logger.info("IFMM | %s → %s", start_date, end_date)
    serie = _load_anbima_series(
        remote_urls=_IFMM_URLS,
        local_names=_IFMM_LOCAL_NAMES,
        col_variants=_IFMM_COL_VARIANTS,
        series_name="ifmm",
    )
    return _filter_and_rename(serie, start_date, end_date, "ifmm")


# ─────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────

def _load_anbima_series(
    remote_urls: list[str],
    local_names: list[str],
    col_variants: list[str],
    series_name: str,
) -> pd.Series:
    """Tenta download remoto e, em seguida, leitura local."""
    # Tentativa 1: download remoto
    raw_bytes = _try_download(remote_urls)
    if raw_bytes is not None:
        try:
            return _parse_anbima_excel(raw_bytes, col_variants)
        except Exception as exc:
            logger.warning(
                "Download remoto OK mas parse falhou para %s: %s. "
                "Tentando fallback local.",
                series_name, exc,
            )

    # Tentativa 2: arquivo local
    local_path = _try_local(local_names)
    if local_path is not None:
        suffix = local_path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            return _parse_anbima_excel(local_path, col_variants)
        elif suffix == ".csv":
            return _parse_anbima_csv(local_path, col_variants)
        else:
            raise AnbimaDataUnavailableError(
                f"Extensão não suportada: {suffix}. Use .xlsx, .xls ou .csv."
            )

    # Nenhuma fonte disponível
    instructions = (
        f"\n\nNão foi possível obter dados da ANBIMA para '{series_name}'.\n"
        "Faça o download manual conforme as instruções no início de:\n"
        "  src/data_fetch/anbima_fetcher.py\n"
        f"e salve o arquivo em: data/raw/{local_names[0]}\n"
    )
    raise AnbimaDataUnavailableError(instructions)


def _filter_and_rename(
    serie: pd.Series,
    start_date: str,
    end_date: str,
    name: str,
) -> pd.Series:
    """Filtra pelo período e renomeia a série."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    filtered = serie.loc[start:end].copy()
    filtered.name = name

    if filtered.empty:
        raise ValueError(
            f"Série '{name}' não possui dados entre {start_date} e {end_date}. "
            "Verifique se o arquivo baixado cobre o período necessário."
        )

    logger.info(
        "  → %d obs | %s a %s",
        len(filtered),
        filtered.index[0].date(),
        filtered.index[-1].date(),
    )
    return filtered


class AnbimaDataUnavailableError(RuntimeError):
    """Levantada quando nenhuma fonte de dados ANBIMA está disponível."""
