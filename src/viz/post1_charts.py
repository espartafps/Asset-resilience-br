"""
Visualizações do Post 1: Resiliência de Ativos Brasileiros em Crises.

Gera 4 figuras:
  1. Séries normalizadas (base 100 em T-5) — 3 subplots por evento.
  2. Heatmap de drawdown máximo — ativos × eventos × janelas.
  3. Barras de tempo de recuperação — ativos × eventos.
  4. Tabela consolidada em PNG — métricas principais.

Estilo: mobile-first, limpo, sem clutter de eixos.
Paleta: azul escuro (#1B3A6B), azul médio (#4A7FB5), azul claro (#7BAFD4),
        cinza (#8E9BAE), vermelho discreto (#C0392B).
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paleta e estilos ────────────────────────────────────────────────
COLORS = {
    "ibov":   "#1B3A6B",
    "ifmm":   "#4A7FB5",
    "ima_b5": "#7BAFD4",
    "cdi":    "#8E9BAE",
    "neg":    "#C0392B",
    "neutral":"#ECF0F1",
    "zero":   "#CCCCCC",
}

ASSET_LABELS = {
    "ibov":   "Ibovespa",
    "ifmm":   "IFMM",
    "ima_b5": "IMA-B 5",
    "cdi":    "CDI (ref.)",
}

EVENT_LABELS = {
    "Covid":      "Covid-19\n(fev/2020)",
    "Ukraine":    "Invasão\nUcrânia\n(fev/2022)",
    "Americanas": "Americanas\n(jan/2023)",
}

_ORDERED_ASSETS  = ["ibov", "ifmm", "ima_b5"]
_ORDERED_EVENTS  = ["Covid", "Ukraine", "Americanas"]

DPI         = 150
FONT_BASE   = 11
TITLE_SIZE  = 13


def _apply_style() -> None:
    mpl.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          FONT_BASE,
        "axes.titlesize":     TITLE_SIZE,
        "axes.labelsize":     FONT_BASE,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "--",
        "figure.dpi":         DPI,
        "legend.fontsize":    9,
        "legend.framealpha":  0.85,
    })


def _save(fig: plt.Figure, path: Optional[Path]) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        logger.info("Figura salva: %s", path)


# ── Figura 1: Séries normalizadas ──────────────────────────────────

def plot_normalized_series(
    series_dict: dict[str, Optional[pd.Series]],
    events: dict,
    cdi_rates: Optional[pd.Series] = None,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Gera gráfico de séries normalizadas (base 100 em T-5).

    3 subplots horizontais, um por evento. Cada subplot contém linhas
    para os ativos disponíveis. O CDI aparece como linha tracejada de
    referência (custo de oportunidade), não como ativo comparável.

    Args:
        series_dict: {'ibov': pd.Series | None, 'ifmm': ..., 'ima_b5': ...}
        events: {'Covid': {'date': Timestamp, 'label': str}, ...}
        cdi_rates: Série de taxas CDI diárias (% ao dia) para construir
                   o índice de referência. Se None, CDI não é plotado.
        output_path: Caminho para salvar a figura (None = não salva).

    Returns:
        Figura matplotlib.
    """
    _apply_style()
    n_ev = len(events)
    fig, axes = plt.subplots(1, n_ev, figsize=(5.2 * n_ev, 6), sharey=False)
    if n_ev == 1:
        axes = [axes]

    for ax, (ev_name, ev_info) in zip(axes, events.items()):
        t_event = ev_info["date"]
        t_start = t_event - pd.tseries.offsets.BusinessDay(5)
        t_end   = t_event + pd.tseries.offsets.BusinessDay(180)

        for asset in _ORDERED_ASSETS:
            s = series_dict.get(asset)
            if s is None or s.empty:
                continue
            win = s.loc[t_start:t_end].dropna()
            if len(win) < 2:
                continue
            norm = 100.0 * win / win.iloc[0]
            ax.plot(
                win.index, norm,
                label=ASSET_LABELS.get(asset, asset),
                color=COLORS[asset],
                linewidth=2.0,
                zorder=3,
            )

        # CDI como referência tracejada
        if cdi_rates is not None:
            from src.data_fetch.bcb_fetcher import build_cdi_index
            try:
                cdi_idx = build_cdi_index(cdi_rates, t_start)
                cdi_win = cdi_idx.loc[t_start:t_end]
                if len(cdi_win) > 1:
                    ax.plot(
                        cdi_win.index, cdi_win,
                        label=ASSET_LABELS["cdi"],
                        color=COLORS["cdi"],
                        linewidth=1.5,
                        linestyle="--",
                        alpha=0.75,
                        zorder=2,
                    )
            except Exception as exc:
                logger.warning("CDI não plotado para %s: %s", ev_name, exc)

        # Linha de referência 100
        ax.axhline(100, color=COLORS["zero"], linewidth=0.9, linestyle=":", zorder=1)

        # Linha vertical do evento
        ax.axvline(t_event, color=COLORS["neg"], linewidth=1.2,
                   linestyle=":", alpha=0.6, zorder=4)

        # Sombra janela curta (T-5 a T+90)
        t_end_90 = t_event + pd.tseries.offsets.BusinessDay(90)
        ax.axvspan(t_start, t_end_90, alpha=0.035, color=COLORS["ibov"], zorder=0)

        title = EVENT_LABELS.get(ev_name, ev_name)
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=8)
        ax.set_xlabel("Data", fontsize=FONT_BASE - 1)
        ax.tick_params(axis="x", rotation=30)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        ax.legend(loc="lower left")

    axes[0].set_ylabel("Nível (base 100 em T-5)", fontsize=FONT_BASE)

    fig.suptitle(
        "Resiliência de Ativos Brasileiros — Séries Normalizadas (base 100 em T-5)",
        fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, output_path)
    return fig


# ── Figura 2: Heatmap de drawdown máximo ──────────────────────────

def plot_drawdown_heatmap(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Gera heatmap de drawdown máximo: ativos × (eventos × janelas).

    A matriz tem as janelas 90d e 180d lado a lado para cada evento,
    permitindo comparar a sensibilidade à duração da janela.

    Args:
        results_df: DataFrame consolidado com colunas:
                    ['ativo', 'evento', 'janela', 'drawdown_max', ...]
        output_path: Caminho para salvar.

    Returns:
        Figura matplotlib.
    """
    _apply_style()

    # Pivotar: linhas = ativo, colunas = (evento, janela)
    pivot = results_df.pivot_table(
        index="ativo",
        columns=["evento", "janela"],
        values="drawdown_max",
        aggfunc="first",
    )

    # Reordenar índices
    asset_order = [a for a in _ORDERED_ASSETS + ["cdi"] if a in pivot.index]
    ev_order    = [e for e in _ORDERED_EVENTS if e in pivot.columns.get_level_values(0)]
    pivot = pivot.reindex(index=asset_order)
    pivot = pivot.reindex(columns=[(e, j) for e in ev_order for j in ["90d", "180d"]], fill_value=np.nan)

    labels_matrix = (pivot * 100).round(1).astype(str).replace("nan", "—")
    labels_matrix = labels_matrix.applymap(
        lambda x: f"{x}%" if x != "—" else x
    )

    fig, ax = plt.subplots(figsize=(13, 4))
    import matplotlib.colors as mcolors

    # Colormap: branco (0%) → vermelho (-40%+)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "dd_map", ["#FFFFFF", "#FFD0CC", "#C0392B"]
    )
    # Valores já negativos; usar vmin/vmax invertidos para colormap
    im = ax.imshow(
        pivot.values.astype(float),
        cmap=cmap,
        vmin=-0.45, vmax=0.0,
        aspect="auto",
    )

    # Eixos
    col_labels = [f"{e}\n{j}" for e, j in pivot.columns]
    row_labels  = [ASSET_LABELS.get(a, a) for a in pivot.index]

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=FONT_BASE)

    # Adicionar linhas divisórias entre eventos
    for x in [1.5, 3.5]:
        ax.axvline(x, color="white", linewidth=2.5)

    # Texto dentro de cada célula
    for r, asset in enumerate(pivot.index):
        for c, col in enumerate(pivot.columns):
            val = pivot.loc[asset, col]
            txt = labels_matrix.loc[asset, col]
            color = "white" if (not np.isnan(val) and val < -0.20) else "#1a1a1a"
            ax.text(c, r, txt, ha="center", va="center",
                    fontsize=FONT_BASE - 1, color=color, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Drawdown máximo", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    ax.set_title(
        "Drawdown Máximo por Ativo, Evento e Janela",
        fontsize=TITLE_SIZE, fontweight="bold", pad=10,
    )
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    plt.tight_layout()
    _save(fig, output_path)
    return fig


# ── Figura 3: Barras de tempo de recuperação ──────────────────────

def plot_recovery_bars(
    results_df: pd.DataFrame,
    janela: str = "180d",
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Gera gráfico de barras com tempo de recuperação por ativo × evento.

    Usa os dados da janela mais longa (180d) para maximizar a chance de
    observar recuperações completas. Barras cinzas indicam "não recuperou"
    dentro da janela.

    Args:
        results_df: DataFrame com ['ativo', 'evento', 'janela',
                                   'recuperou', 'dias_recuperacao'].
        janela: Janela de referência ('90d' ou '180d').
        output_path: Caminho para salvar.

    Returns:
        Figura matplotlib.
    """
    _apply_style()

    df = results_df[
        (results_df["janela"] == janela) &
        (results_df["ativo"].isin(_ORDERED_ASSETS))
    ].copy()

    events = [e for e in _ORDERED_EVENTS if e in df["evento"].values]
    assets = [a for a in _ORDERED_ASSETS if a in df["ativo"].values]
    n_ev   = len(events)
    n_as   = len(assets)

    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(n_ev)
    width = 0.22
    offset = -(n_as - 1) / 2 * width

    max_val = 0
    for i, asset in enumerate(assets):
        vals, colors, hatches = [], [], []
        for ev in events:
            row = df[(df["ativo"] == asset) & (df["evento"] == ev)]
            if row.empty:
                vals.append(0)
                colors.append(COLORS["neutral"])
                hatches.append("")
                continue
            r = row.iloc[0]
            if r["recuperou"] and r["dias_recuperacao"] is not None:
                v = r["dias_recuperacao"]
                c = COLORS[asset]
                h = ""
            else:
                v = int(janela[:-1])  # barra no limite da janela
                c = "#D0D0D0"
                h = "//"
            vals.append(v)
            colors.append(c)
            hatches.append(h)
            max_val = max(max_val, v)

        bars = ax.bar(
            x + offset + i * width, vals,
            width=width * 0.9,
            label=ASSET_LABELS.get(asset, asset),
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels([EVENT_LABELS.get(e, e).replace("\n", " ") for e in events],
                       fontsize=FONT_BASE)
    ax.set_ylabel("Dias úteis até recuperação", fontsize=FONT_BASE)
    ax.set_ylim(0, max_val * 1.18 + 10)

    # Legenda manual: cores por ativo + cinza = "não recuperou"
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[a], label=ASSET_LABELS[a])
        for a in assets
    ]
    legend_elements.append(
        Patch(facecolor="#D0D0D0", hatch="//", label=f"Não recuperou (>{janela})")
    )
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_title(
        f"Tempo de Recuperação ao Nível Pré-Evento (janela {janela})",
        fontsize=TITLE_SIZE, fontweight="bold",
    )
    ax.axhline(0, color=COLORS["zero"], linewidth=0.8)
    plt.tight_layout()
    _save(fig, output_path)
    return fig


# ── Figura 4: Tabela consolidada ──────────────────────────────────

def plot_summary_table(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Gera tabela consolidada como figura PNG com as principais métricas.

    Colunas: Ativo | Evento | DD 90d | DD 180d | DD Normaliz. (90d) |
             Recuperou? | Dias Rec. | Retorno 180d

    Args:
        results_df: DataFrame consolidado de métricas.
        output_path: Caminho para salvar.

    Returns:
        Figura matplotlib.
    """
    _apply_style()

    # Montar tabela pivotada: uma linha por (ativo, evento)
    rows = []
    for asset in _ORDERED_ASSETS:
        for ev in _ORDERED_EVENTS:
            r90  = results_df[(results_df["ativo"] == asset) &
                              (results_df["evento"] == ev) &
                              (results_df["janela"] == "90d")]
            r180 = results_df[(results_df["ativo"] == asset) &
                              (results_df["evento"] == ev) &
                              (results_df["janela"] == "180d")]
            if r90.empty and r180.empty:
                continue

            def _fmt_pct(x) -> str:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return "—"
                return f"{x*100:.1f}%"

            def _fmt_norm(x) -> str:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return "—"
                return f"{x:.2f}σ"

            def _fmt_int(x) -> str:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return "—"
                return str(int(x))

            dd_90  = _fmt_pct(r90.iloc[0]["drawdown_max"]  if not r90.empty  else None)
            dd_180 = _fmt_pct(r180.iloc[0]["drawdown_max"] if not r180.empty else None)
            dd_n   = _fmt_norm(r90.iloc[0]["drawdown_normalizado"] if not r90.empty else None)
            rec    = "Sim" if (not r180.empty and r180.iloc[0]["recuperou"]) else "Não"
            dias   = _fmt_int(r180.iloc[0]["dias_recuperacao"] if not r180.empty else None)
            ret180 = _fmt_pct(r180.iloc[0]["retorno_acumulado_janela"] if not r180.empty else None)

            rows.append([
                ASSET_LABELS.get(asset, asset),
                ev,
                dd_90, dd_180, dd_n, rec, dias, ret180,
            ])

    col_labels = [
        "Ativo", "Evento",
        "DD 90d", "DD 180d", "DD Norm. (90d)",
        "Recup.?", "Dias Rec.", "Retorno 180d",
    ]

    n_rows = len(rows)
    fig_h = max(3.5, 0.45 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(FONT_BASE - 1)
    tbl.scale(1.0, 1.55)

    # Estilizar cabeçalho
    for j in range(len(col_labels)):
        cell = tbl[(0, j)]
        cell.set_facecolor(COLORS["ibov"])
        cell.set_text_props(color="white", fontweight="bold")

    # Colorir células de drawdown negativo
    for i, row_data in enumerate(rows, start=1):
        # DD 90d (col 2) e DD 180d (col 3)
        for col_idx in [2, 3]:
            val_str = row_data[col_idx - 0]
            cell = tbl[(i, col_idx)]
            if val_str != "—" and val_str.startswith("-"):
                cell.set_facecolor("#FDECEA")
        # Zebra
        if i % 2 == 0:
            for j in range(len(col_labels)):
                if tbl[(i, j)].get_facecolor()[0] == 1.0:  # se ainda branco
                    tbl[(i, j)].set_facecolor("#F5F8FA")

    ax.set_title(
        "Tabela Consolidada — Métricas de Resiliência",
        fontsize=TITLE_SIZE, fontweight="bold", pad=16,
    )
    plt.tight_layout()
    _save(fig, output_path)
    return fig
