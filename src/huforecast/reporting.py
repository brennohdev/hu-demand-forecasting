"""Reporting layer — generate EDA figures into docs/figuras.

Kept separate from modeling so that plotting never interferes with the pipeline.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import run_pipeline
from .config import FIGURES_DIR, ensure_dirs
from .features.timeseries import total_monthly_demand

AZUL = "#14466e"; LARANJA = "#be6e1e"


def make_figures() -> list:
    """Regenerate the five EDA figures from the clean base. Returns saved paths."""
    ensure_dirs()
    art = run_pipeline()
    clean, series = art["clean"], art["series"]
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
    saved = []

    # 1. top antimicrobials
    top = clean["principio_ativo"].value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top.index, top.values, color=AZUL)
    ax.set_title("Top 12 antimicrobials by number of treatments"); ax.set_xlabel("treatments")
    fig.tight_layout(); p = FIGURES_DIR / "g1_top_antimicrobianos.png"; fig.savefig(p); plt.close(); saved.append(p)

    # 2. monthly demand
    m = total_monthly_demand(series)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(range(len(m)), m["qtd_total"].values, marker="o", color=AZUL)
    ax.set_xticks(range(len(m))); ax.set_xticklabels(m["mes_inicio"], rotation=90, fontsize=6)
    ax.set_title("Total monthly antimicrobial demand"); ax.set_ylabel("quantity")
    fig.tight_layout(); p = FIGURES_DIR / "g2_demanda_mensal.png"; fig.savefig(p); plt.close(); saved.append(p)

    # 3. treatments by sector
    setor = clean["setor"].value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(setor.index, setor.values, color=LARANJA)
    ax.set_title("Treatments by sector (top 10)")
    fig.tight_layout(); p = FIGURES_DIR / "g3_setor.png"; fig.savefig(p); plt.close(); saved.append(p)

    # 4. status
    sit = clean["situacao"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(sit.index, sit.values, color=[AZUL, LARANJA, "#777"][: len(sit)])
    ax.set_title("Treatment status"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); p = FIGURES_DIR / "g4_situacao.png"; fig.savefig(p); plt.close(); saved.append(p)

    # 5. seasonality
    mes = clean["data_inicio"].dt.month.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.bar(mes.index, mes.values, color=AZUL)
    ax.set_title("Treatments started per month of year (seasonality)")
    ax.set_xlabel("month"); ax.set_xticks(range(1, 13))
    fig.tight_layout(); p = FIGURES_DIR / "g5_sazonalidade.png"; fig.savefig(p); plt.close(); saved.append(p)

    return saved


if __name__ == "__main__":
    for p in make_figures():
        print("saved", p)
