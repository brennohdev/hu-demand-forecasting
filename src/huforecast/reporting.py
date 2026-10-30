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


def make_model_residual_figure() -> list:
    """Residual diagnostics for the first regression model (Deliverable 5).

    Two panels on the temporal hold-out: (left) predicted vs actual with the identity
    line; (right) residuals vs predicted. Kept honest — shows exactly where the model
    errs relative to the per-item baseline it ties with.
    """
    import numpy as np

    from .config import TEST_MONTHS
    from .modeling.model import FEATURES, TARGET, build_panel, make_pipeline

    ensure_dirs()
    art = run_pipeline()
    long, months = build_panel(art["series"])
    test_months = months[-TEST_MONTHS:]
    model_df = long.dropna(subset=FEATURES).copy()
    train = model_df[~model_df["mes_inicio"].isin(test_months)]
    test = model_df[model_df["mes_inicio"].isin(test_months)]
    pipe = make_pipeline(); pipe.fit(train[FEATURES], train[TARGET])
    y_pred = np.clip(pipe.predict(test[FEATURES]), 0.0, None)
    y_true = test[TARGET].to_numpy(float)

    plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax1.scatter(y_true, y_pred, color=AZUL, alpha=0.7, s=28)
    ax1.plot([0, lim], [0, lim], color="#999", ls="--", lw=1)
    ax1.set_xlim(0, lim); ax1.set_ylim(0, lim)
    ax1.set_xlabel("actual demand"); ax1.set_ylabel("predicted demand")
    ax1.set_title("Predicted vs actual (hold-out)")

    ax2.scatter(y_pred, y_true - y_pred, color=LARANJA, alpha=0.7, s=28)
    ax2.axhline(0, color="#999", ls="--", lw=1)
    ax2.set_xlabel("predicted demand"); ax2.set_ylabel("residual (actual - pred)")
    ax2.set_title("Residuals vs predicted")

    fig.tight_layout()
    p = FIGURES_DIR / "m1_residuos_modelo.png"
    fig.savefig(p); plt.close()
    return [p]


if __name__ == "__main__":
    for p in make_figures():
        print("saved", p)
    for p in make_model_residual_figure():
        print("saved", p)
