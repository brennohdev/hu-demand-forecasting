"""Feature & formulation investigation (Deliverable 5, read-only EDA).

Goal
----
Measure, with temporal rigor and the project's own baseline/metric, which
combination of TARGET + UNIT + FEATURES has the best chance of beating the
corresponding baseline on the SAME temporal split used all semester:

    train: 2025-03 .. 2026-05  (15 months)
    test : 2026-06 .. 2026-09  (4 months)
    2026-10 excluded (partial collection on 2026-10-22)

Fronts
------
A) Structural features "inside" the demand at the antimicrobial x month unit:
   active treatments, started, ended, sum of qtd_dia, mean duration,
   distinct sectors, sector mix. Built in two timings: contemporaneous (month t,
   LEAKAGE) and lagged (month t-1, PERMITTED).
B) Alternative units/targets: item x month, item x sector x month, total monthly,
   and weekly aggregation.
C) Relationships in several layouts: Pearson & Spearman correlation of each
   feature with the target; feature-feature correlation (multicollinearity);
   tree-based importance ranking (exploratory only); incremental R^2.
D) Synthesis: recommend the formulation with the best chance of beating the
   baseline, with numeric evidence, or state clearly that none beats it.

Anti-leakage rule
-----------------
To predict month t only information available up to the end of t-1 is allowed.
Every candidate feature is classified PERMITIDA / VAZAMENTO explicitly.

Run:
    python -m huforecast.analysis.features_investigation

All randomness seeded. No absolute paths (uses huforecast.config).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..analysis import SEED

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(SEED)

INCOMPLETE_MONTH = config.INCOMPLETE_MONTH  # "2026-10"
N_TEST = config.TEST_MONTHS                 # 4

# Figures: only import matplotlib when we actually plot (keeps the module
# importable in headless/test contexts).


# ===========================================================================
# Metrics (same definitions as modeling.baseline_by_item)
# ===========================================================================
def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def mape_mean_by_group(df: pd.DataFrame, group_col: str,
                       y_true_col: str, y_pred_col: str) -> float:
    """Unweighted mean of per-group MAPE (secondary metric)."""
    mapes = []
    for _, g in df.groupby(group_col):
        yt = g[y_true_col].to_numpy(float)
        yp = g[y_pred_col].to_numpy(float)
        nz = yt != 0
        if nz.sum() == 0:
            continue
        mapes.append(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100)
    return float(np.mean(mapes)) if mapes else float("nan")


# ===========================================================================
# Base data
# ===========================================================================
def load_clean() -> pd.DataFrame:
    """raw -> clean, via the production pipeline (no re-decisions)."""
    import huforecast as hf
    raw = hf.load_raw_treatments()
    clean = hf.clean_treatments(raw)
    return clean


def month_period(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M")


def all_months_excluding_incomplete(clean: pd.DataFrame) -> list[str]:
    months = sorted(clean["mes_inicio"].dropna().unique())
    return [m for m in months if m != INCOMPLETE_MONTH and m != "NaT"]


# ===========================================================================
# Frente A — structural feature panel at antimicrobial x month
# ===========================================================================
# A treatment occupies the interval [data_inicio, data_termino]. We expand each
# treatment over the calendar months its interval covers to compute "active"
# counts, and we use mes_inicio for "started" counts. Demand (target) stays as
# the production definition: qtd_total summed by principio_ativo x mes_inicio.

@dataclass
class Panel:
    df: pd.DataFrame
    months: list[str]
    feature_cols: list[str] = field(default_factory=list)


def _month_list_between(start, end) -> list[pd.Period]:
    if pd.isna(start):
        return []
    if pd.isna(end) or end < start:
        end = start
    pstart = pd.Period(start, freq="M")
    pend = pd.Period(end, freq="M")
    n = (pend - pstart).n
    return [pstart + i for i in range(n + 1)]


def build_item_month_panel(clean: pd.DataFrame) -> Panel:
    """One row per (principio_ativo, month) with the TARGET and structural
    features measured in that same month (contemporaneous timing)."""
    months = all_months_excluding_incomplete(clean)
    month_set = set(months)

    base = clean.dropna(subset=["principio_ativo", "data_inicio"]).copy()

    # --- TARGET + "started/ended in month" features (keyed by mes_inicio) ---
    base["mes_termino"] = month_period(base["data_termino"]).astype(str)

    started = (base.groupby(["principio_ativo", "mes_inicio"])
               .agg(qtd_total=("qtd_total", "sum"),
                    n_iniciados=("principio_ativo", "size"),
                    qtd_dia_iniciados=("qtd_dia", "sum"),
                    dur_media_iniciados=("dias_tratamento", "mean"),
                    n_setores_iniciados=("setor", "nunique"))
               .reset_index()
               .rename(columns={"mes_inicio": "mes"}))

    ended = (base.groupby(["principio_ativo", "mes_termino"])
             .agg(n_encerrados=("principio_ativo", "size"))
             .reset_index()
             .rename(columns={"mes_termino": "mes"}))

    # --- "active in month" features: expand each treatment over its months ---
    active_rows = []
    for row in base.itertuples(index=False):
        for p in _month_list_between(row.data_inicio, row.data_termino):
            mp = str(p)
            if mp in month_set:
                active_rows.append((row.principio_ativo, mp, row.qtd_dia, row.setor))
    act = pd.DataFrame(active_rows, columns=["principio_ativo", "mes", "qtd_dia", "setor"])
    active = (act.groupby(["principio_ativo", "mes"])
              .agg(n_ativos=("principio_ativo", "size"),
                   qtd_dia_ativos=("qtd_dia", "sum"),
                   n_setores_ativos=("setor", "nunique"))
              .reset_index())

    # --- sector mix of started treatments (proportion from key sectors) ---
    def norm_sector(s):
        if pd.isna(s):
            return "OUTRO"
        s = str(s).upper()
        if s.startswith("UTI"):
            return "UTI"
        if "CLÍNICA MÉDICA" in s or "CLINICA MEDICA" in s or "CLINICA MÉDICA" in s:
            return "CLINICA_MEDICA"
        if s == "DIP":
            return "DIP"
        return "OUTRO"

    base["setor_grp"] = base["setor"].map(norm_sector)
    mix = (base.groupby(["principio_ativo", "mes_inicio", "setor_grp"])
           .size().reset_index(name="n"))
    mix_piv = (mix.pivot_table(index=["principio_ativo", "mes_inicio"],
                               columns="setor_grp", values="n", aggfunc="sum")
               .fillna(0.0))
    for col in ["UTI", "CLINICA_MEDICA", "DIP", "OUTRO"]:
        if col not in mix_piv:
            mix_piv[col] = 0.0
    tot = mix_piv[["UTI", "CLINICA_MEDICA", "DIP", "OUTRO"]].sum(axis=1).replace(0, np.nan)
    mix_piv["frac_uti"] = (mix_piv["UTI"] / tot).fillna(0.0)
    mix_piv["frac_clinmed"] = (mix_piv["CLINICA_MEDICA"] / tot).fillna(0.0)
    mix_piv["frac_dip"] = (mix_piv["DIP"] / tot).fillna(0.0)
    mix_piv = (mix_piv[["frac_uti", "frac_clinmed", "frac_dip"]]
               .reset_index().rename(columns={"mes_inicio": "mes"}))

    # --- assemble the dense panel: every item x every month ---
    items = sorted(base["principio_ativo"].unique())
    idx = pd.MultiIndex.from_product([items, months], names=["principio_ativo", "mes"])
    panel = pd.DataFrame(index=idx).reset_index()

    for part in (started, ended, active, mix_piv):
        panel = panel.merge(part, on=["principio_ativo", "mes"], how="left")

    fill0 = ["qtd_total", "n_iniciados", "qtd_dia_iniciados", "n_setores_iniciados",
             "n_encerrados", "n_ativos", "qtd_dia_ativos", "n_setores_ativos",
             "frac_uti", "frac_clinmed", "frac_dip"]
    panel[fill0] = panel[fill0].fillna(0.0)
    panel["dur_media_iniciados"] = panel["dur_media_iniciados"].fillna(0.0)

    panel = panel.sort_values(["principio_ativo", "mes"]).reset_index(drop=True)

    feature_cols = [
        "n_ativos", "n_iniciados", "n_encerrados", "qtd_dia_ativos",
        "qtd_dia_iniciados", "dur_media_iniciados", "n_setores_ativos",
        "n_setores_iniciados", "frac_uti", "frac_clinmed", "frac_dip",
    ]
    return Panel(df=panel, months=months, feature_cols=feature_cols)


def add_lagged_features(panel: Panel) -> Panel:
    """Add t-1 (lagged) versions of every structural feature + the lagged target
    (persistence feature). These are the PERMITTED features to predict month t."""
    df = panel.df.copy()
    lag_cols = panel.feature_cols + ["qtd_total"]
    for c in lag_cols:
        df[f"{c}_lag1"] = df.groupby("principio_ativo")[c].shift(1)
    # item's own expanding mean up to t-1 (leakage-safe "level")
    df["media_item_ate_tm1"] = (
        df.groupby("principio_ativo")["qtd_total"]
          .transform(lambda s: s.shift(1).expanding().mean())
    )
    new_panel = Panel(df=df, months=panel.months, feature_cols=panel.feature_cols)
    new_panel.lagged_cols = [f"{c}_lag1" for c in lag_cols] + ["media_item_ate_tm1"]
    return new_panel


# ===========================================================================
# Leakage classification (explicit)
# ===========================================================================
LEAKAGE_TABLE = {
    # contemporaneous (month t) structural features — known only during/after t
    "n_ativos": ("VAZAMENTO", "Tratamentos ativos no mes t so sao conhecidos durante/apos t."),
    "n_iniciados": ("VAZAMENTO", "Inicios do mes t sao conhecidos durante t, nao no corte (fim de t-1)."),
    "n_encerrados": ("VAZAMENTO", "Encerramentos do mes t ocorrem dentro de t."),
    "qtd_dia_ativos": ("VAZAMENTO", "Soma qtd/dia dos ativos em t depende de tratamentos de t."),
    "qtd_dia_iniciados": ("VAZAMENTO", "Depende dos inicios de t."),
    "dur_media_iniciados": ("VAZAMENTO", "Duracao dos tratamentos iniciados em t; conhecida ao longo/fim de t."),
    "n_setores_ativos": ("VAZAMENTO", "Setores ativos em t dependem de tratamentos de t."),
    "n_setores_iniciados": ("VAZAMENTO", "Setores dos inicios de t."),
    "frac_uti": ("VAZAMENTO", "Mix setorial dos inicios de t."),
    "frac_clinmed": ("VAZAMENTO", "Mix setorial dos inicios de t."),
    "frac_dip": ("VAZAMENTO", "Mix setorial dos inicios de t."),
    # lagged (month t-1) versions — fully known at the cut
    "n_ativos_lag1": ("PERMITIDA", "Ativos no mes anterior; conhecido no corte."),
    "n_iniciados_lag1": ("PERMITIDA", "Inicios do mes anterior; fechado no corte."),
    "n_encerrados_lag1": ("PERMITIDA", "Encerramentos do mes anterior."),
    "qtd_dia_ativos_lag1": ("PERMITIDA", "Carga diaria dos ativos no mes anterior."),
    "qtd_dia_iniciados_lag1": ("PERMITIDA", "Inicios do mes anterior."),
    "dur_media_iniciados_lag1": ("PERMITIDA", "Duracao media dos inicios do mes anterior."),
    "n_setores_ativos_lag1": ("PERMITIDA", "Setores ativos no mes anterior."),
    "n_setores_iniciados_lag1": ("PERMITIDA", "Setores dos inicios do mes anterior."),
    "frac_uti_lag1": ("PERMITIDA", "Mix UTI do mes anterior."),
    "frac_clinmed_lag1": ("PERMITIDA", "Mix clinica medica do mes anterior."),
    "frac_dip_lag1": ("PERMITIDA", "Mix DIP do mes anterior."),
    "qtd_total_lag1": ("PERMITIDA", "Demanda do mes anterior (persistencia)."),
    "media_item_ate_tm1": ("PERMITIDA", "Media historica do item ate t-1 (nivel/baseline trivial)."),
}


# ===========================================================================
# Frente C — correlations, multicollinearity, importance, incremental R^2
# ===========================================================================
def correlation_table(df: pd.DataFrame, feature_cols: list[str],
                      target: str = "qtd_total") -> pd.DataFrame:
    rows = []
    sub = df.dropna(subset=[target])
    for c in feature_cols:
        s = sub[[c, target]].dropna()
        if len(s) < 3 or s[c].nunique() < 2:
            pear = spear = np.nan
        else:
            pear = s[c].corr(s[target], method="pearson")
            spear = s[c].corr(s[target], method="spearman")
        rows.append({"feature": c, "pearson": pear, "spearman": spear,
                     "n": len(s)})
    out = pd.DataFrame(rows).sort_values("spearman", key=lambda s: s.abs(),
                                         ascending=False)
    return out.reset_index(drop=True)


def feature_corr_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].corr(method="spearman")


def tree_importance(train_df: pd.DataFrame, feature_cols: list[str],
                    target: str = "qtd_total") -> pd.DataFrame:
    from sklearn.ensemble import RandomForestRegressor
    sub = train_df.dropna(subset=feature_cols + [target])
    X = sub[feature_cols].to_numpy(float)
    y = sub[target].to_numpy(float)
    rf = RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_})
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def incremental_r2(train_df: pd.DataFrame, feature_cols: list[str],
                   target: str = "qtd_total") -> pd.DataFrame:
    """Greedy forward selection by in-train R^2 (exploratory ranking only)."""
    from sklearn.linear_model import LinearRegression
    sub = train_df.dropna(subset=feature_cols + [target])
    y = sub[target].to_numpy(float)
    remaining = list(feature_cols)
    chosen: list[str] = []
    rows = []
    prev = 0.0
    while remaining:
        best_f, best_r2 = None, -np.inf
        for f in remaining:
            X = sub[chosen + [f]].to_numpy(float)
            lr = LinearRegression().fit(X, y)
            r2 = lr.score(X, y)
            if r2 > best_r2:
                best_f, best_r2 = f, r2
        chosen.append(best_f)
        remaining.remove(best_f)
        rows.append({"step": len(chosen), "added": best_f,
                     "cum_r2": best_r2, "delta_r2": best_r2 - prev})
        prev = best_r2
    return pd.DataFrame(rows)


# ===========================================================================
# Model evaluation on the temporal split (honest comparison)
# ===========================================================================
def dense_items(panel: Panel, min_coverage: float = 1.0) -> list[str]:
    """Items whose demand (qtd_total>0) appears in at least `min_coverage` of
    months — the semester's established scope (20 items at coverage=1.0)."""
    df = panel.df
    present = (df.assign(has=(df["qtd_total"] > 0).astype(int))
               .groupby("principio_ativo")["has"].mean())
    return sorted(present[present >= min_coverage].index.tolist())


def temporal_eval_item_month(panel: Panel, feature_sets: dict[str, list[str]],
                             item_subset: list[str] | None = None) -> pd.DataFrame:
    """Train each feature set on train months, evaluate WAPE/MAPE on test months,
    at the antimicrobial x month unit. Compares against the trivial (item mean)
    and persistence baselines recomputed on the identical rows. If `item_subset`
    is given, restricts to those items (e.g. the dense 20-item scope)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    df = panel.df
    if item_subset is not None:
        df = df[df["principio_ativo"].isin(item_subset)]
    months = panel.months
    train_months = months[:-N_TEST]
    test_months = months[-N_TEST:]

    train = df[df["mes"].isin(train_months)].copy()
    test = df[df["mes"].isin(test_months)].copy()

    results = []

    # --- baselines on exactly these rows ---
    # trivial: item mean over train
    item_mean = train.groupby("principio_ativo")["qtd_total"].mean()
    test_b = test.copy()
    test_b["pred_trivial"] = test_b["principio_ativo"].map(item_mean).fillna(0.0)
    results.append({
        "model": "baseline_trivial (media do item)", "features": "-",
        "WAPE": wape(test_b["qtd_total"], test_b["pred_trivial"]),
        "MAPE_mean": mape_mean_by_group(test_b, "principio_ativo", "qtd_total", "pred_trivial"),
    })
    # operational: persistence via qtd_total_lag1
    if "qtd_total_lag1" in df.columns:
        tb = test.dropna(subset=["qtd_total_lag1"]).copy()
        results.append({
            "model": "baseline_persistencia (mes anterior)", "features": "qtd_total_lag1",
            "WAPE": wape(tb["qtd_total"], tb["qtd_total_lag1"]),
            "MAPE_mean": mape_mean_by_group(tb, "principio_ativo", "qtd_total", "qtd_total_lag1"),
        })

    estimators = {
        "LinearRegression": lambda: LinearRegression(),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=400, random_state=SEED, n_jobs=-1),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=SEED),
    }

    for set_name, cols in feature_sets.items():
        tr = train.dropna(subset=cols + ["qtd_total"])
        te = test.dropna(subset=cols + ["qtd_total"])
        if len(tr) < 20 or len(te) == 0:
            continue
        Xtr, ytr = tr[cols].to_numpy(float), tr["qtd_total"].to_numpy(float)
        Xte = te[cols].to_numpy(float)
        for est_name, make in estimators.items():
            model = make()
            model.fit(Xtr, ytr)
            pred = np.clip(model.predict(Xte), 0, None)
            te_eval = te.copy()
            te_eval["pred"] = pred
            results.append({
                "model": est_name, "features": set_name,
                "WAPE": wape(te_eval["qtd_total"], te_eval["pred"]),
                "MAPE_mean": mape_mean_by_group(te_eval, "principio_ativo", "qtd_total", "pred"),
            })
    return pd.DataFrame(results)


# ===========================================================================
# Frente B — alternative units/targets
# ===========================================================================
def eval_item_sector_month(clean: pd.DataFrame) -> dict:
    """Unit: antimicrobial x sector x month. Target qtd_total. Baseline =
    per-(item,sector) train mean. Reports WAPE and data sparsity."""
    months = all_months_excluding_incomplete(clean)
    base = clean.dropna(subset=["principio_ativo", "data_inicio", "setor"]).copy()
    g = (base.groupby(["principio_ativo", "setor", "mes_inicio"])
         ["qtd_total"].sum().reset_index())
    g = g[g["mes_inicio"].isin(months)]
    items = g[["principio_ativo", "setor"]].drop_duplicates()
    idx = pd.MultiIndex.from_product(
        [range(len(items)), months], names=["pair_id", "mes"])
    pairs = items.reset_index(drop=True)
    panel = pd.DataFrame(index=idx).reset_index()
    panel = panel.merge(pairs.reset_index().rename(columns={"index": "pair_id"}),
                        on="pair_id")
    panel = panel.merge(g, left_on=["principio_ativo", "setor", "mes"],
                        right_on=["principio_ativo", "setor", "mes_inicio"],
                        how="left")
    panel["qtd_total"] = panel["qtd_total"].fillna(0.0)
    train_months = months[:-N_TEST]
    test_months = months[-N_TEST:]
    train = panel[panel["mes"].isin(train_months)]
    test = panel[panel["mes"].isin(test_months)].copy()
    cell_mean = train.groupby("pair_id")["qtd_total"].mean()
    test["pred"] = test["pair_id"].map(cell_mean).fillna(0.0)
    # persistence
    panel = panel.sort_values(["pair_id", "mes"])
    panel["lag1"] = panel.groupby("pair_id")["qtd_total"].shift(1)
    test_p = panel[panel["mes"].isin(test_months)].dropna(subset=["lag1"])
    frac_zero = float((panel["qtd_total"] == 0).mean())
    return {
        "n_cells": int(len(pairs)),
        "frac_zero_cells": frac_zero,
        "baseline_trivial_WAPE": wape(test["qtd_total"], test["pred"]),
        "baseline_persist_WAPE": wape(test_p["qtd_total"], test_p["lag1"]),
    }


def eval_total_monthly(clean: pd.DataFrame) -> dict:
    """Aggregate total monthly demand. Baseline trivial (train mean) / persistence."""
    months = all_months_excluding_incomplete(clean)
    base = clean.dropna(subset=["data_inicio"]).copy()
    m = (base[base["mes_inicio"].isin(months)]
         .groupby("mes_inicio")["qtd_total"].sum().reset_index()
         .sort_values("mes_inicio"))
    vals = m["qtd_total"].to_numpy(float)
    train, test = vals[:-N_TEST], vals[-N_TEST:]
    trivial = np.repeat(train.mean(), len(test))
    persist = vals[len(vals) - N_TEST - 1: len(vals) - 1]

    def mape(y, yh):
        y = np.asarray(y, float); yh = np.asarray(yh, float)
        nz = y != 0
        return float(np.mean(np.abs((y[nz] - yh[nz]) / y[nz])) * 100)
    return {
        "n_months": len(m),
        "baseline_trivial_MAPE": mape(test, trivial),
        "baseline_trivial_WAPE": wape(test, trivial),
        "baseline_persist_MAPE": mape(test, persist),
        "baseline_persist_WAPE": wape(test, persist),
    }


def eval_weekly(clean: pd.DataFrame) -> dict:
    """Weekly aggregation of total demand: more temporal points. Measures lag-1
    autocorrelation and whether persistence/mean baselines are beatable in-sample."""
    base = clean.dropna(subset=["data_inicio"]).copy()
    base = base[base["mes_inicio"] != INCOMPLETE_MONTH]
    base["week"] = month_to_week(base["data_inicio"])
    w = base.groupby("week")["qtd_total"].sum().reset_index().sort_values("week")
    s = w["qtd_total"].to_numpy(float)
    # drop the last (partial) week
    if len(s) > 2:
        s = s[:-1]
    ac1 = float(pd.Series(s).autocorr(lag=1)) if len(s) > 3 else float("nan")
    ac4 = float(pd.Series(s).autocorr(lag=4)) if len(s) > 6 else float("nan")
    return {"n_weeks": int(len(s)), "autocorr_lag1": ac1, "autocorr_lag4": ac4,
            "mean": float(np.mean(s)), "cv": float(np.std(s) / np.mean(s))}


def month_to_week(dt: pd.Series) -> pd.Series:
    d = pd.to_datetime(dt)
    return d.dt.to_period("W-MON").astype(str)


# ===========================================================================
# Figures
# ===========================================================================
def save_figures(panel_lag: Panel, corr_item, corr_matrix_df,
                 clean: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.ensure_dirs()
    outdir = config.FIGURES_DIR
    saved = []

    # F1: feature-target Spearman correlation bar (permitted features)
    perm = corr_item[corr_item["feature"].isin(
        [c for c in corr_item["feature"] if LEAKAGE_TABLE.get(c, ("", ""))[0] == "PERMITIDA"]
    )].copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    perm_sorted = perm.sort_values("spearman")
    ax.barh(perm_sorted["feature"], perm_sorted["spearman"], color="#3b7dd8")
    ax.set_title("Spearman: features PERMITIDAS vs demanda (item x mes)")
    ax.set_xlabel("correlacao de Spearman com qtd_total")
    fig.tight_layout()
    p = outdir / "f1_corr_features_permitidas.png"
    fig.savefig(p, dpi=120); plt.close(fig); saved.append(p.name)

    # F2: correlation heatmap among permitted lagged features
    perm_cols = [c for c in corr_matrix_df.columns
                 if LEAKAGE_TABLE.get(c, ("", ""))[0] == "PERMITIDA"]
    cm = corr_matrix_df.loc[perm_cols, perm_cols]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(perm_cols))); ax.set_xticklabels(perm_cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(perm_cols))); ax.set_yticklabels(perm_cols, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Multicolinearidade (Spearman) entre features PERMITIDAS")
    fig.tight_layout()
    p = outdir / "f2_matriz_correlacao.png"
    fig.savefig(p, dpi=120); plt.close(fig); saved.append(p.name)

    # F3: scatter of the strongest permitted feature vs target
    strongest = perm.iloc[0]["feature"] if len(perm) else "qtd_total_lag1"
    d = panel_lag.df.dropna(subset=[strongest, "qtd_total"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(d[strongest], d["qtd_total"], s=8, alpha=0.3, color="#d8603b")
    ax.set_xlabel(strongest); ax.set_ylabel("qtd_total (demanda do mes t)")
    ax.set_title(f"Dispersao: {strongest} (t-1) vs demanda (t)")
    fig.tight_layout()
    p = outdir / "f3_dispersao_feature_mais_forte.png"
    fig.savefig(p, dpi=120); plt.close(fig); saved.append(p.name)

    # F4: series of top items
    top = (clean[clean["mes_inicio"] != INCOMPLETE_MONTH]
           .groupby("principio_ativo")["qtd_total"].sum()
           .sort_values(ascending=False).head(5).index.tolist())
    months = all_months_excluding_incomplete(clean)
    fig, ax = plt.subplots(figsize=(10, 6))
    for it in top:
        s = (clean[(clean["principio_ativo"] == it) &
                   (clean["mes_inicio"].isin(months))]
             .groupby("mes_inicio")["qtd_total"].sum().reindex(months).fillna(0))
        ax.plot(months, s.values, marker="o", label=it, linewidth=1.5)
    ax.set_xticklabels(months, rotation=90, fontsize=7)
    ax.axvline(len(months) - N_TEST - 0.5, color="gray", ls="--", label="corte treino/teste")
    ax.set_title("Series mensais dos 5 itens de maior volume")
    ax.set_ylabel("qtd_total"); ax.legend(fontsize=7)
    fig.tight_layout()
    p = outdir / "f4_series_top_itens.png"
    fig.savefig(p, dpi=120); plt.close(fig); saved.append(p.name)

    return saved


# ===========================================================================
# Orchestration
# ===========================================================================
def run(verbose: bool = True) -> dict:
    clean = load_clean()
    panel = build_item_month_panel(clean)
    panel_lag = add_lagged_features(panel)

    permitted = [c for c in panel_lag.lagged_cols
                 if LEAKAGE_TABLE.get(c, ("", ""))[0] == "PERMITIDA"]
    leak = panel.feature_cols  # contemporaneous = leakage

    # Frente C: correlations (both permitted and leakage, for contrast)
    all_feats = leak + permitted
    corr_item = correlation_table(panel_lag.df, all_feats)
    corr_matrix = feature_corr_matrix(panel_lag.df.dropna(subset=permitted), permitted + ["qtd_total"])

    months = panel_lag.months
    train_months = months[:-N_TEST]
    train_df = panel_lag.df[panel_lag.df["mes"].isin(train_months)]

    imp = tree_importance(train_df.dropna(subset=permitted), permitted)
    incr = incremental_r2(train_df, permitted)

    # Model evaluation on the split
    feature_sets = {
        "SO_PERMITIDAS_estruturais": [c for c in permitted
                                      if c not in ("qtd_total_lag1", "media_item_ate_tm1")],
        "PERMITIDAS_+_media_item": [c for c in permitted if c != "qtd_total_lag1"],
        "PERMITIDAS_completo": permitted,
        "SO_media_item": ["media_item_ate_tm1"],
        "SO_persistencia": ["qtd_total_lag1"],
        "VAZAMENTO_contemporaneo(teto)": leak,  # upper bound reference, leaky
    }
    eval_tbl = temporal_eval_item_month(panel_lag, feature_sets)
    dense = dense_items(panel_lag, 1.0)
    eval_tbl_dense = temporal_eval_item_month(panel_lag, feature_sets, item_subset=dense)

    # Frente B
    frente_b = {
        "item_x_mes (atual)": {
            "baseline_trivial_WAPE": float(eval_tbl.loc[
                eval_tbl["model"].str.startswith("baseline_trivial"), "WAPE"].iloc[0]),
        },
        "item_x_setor_x_mes": eval_item_sector_month(clean),
        "total_mensal": eval_total_monthly(clean),
        "semanal_agregado": eval_weekly(clean),
    }

    figs = save_figures(panel_lag, corr_item, corr_matrix, clean)

    result = {
        "corr_item": corr_item,
        "corr_matrix": corr_matrix,
        "importance": imp,
        "incremental_r2": incr,
        "eval_item_month": eval_tbl,
        "eval_item_month_dense": eval_tbl_dense,
        "n_dense_items": len(dense),
        "frente_b": frente_b,
        "figuras": figs,
        "permitted_features": permitted,
        "leakage_features": leak,
    }

    if verbose:
        _print(result)
    return result


def _print(r: dict) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n================ FRENTE C: correlacao feature x alvo ================")
    print(r["corr_item"].to_string(index=False))
    print("\n================ FRENTE C: importancia (RandomForest, so ranking) ====")
    print(r["importance"].to_string(index=False))
    print("\n================ FRENTE C: R^2 incremental (forward, linear) =========")
    print(r["incremental_r2"].to_string(index=False))
    print("\n================ AVALIACAO item x mes - TODOS (32) itens =============")
    print(r["eval_item_month"].sort_values("WAPE").to_string(index=False))
    print(f"\n========= AVALIACAO item x mes - itens DENSOS ({r['n_dense_items']}) =========")
    print(r["eval_item_month_dense"].sort_values("WAPE").to_string(index=False))
    print("\n================ FRENTE B: unidades/alvos alternativos ===============")
    print(json.dumps(r["frente_b"], indent=2, default=float, ensure_ascii=False))
    print("\n================ FIGURAS salvas ======================================")
    print(r["figuras"])


if __name__ == "__main__":
    run(verbose=True)
