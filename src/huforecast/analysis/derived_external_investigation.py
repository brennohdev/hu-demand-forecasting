"""Third investigation: HONEST extra signal for the demand regression.

Two fronts, same fixed temporal split and metric used all semester:

    train: 2025-03 .. 2026-05   (15 months)
    test : 2026-06 .. 2026-09   (4 months)
    2026-10 excluded (partial collection on 2026-10-22)
    primary metric: WAPE = sum|error| / sum|actual|
    secondary: unweighted mean per-item MAPE
    seed: 42

FRENTE 1 — internal DERIVED features (all lagged to t-1, leakage-safe):
  - hospital-pressure proxies: total active treatments and antimicrobial
    diversity in the hospital in t-1 (shared across items that month);
  - posology profile: mean doses/day from `posologia` (12/12 -> 2) per item in t-1;
  - calendar: month-of-year, quarter (PERMITTED — known for any future month);
  - rolling means of the item's own demand (2 and 3 months, t-1);
  - local trend: demand(t-1) - demand(t-2).
  Plus the interaction item × dominant-sector (static item attribute).

FRENTE 2 — external public data (João Pessoa/PB), from
  huforecast.analysis.external_sources (REALLY downloaded): climate (temp/humidity)
  and arbovirus cases (dengue, chikungunya). Each is tested both at lag-1 (honest:
  known at the t-1 cut) and contemporaneous (t; labelled LEAKAGE for a real
  end-of-t-1 forecast, reported only as a ceiling).

Anti-leakage rule: to predict month t only information available up to the end of
t-1 is allowed. Every feature is classified PERMITIDA / VAZAMENTO in LEAKAGE_TABLE.

Run:
    python -m huforecast.analysis.derived_external_investigation

No absolute paths (uses huforecast.config). Deterministic (SEED = 42).
"""
from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd

from .. import config
from ..analysis import SEED
from . import external_sources as ext

warnings.filterwarnings("ignore")
np.random.seed(SEED)

INCOMPLETE_MONTH = config.INCOMPLETE_MONTH  # "2026-10"
N_TEST = config.TEST_MONTHS                 # 4


# ===========================================================================
# Metrics (identical definitions to modeling.baseline_by_item)
# ===========================================================================
def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def mape_mean_by_group(df: pd.DataFrame, group_col: str, yt: str, yp: str) -> float:
    mapes = []
    for _, g in df.groupby(group_col):
        a = g[yt].to_numpy(float); b = g[yp].to_numpy(float)
        nz = a != 0
        if nz.sum() == 0:
            continue
        mapes.append(np.mean(np.abs((a[nz] - b[nz]) / a[nz])) * 100)
    return float(np.mean(mapes)) if mapes else float("nan")


def mape_vec(y, yh) -> float:
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    nz = y != 0
    return float(np.mean(np.abs((y[nz] - yh[nz]) / y[nz])) * 100)


# ===========================================================================
# Base data
# ===========================================================================
def load_clean() -> pd.DataFrame:
    import huforecast as hf
    return hf.clean_treatments(hf.load_raw_treatments())


def months_excluding_incomplete(clean: pd.DataFrame) -> list[str]:
    ms = sorted(clean["mes_inicio"].dropna().unique())
    return [m for m in ms if m not in (INCOMPLETE_MONTH, "NaT")]


def parse_doses_per_day(poso) -> float:
    """`posologia` interval -> doses/day. '12/12' -> 2, '8/8' -> 3, '24/24' -> 1.

    Takes the first integer as the dosing interval in hours. Non-parseable or
    absurd values (>48h or <=0) return NaN.
    """
    if poso is None or (isinstance(poso, float) and np.isnan(poso)):
        return np.nan
    s = str(poso).strip()
    m = re.match(r"^\s*(\d{1,2})", s)
    if not m:
        return np.nan
    h = int(m.group(1))
    if h <= 0 or h > 48:
        return np.nan
    return 24.0 / h


# ===========================================================================
# FRENTE 1 — internal derived panel (item x month) with lagged features
# ===========================================================================
def build_panel(clean: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    months = months_excluding_incomplete(clean)
    month_set = set(months)
    base = clean.dropna(subset=["principio_ativo", "data_inicio"]).copy()
    base = base[base["mes_inicio"].isin(month_set)]

    # ---- TARGET: qtd_total by item x month (production definition) ----
    tgt = (base.groupby(["principio_ativo", "mes_inicio"])["qtd_total"].sum()
           .reset_index().rename(columns={"mes_inicio": "mes"}))

    # ---- posology profile per item x month (contemporaneous; lagged later) ----
    base["doses_dia"] = base["posologia"].map(parse_doses_per_day)
    poso = (base.groupby(["principio_ativo", "mes_inicio"])["doses_dia"].mean()
            .reset_index().rename(columns={"mes_inicio": "mes",
                                           "doses_dia": "doses_dia_item"}))

    # ---- dominant sector of each item (static attribute, over all history) ----
    def norm_sector(s):
        if pd.isna(s):
            return "OUTRO"
        s = str(s).upper()
        if s.startswith("UTI"):
            return "UTI"
        if "CL" in s and "MED" in s:
            return "CLINICA_MEDICA"
        if s == "DIP":
            return "DIP"
        return "OUTRO"

    base["setor_grp"] = base["setor"].map(norm_sector)
    dom = (base.groupby("principio_ativo")["setor_grp"]
           .agg(lambda s: s.value_counts().idxmax()).rename("setor_dominante"))

    # ---- hospital-pressure proxies per month (shared across items) ----
    # active treatments in each month: expand [inicio, termino] over months.
    act_rows = []
    for row in base.itertuples(index=False):
        start = row.data_inicio
        end = row.data_termino
        if pd.isna(start):
            continue
        p0 = pd.Period(start, freq="M")
        p1 = pd.Period(end, freq="M") if not pd.isna(end) and end >= start else p0
        n = (p1 - p0).n
        for i in range(n + 1):
            mp = str(p0 + i)
            if mp in month_set:
                act_rows.append((mp, row.principio_ativo))
    act = pd.DataFrame(act_rows, columns=["mes", "principio_ativo"])
    pressure = (act.groupby("mes")
                .agg(hosp_ativos=("principio_ativo", "size"),
                     hosp_diversidade=("principio_ativo", "nunique"))
                .reset_index())

    # ---- assemble dense panel: every item x every month ----
    items = sorted(base["principio_ativo"].unique())
    idx = pd.MultiIndex.from_product([items, months], names=["principio_ativo", "mes"])
    panel = pd.DataFrame(index=idx).reset_index()
    panel = panel.merge(tgt, on=["principio_ativo", "mes"], how="left")
    panel["qtd_total"] = panel["qtd_total"].fillna(0.0)
    panel = panel.merge(poso, on=["principio_ativo", "mes"], how="left")
    panel = panel.merge(pressure, on="mes", how="left")
    panel = panel.merge(dom, on="principio_ativo", how="left")

    panel = panel.sort_values(["principio_ativo", "mes"]).reset_index(drop=True)

    # ---- calendar features (PERMITTED: known for any future month) ----
    mdt = pd.to_datetime(panel["mes"] + "-01")
    panel["mes_do_ano"] = mdt.dt.month
    panel["trimestre"] = mdt.dt.quarter

    # ---- lag / rolling / trend of the item's own demand ----
    g = panel.groupby("principio_ativo")
    panel["qtd_total_lag1"] = g["qtd_total"].shift(1)
    panel["qtd_total_lag2"] = g["qtd_total"].shift(2)
    panel["media_item_ate_tm1"] = g["qtd_total"].transform(
        lambda s: s.shift(1).expanding().mean())
    panel["rollmean2_lag1"] = g["qtd_total"].transform(
        lambda s: s.shift(1).rolling(2).mean())
    panel["rollmean3_lag1"] = g["qtd_total"].transform(
        lambda s: s.shift(1).rolling(3).mean())
    panel["tendencia_local"] = panel["qtd_total_lag1"] - panel["qtd_total_lag2"]

    # ---- lagged versions of contemporaneous derived features ----
    panel["doses_dia_item_lag1"] = g["doses_dia_item"].shift(1)
    panel["hosp_ativos_lag1"] = g["hosp_ativos"].shift(1)
    panel["hosp_diversidade_lag1"] = g["hosp_diversidade"].shift(1)

    permitted = [
        "media_item_ate_tm1", "qtd_total_lag1", "rollmean2_lag1", "rollmean3_lag1",
        "tendencia_local", "mes_do_ano", "trimestre",
        "doses_dia_item_lag1", "hosp_ativos_lag1", "hosp_diversidade_lag1",
    ]
    return panel, permitted


# ===========================================================================
# FRENTE 2 — merge external monthly table (lagged + contemporaneous)
# ===========================================================================
EXTERNAL_BASE_COLS = [
    "tempmed", "tempmin", "tempmax", "umidmed", "umidmin", "umidmax",
    "casos_dengue", "casos_chikungunya",
]


def attach_external(panel: pd.DataFrame, external_monthly: pd.DataFrame
                    ) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Merge external monthly data. Build lag-1 (PERMITTED) and contemporaneous
    (LEAKAGE) versions. Returns (panel, permitted_ext, leakage_ext)."""
    if external_monthly.empty:
        return panel, [], []
    cols = [c for c in EXTERNAL_BASE_COLS if c in external_monthly.columns]
    extm = external_monthly[["mes"] + cols].copy()

    # contemporaneous merge (month t) -> LEAKAGE for a real end-of-(t-1) forecast
    panel = panel.merge(extm, on="mes", how="left")
    leak_ext = cols[:]

    # lag-1 version: external values of month t-1, keyed per item (same month grid)
    month_order = sorted(panel["mes"].unique())
    shift_map = {m: (month_order[i - 1] if i > 0 else None)
                 for i, m in enumerate(month_order)}
    ext_lag = extm.copy()
    ext_lag["mes_target"] = ext_lag["mes"].map(
        {prev: cur for cur, prev in shift_map.items() if prev is not None})
    ext_lag = ext_lag.dropna(subset=["mes_target"])
    ext_lag = ext_lag.rename(columns={c: f"{c}_lag1" for c in cols})
    ext_lag = ext_lag[["mes_target"] + [f"{c}_lag1" for c in cols]]
    panel = panel.merge(ext_lag, left_on="mes", right_on="mes_target", how="left")
    panel = panel.drop(columns=["mes_target"])
    perm_ext = [f"{c}_lag1" for c in cols]
    return panel, perm_ext, leak_ext


# ===========================================================================
# Leakage classification (explicit, obligatory)
# ===========================================================================
def build_leakage_table(perm_ext, leak_ext) -> dict:
    t = {
        # internal derived — PERMITIDA (lagged / calendar)
        "media_item_ate_tm1": ("PERMITIDA", "Média histórica do item até t-1 (nível)."),
        "qtd_total_lag1": ("PERMITIDA", "Demanda do item no mês anterior (persistência)."),
        "rollmean2_lag1": ("PERMITIDA", "Média móvel 2 meses da demanda até t-1."),
        "rollmean3_lag1": ("PERMITIDA", "Média móvel 3 meses da demanda até t-1."),
        "tendencia_local": ("PERMITIDA", "Variação demanda(t-1)-demanda(t-2); fechada no corte."),
        "mes_do_ano": ("PERMITIDA", "Mês do calendário; conhecido para qualquer mês futuro."),
        "trimestre": ("PERMITIDA", "Trimestre do calendário; conhecido a priori."),
        "doses_dia_item_lag1": ("PERMITIDA", "Perfil posológico médio do item em t-1."),
        "hosp_ativos_lag1": ("PERMITIDA", "Pressão hospitalar (tratamentos ativos) em t-1."),
        "hosp_diversidade_lag1": ("PERMITIDA", "Diversidade de antimicrobianos em uso em t-1."),
    }
    for c in perm_ext:
        base = c[:-5]
        kind = "clima" if base.startswith(("temp", "umid")) else "epidemia (casos)"
        t[c] = ("PERMITIDA", f"Externo {kind} de João Pessoa no mês t-1; conhecido no corte.")
    for c in leak_ext:
        kind = "clima" if c.startswith(("temp", "umid")) else "epidemia (casos)"
        t[c] = ("VAZAMENTO",
                f"Externo {kind} do próprio mês t: não disponível no fim de t-1 "
                f"para uma previsão real (usar versão _lag1).")
    return t


# ===========================================================================
# Frente C-style diagnostics (correlation, importance, incremental R^2)
# ===========================================================================
def correlation_table(df, feats, target="qtd_total") -> pd.DataFrame:
    rows = []
    sub = df.dropna(subset=[target])
    for c in feats:
        s = sub[[c, target]].dropna()
        if len(s) < 3 or s[c].nunique() < 2:
            pe = sp = np.nan
        else:
            pe = s[c].corr(s[target], method="pearson")
            sp = s[c].corr(s[target], method="spearman")
        rows.append({"feature": c, "pearson": pe, "spearman": sp, "n": len(s)})
    return (pd.DataFrame(rows)
            .sort_values("spearman", key=lambda s: s.abs(), ascending=False)
            .reset_index(drop=True))


def tree_importance(train_df, feats, target="qtd_total") -> pd.DataFrame:
    from sklearn.ensemble import RandomForestRegressor
    sub = train_df.dropna(subset=feats + [target])
    rf = RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)
    rf.fit(sub[feats].to_numpy(float), sub[target].to_numpy(float))
    return (pd.DataFrame({"feature": feats, "importance": rf.feature_importances_})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def incremental_r2(train_df, feats, target="qtd_total") -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression
    sub = train_df.dropna(subset=feats + [target])
    y = sub[target].to_numpy(float)
    remaining, chosen, rows, prev = list(feats), [], [], 0.0
    while remaining:
        best_f, best_r2 = None, -np.inf
        for f in remaining:
            X = sub[chosen + [f]].to_numpy(float)
            r2 = LinearRegression().fit(X, y).score(X, y)
            if r2 > best_r2:
                best_f, best_r2 = f, r2
        chosen.append(best_f); remaining.remove(best_f)
        rows.append({"step": len(chosen), "added": best_f,
                     "cum_r2": best_r2, "delta_r2": best_r2 - prev})
        prev = best_r2
    return pd.DataFrame(rows)


# ===========================================================================
# Temporal evaluation — item x month (dense) and aggregate total monthly
# ===========================================================================
def dense_items(panel, min_cov=1.0) -> list[str]:
    present = (panel.assign(has=(panel["qtd_total"] > 0).astype(int))
               .groupby("principio_ativo")["has"].mean())
    return sorted(present[present >= min_cov].index.tolist())


def _onehot_interaction(df, col="setor_dominante"):
    """One-hot of the item's dominant sector — the item×sector interaction."""
    return pd.get_dummies(df[col].fillna("OUTRO"), prefix="dom").astype(float)


def eval_item_month(panel, feature_sets, months, item_subset=None,
                    with_interaction: set[str] | None = None) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    df = panel if item_subset is None else panel[panel["principio_ativo"].isin(item_subset)]
    train_months, test_months = months[:-N_TEST], months[-N_TEST:]
    train = df[df["mes"].isin(train_months)].copy()
    test = df[df["mes"].isin(test_months)].copy()
    with_interaction = with_interaction or set()

    results = []
    # baseline trivial (item mean over train)
    item_mean = train.groupby("principio_ativo")["qtd_total"].mean()
    tb = test.copy()
    tb["pred"] = tb["principio_ativo"].map(item_mean).fillna(0.0)
    results.append({"model": "baseline_trivial (média do item)", "features": "-",
                    "WAPE": wape(tb["qtd_total"], tb["pred"]),
                    "MAPE_mean": mape_mean_by_group(tb, "principio_ativo", "qtd_total", "pred")})

    estimators = {
        "LinearRegression": lambda: LinearRegression(),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=SEED),
    }
    for name, cols in feature_sets.items():
        tr = train.dropna(subset=cols + ["qtd_total"]).copy()
        te = test.dropna(subset=cols + ["qtd_total"]).copy()
        if len(tr) < 20 or len(te) == 0:
            continue
        Xtr_cols, Xte_cols = tr[cols].copy(), te[cols].copy()
        if name in with_interaction:
            itr = _onehot_interaction(tr); ite = _onehot_interaction(te)
            ite = ite.reindex(columns=itr.columns, fill_value=0.0)
            Xtr_cols = pd.concat([Xtr_cols.reset_index(drop=True), itr.reset_index(drop=True)], axis=1)
            Xte_cols = pd.concat([Xte_cols.reset_index(drop=True), ite.reset_index(drop=True)], axis=1)
        Xtr, ytr = Xtr_cols.to_numpy(float), tr["qtd_total"].to_numpy(float)
        Xte = Xte_cols.to_numpy(float)
        for est_name, make in estimators.items():
            model = make().fit(Xtr, ytr)
            te_e = te.copy()
            te_e["pred"] = np.clip(model.predict(Xte), 0, None)
            results.append({"model": est_name, "features": name,
                            "WAPE": wape(te_e["qtd_total"], te_e["pred"]),
                            "MAPE_mean": mape_mean_by_group(te_e, "principio_ativo", "qtd_total", "pred")})
    return pd.DataFrame(results)


def build_aggregate_table(panel, external_monthly, months) -> pd.DataFrame:
    """Aggregate total monthly demand + external features (lag1 & contemporaneous)."""
    agg = (panel.groupby("mes")["qtd_total"].sum().reset_index()
           .sort_values("mes"))
    agg["qtd_total_lag1"] = agg["qtd_total"].shift(1)
    agg["media_ate_tm1"] = agg["qtd_total"].shift(1).expanding().mean()
    agg["rollmean2_lag1"] = agg["qtd_total"].shift(1).rolling(2).mean()
    agg["rollmean3_lag1"] = agg["qtd_total"].shift(1).rolling(3).mean()
    mdt = pd.to_datetime(agg["mes"] + "-01")
    agg["mes_do_ano"] = mdt.dt.month
    agg["trimestre"] = mdt.dt.quarter
    if not external_monthly.empty:
        cols = [c for c in EXTERNAL_BASE_COLS if c in external_monthly.columns]
        agg = agg.merge(external_monthly[["mes"] + cols], on="mes", how="left")
        for c in cols:
            agg[f"{c}_lag1"] = agg[c].shift(1)
    return agg[agg["mes"].isin(months)].reset_index(drop=True)


def backtest_aggregate(agg, months, clima_cols) -> pd.DataFrame:
    """Expanding-origin one-step-ahead backtest at the aggregate level.

    The fixed 4-month test window can flatter a feature set by chance. This walks
    the origin forward (predict each month from all prior months) to check whether
    an apparent gain is robust. Compares the mean baseline against media+clima_lag1.
    """
    from sklearn.linear_model import LinearRegression
    vals = agg[agg["mes"].isin(months)].sort_values("mes").reset_index(drop=True)
    cols = ["media_ate_tm1"] + [c for c in clima_cols if c in vals.columns]
    y_m, p_m, y_c, p_c = [], [], [], []
    start = max(8, len(vals) - 10)  # leave enough history before the first origin
    for i in range(start, len(vals)):
        tr, te = vals.iloc[:i], vals.iloc[i:i + 1]
        base = tr["qtd_total"].mean()
        y_m.append(te["qtd_total"].iloc[0]); p_m.append(base)
        trc = tr.dropna(subset=cols + ["qtd_total"])
        if len(trc) >= 6 and te[cols].notna().all(axis=1).iloc[0]:
            m = LinearRegression().fit(trc[cols], trc["qtd_total"])
            pred = max(0.0, float(m.predict(te[cols])[0]))
        else:
            pred = base
        y_c.append(te["qtd_total"].iloc[0]); p_c.append(pred)
    return pd.DataFrame([
        {"modelo": "baseline média (expanding)", "n_origens": len(y_m),
         "MAPE": mape_vec(y_m, p_m), "WAPE": wape(y_m, p_m)},
        {"modelo": "media+clima_lag1 (expanding)", "n_origens": len(y_c),
         "MAPE": mape_vec(y_c, p_c), "WAPE": wape(y_c, p_c)},
    ])


def eval_aggregate(agg, feature_sets, months) -> pd.DataFrame:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    train_months, test_months = months[:-N_TEST], months[-N_TEST:]
    train = agg[agg["mes"].isin(train_months)]
    test = agg[agg["mes"].isin(test_months)]
    yte = test["qtd_total"].to_numpy(float)

    rows = []
    mean_hat = np.repeat(train["qtd_total"].mean(), len(test))
    rows.append({"model": "baseline_trivial (média)", "features": "-",
                 "MAPE": mape_vec(yte, mean_hat), "WAPE": wape(yte, mean_hat)})
    # persistence
    full = agg.sort_values("mes")["qtd_total"].to_numpy(float)
    k = len(full) - N_TEST
    rows.append({"model": "baseline_persistência", "features": "mês anterior",
                 "MAPE": mape_vec(full[k:], full[k-1:len(full)-1]),
                 "WAPE": wape(full[k:], full[k-1:len(full)-1])})

    for name, cols in feature_sets.items():
        tr = train.dropna(subset=cols + ["qtd_total"])
        te = test.dropna(subset=cols + ["qtd_total"])
        if len(tr) < 6 or len(te) == 0:
            continue
        Xtr, ytr = tr[cols].to_numpy(float), tr["qtd_total"].to_numpy(float)
        Xte, yt2 = te[cols].to_numpy(float), te["qtd_total"].to_numpy(float)
        for est_name, make in {
            "LinearRegression": lambda: LinearRegression(),
            "RandomForest": lambda: RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
        }.items():
            model = make().fit(Xtr, ytr)
            pred = np.clip(model.predict(Xte), 0, None)
            rows.append({"model": est_name, "features": name,
                         "MAPE": mape_vec(yt2, pred), "WAPE": wape(yt2, pred)})
    return pd.DataFrame(rows)


# ===========================================================================
# Figures
# ===========================================================================
def save_figures(agg, external_monthly, corr_item, panel, months) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    config.ensure_dirs()
    out = config.FIGURES_DIR
    saved = []

    if not external_monthly.empty:
        # G1: aggregate demand vs climate (temp) and humidity
        m = agg[agg["mes"].isin(months)]
        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax1.plot(m["mes"], m["qtd_total"], marker="o", color="#222", label="demanda total")
        ax1.set_ylabel("demanda total (qtd_total)")
        ax1.set_xticklabels(m["mes"], rotation=90, fontsize=7)
        ax2 = ax1.twinx()
        if "tempmed" in m:
            ax2.plot(m["mes"], m["tempmed"], marker="s", color="#d8603b", label="temp média (°C)")
        if "umidmed" in m:
            ax2.plot(m["mes"], m["umidmed"], marker="^", color="#3b7dd8", label="umidade média (%)")
        ax2.set_ylabel("clima (°C / %)")
        ax1.set_title("Demanda mensal agregada vs clima (João Pessoa)")
        fig.legend(loc="upper right", fontsize=8)
        fig.tight_layout(); p = out / "fe1_demanda_vs_clima.png"
        fig.savefig(p, dpi=120); plt.close(fig); saved.append(p.name)

        # G2: aggregate demand vs arbovirus cases
        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax1.plot(m["mes"], m["qtd_total"], marker="o", color="#222", label="demanda total")
        ax1.set_ylabel("demanda total"); ax1.set_xticklabels(m["mes"], rotation=90, fontsize=7)
        ax2 = ax1.twinx()
        if "casos_dengue" in m:
            ax2.plot(m["mes"], m["casos_dengue"], marker="s", color="#2ca02c", label="dengue (casos)")
        if "casos_chikungunya" in m:
            ax2.plot(m["mes"], m["casos_chikungunya"], marker="^", color="#9467bd", label="chikungunya (casos)")
        ax2.set_ylabel("casos")
        ax1.set_title("Demanda mensal agregada vs epidemia (arbovírus, João Pessoa)")
        fig.legend(loc="upper right", fontsize=8)
        fig.tight_layout(); p = out / "fe2_demanda_vs_epidemia.png"
        fig.savefig(p, dpi=120); plt.close(fig); saved.append(p.name)

    # G3: correlation bar of derived + external features vs target (item x month)
    top = corr_item.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#d8603b" if "lag1" not in f and f in ("tempmed","umidmed") else "#3b7dd8"
              for f in top["feature"]]
    ax.barh(top["feature"], top["spearman"], color=colors)
    ax.set_title("Spearman vs demanda (item × mês) — derivadas + externas")
    ax.set_xlabel("correlação de Spearman com qtd_total")
    fig.tight_layout(); p = out / "fe3_corr_derivadas_externas.png"
    fig.savefig(p, dpi=120); plt.close(fig); saved.append(p.name)

    # G4: climate seasonality (monthly temp/humidity profile)
    if not external_monthly.empty and "tempmed" in external_monthly:
        em = external_monthly.copy()
        em["mdoano"] = pd.to_datetime(em["mes"] + "-01").dt.month
        prof = em.groupby("mdoano")[["tempmed", "umidmed"]].mean()
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.plot(prof.index, prof["tempmed"], marker="o", color="#d8603b", label="temp média")
        ax1.set_xlabel("mês do ano"); ax1.set_ylabel("°C")
        ax2 = ax1.twinx(); ax2.plot(prof.index, prof["umidmed"], marker="s", color="#3b7dd8", label="umidade média")
        ax2.set_ylabel("%")
        ax1.set_title("Sazonalidade climática de João Pessoa (média por mês do ano)")
        fig.legend(loc="upper center", fontsize=8)
        fig.tight_layout(); p = out / "fe4_sazonalidade_clima.png"
        fig.savefig(p, dpi=120); plt.close(fig); saved.append(p.name)

    return saved


# ===========================================================================
# Orchestration
# ===========================================================================
def run(verbose: bool = True) -> dict:
    clean = load_clean()
    months = months_excluding_incomplete(clean)

    panel, permitted_internal = build_panel(clean)
    external_monthly, src_results = ext.build_external_monthly(save=True)
    panel, perm_ext, leak_ext = attach_external(panel, external_monthly)

    leak_table = build_leakage_table(perm_ext, leak_ext)
    permitted = permitted_internal + perm_ext

    # ---- diagnostics (item x month) ----
    diag_feats = permitted + leak_ext  # include contemporaneous external for contrast
    corr_item = correlation_table(panel, [c for c in diag_feats if c in panel])
    train_df = panel[panel["mes"].isin(months[:-N_TEST])]
    imp = tree_importance(train_df, [c for c in permitted if c in panel])
    incr = incremental_r2(train_df, [c for c in permitted if c in panel])

    # ---- feature sets (item x month) ----
    internal_struct = ["rollmean2_lag1", "rollmean3_lag1", "tendencia_local",
                       "mes_do_ano", "trimestre", "doses_dia_item_lag1",
                       "hosp_ativos_lag1", "hosp_diversidade_lag1"]
    fs_item = {
        "baseline_media_item": ["media_item_ate_tm1"],
        "media+persist": ["media_item_ate_tm1", "qtd_total_lag1"],
        "media+rolling+trend+cal": ["media_item_ate_tm1", "rollmean2_lag1",
                                    "rollmean3_lag1", "tendencia_local",
                                    "mes_do_ano", "trimestre"],
        "media+pressao+posologia": ["media_item_ate_tm1", "hosp_ativos_lag1",
                                    "hosp_diversidade_lag1", "doses_dia_item_lag1"],
        "derivadas_internas_completo": ["media_item_ate_tm1"] + internal_struct,
    }
    if perm_ext:
        fs_item["media+externas_lag1"] = ["media_item_ate_tm1"] + perm_ext
        fs_item["derivadas+externas_lag1 (honesto completo)"] = (
            ["media_item_ate_tm1"] + internal_struct + perm_ext)
    if leak_ext:
        fs_item["+externas_CONTEMPORANEAS (VAZAMENTO/teto)"] = (
            ["media_item_ate_tm1"] + leak_ext)

    interaction_sets = {"media+rolling+trend+cal", "derivadas_internas_completo",
                        "derivadas+externas_lag1 (honesto completo)"}

    dense = dense_items(panel, 1.0)
    eval_dense = eval_item_month(panel, fs_item, months, item_subset=dense,
                                 with_interaction=interaction_sets)
    eval_all = eval_item_month(panel, fs_item, months, item_subset=None,
                               with_interaction=interaction_sets)

    # ---- aggregate total monthly ----
    agg = build_aggregate_table(panel, external_monthly, months)
    fs_agg = {
        "media": ["media_ate_tm1"],
        "cal (mes+trim)": ["mes_do_ano", "trimestre"],
    }
    if "tempmed_lag1" in agg:
        fs_agg["clima_lag1"] = [c for c in ["tempmed_lag1", "tempmin_lag1",
                                            "tempmax_lag1", "umidmed_lag1"] if c in agg]
        fs_agg["epidemia_lag1"] = [c for c in ["casos_dengue_lag1",
                                               "casos_chikungunya_lag1"] if c in agg]
        fs_agg["media+clima_lag1"] = ["media_ate_tm1"] + fs_agg["clima_lag1"]
        fs_agg["media+clima+epidemia_lag1"] = (["media_ate_tm1"]
            + fs_agg["clima_lag1"] + fs_agg["epidemia_lag1"])
        fs_agg["clima_CONTEMPORANEO (VAZAMENTO/teto)"] = [
            c for c in ["tempmed", "tempmin", "tempmax", "umidmed"] if c in agg]
    eval_agg = eval_aggregate(agg, fs_agg, months)
    clima_cols = [c for c in ["tempmed_lag1", "tempmin_lag1", "tempmax_lag1",
                              "umidmed_lag1"] if c in agg]
    backtest = backtest_aggregate(agg, months, clima_cols)

    # correlation at aggregate level too
    agg_feats = [c for c in ["tempmed", "umidmed", "casos_dengue", "casos_chikungunya",
                             "tempmed_lag1", "umidmed_lag1", "casos_dengue_lag1",
                             "mes_do_ano", "trimestre"] if c in agg]
    corr_agg = correlation_table(agg, agg_feats, target="qtd_total")

    figs = save_figures(agg, external_monthly, corr_item, panel, months)

    result = {
        "months": months,
        "n_dense": len(dense),
        "external_sources": [(r.name, r.ok, r.note or r.error, r.url) for r in src_results],
        "external_monthly": external_monthly,
        "leakage_table": leak_table,
        "corr_item": corr_item,
        "corr_agg": corr_agg,
        "importance": imp,
        "incremental_r2": incr,
        "eval_item_dense": eval_dense,
        "eval_item_all": eval_all,
        "eval_aggregate": eval_agg,
        "backtest_aggregate": backtest,
        "figuras": figs,
    }
    if verbose:
        _print(result)
    return result


def _print(r: dict) -> None:
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 25)
    print("\n================ FONTES EXTERNAS ================")
    for name, ok, note, url in r["external_sources"]:
        print(f"[{'OK ' if ok else 'FAIL'}] {name}: {note}\n        {url}")
    print("\n================ CORRELAÇÃO feature × alvo (item × mês) ================")
    print(r["corr_item"].to_string(index=False))
    print("\n================ CORRELAÇÃO (nível agregado total mensal) ================")
    print(r["corr_agg"].to_string(index=False))
    print("\n================ IMPORTÂNCIA (RandomForest, só ranking) ================")
    print(r["importance"].to_string(index=False))
    print("\n================ R² incremental (forward, linear, in-train) ================")
    print(r["incremental_r2"].to_string(index=False))
    print(f"\n============ AVALIAÇÃO item × mês — DENSOS ({r['n_dense']}) ============")
    print(r["eval_item_dense"].sort_values("WAPE").to_string(index=False))
    print("\n============ AVALIAÇÃO item × mês — TODOS (32) ============")
    print(r["eval_item_all"].sort_values("WAPE").to_string(index=False))
    print("\n============ AVALIAÇÃO total mensal AGREGADO (janela fixa 4 meses) ============")
    print(r["eval_aggregate"].sort_values("MAPE").to_string(index=False))
    print("\n============ BACKTEST expanding-origin (robustez do ganho agregado) ============")
    print(r["backtest_aggregate"].to_string(index=False))
    print("\n================ FIGURAS ================")
    print(r["figuras"])


if __name__ == "__main__":
    run(verbose=True)
