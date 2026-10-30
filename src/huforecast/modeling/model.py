"""Modeling layer — Deliverable 5 first regression model (antimicrobial x month).

The three honest investigations (internal structural features, suspension
classification, derived-internal + external public data) converged on the same
finding: HU-UFPB antimicrobial demand is a **stable, level-driven process** with no
temporal structure nor climate/epidemiology response that can be exploited honestly
in this 19-month window. The best honest predictor for next month's demand of each
item is the item's own historical level.

So the first model is deliberately simple and interpretable: a **linear regression
on the single honest feature `media_item_ate_tm1`** (the item's expanding mean up to
month t-1). It is materialized as a scikit-learn `Pipeline` with a `ColumnTransformer`
(Deliverable 5 requirement), fit on the training months only and evaluated on the
temporal hold-out, on the exact same split and metrics as the baselines
(`baseline_by_item.py`). The expected, honestly reported result is a **tie** with the
trivial per-item baseline (WAPE ~22.5% vs 22.91%) — a legitimate outcome: "if the
model does not beat the baseline, report that — it is a result, not a failure."

No leakage: `media_item_ate_tm1` for month t uses only demand observed up to t-1, so
the feature is strictly causal with respect to the decision cut (end of t-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import INCOMPLETE_MONTH, TEST_MONTHS
from .baseline_by_item import (
    build_item_month_matrix,
    mape_mean,
    select_dense_items,
    wape,
)

SEED = 42

# The single honest feature the three investigations converged on: the item's own
# historical level, known strictly before the decision cut.
FEATURES = ["media_item_ate_tm1"]
TARGET = "qtd_total"


def build_panel(series: pd.DataFrame, drop_month: str = INCOMPLETE_MONTH,
                min_coverage: float = 1.0) -> pd.DataFrame:
    """Long panel (item x month) for the dense items, with the honest level feature.

    `media_item_ate_tm1` is the expanding mean of the item's demand using ONLY months
    strictly before the current one (shift(1) before expanding), so it never sees the
    target of month t. For the first observed month the feature is NaN and is dropped
    from the modeling panel (there is no prior history to average).
    """
    piv = select_dense_items(build_item_month_matrix(series, drop_month), min_coverage)
    months = list(piv.columns)

    long = (piv.reset_index()
               .melt(id_vars="principio_ativo", var_name="mes_inicio",
                     value_name=TARGET))
    # deterministic chronological order within each item
    long["_month_idx"] = long["mes_inicio"].map({m: i for i, m in enumerate(months)})
    long = long.sort_values(["principio_ativo", "_month_idx"]).reset_index(drop=True)

    # expanding mean of PAST months only (strictly t-1 and before)
    long["media_item_ate_tm1"] = (
        long.groupby("principio_ativo")[TARGET]
            .transform(lambda s: s.shift(1).expanding().mean())
    )
    return long, months


def make_pipeline() -> Pipeline:
    """Linear regression wrapped in a Pipeline + ColumnTransformer (E5 requirement).

    The ColumnTransformer standardizes the numeric feature; the scaler is fit on the
    training rows only (inside `fit`) and merely applied to the test rows, so there is
    no information flow from the hold-out into preprocessing.
    """
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), FEATURES)],
        remainder="drop",
    )
    return Pipeline(steps=[
        ("pre", pre),
        ("reg", LinearRegression()),
    ])


def run_model(series: pd.DataFrame, n_test: int = TEST_MONTHS,
              min_coverage: float = 1.0) -> dict:
    """Fit the first regression model on the temporal split and score it like the baselines.

    Returns WAPE and mean MAPE on the hold-out, computed exactly as in
    `baseline_by_item.py`, so the model and the baselines are directly comparable.
    """
    long, months = build_panel(series, min_coverage=min_coverage)
    train_months, test_months = months[:-n_test], months[-n_test:]

    # drop the first-month rows with no prior history (NaN feature)
    model_df = long.dropna(subset=FEATURES).copy()

    train = model_df[model_df["mes_inicio"].isin(train_months)]
    test = model_df[model_df["mes_inicio"].isin(test_months)]

    pipe = make_pipeline()
    pipe.fit(train[FEATURES], train[TARGET])
    pred = pipe.predict(test[FEATURES])
    # demand cannot be negative; clip at zero (does not affect the dense items in practice)
    pred = np.clip(pred, 0.0, None)

    y_true = test[TARGET].to_numpy(float)

    # per-item arrays for the unweighted MAPE (same definition as the baselines)
    items = test["principio_ativo"].to_numpy()
    yt_by_item, yp_by_item = [], []
    for it in pd.unique(items):
        mask = items == it
        yt_by_item.append(y_true[mask])
        yp_by_item.append(pred[mask])

    coef = float(pipe.named_steps["reg"].coef_[0])
    intercept = float(pipe.named_steps["reg"].intercept_)

    return {
        "model": "LinearRegression(media_item_ate_tm1) via Pipeline+ColumnTransformer",
        "seed": SEED,
        "features": FEATURES,
        "n_items": int(test["principio_ativo"].nunique()),
        "train_span": (train_months[0], train_months[-1], len(train_months)),
        "test_span": (test_months[0], test_months[-1], len(test_months)),
        "coef_scaled": coef,
        "intercept": intercept,
        "WAPE": wape(y_true, pred),
        "MAPE_mean": mape_mean(yt_by_item, yp_by_item),
        "residuals": (y_true - pred),
        "items": pd.unique(items).tolist(),
    }


if __name__ == "__main__":
    import json
    import warnings

    warnings.filterwarnings("ignore")
    import huforecast as hf

    art = hf.run_pipeline()
    res = run_model(art["series"])
    printable = {k: v for k, v in res.items() if k not in ("residuals", "items")}
    print(json.dumps(printable, indent=2, default=str, ensure_ascii=False))
