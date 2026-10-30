"""Modeling layer — per-item baselines (antimicrobial x month).

Deliverable 5 refines the unit from the aggregate monthly series to
**antimicrobial x month**, because the pharmacy's reorder decision is made item by
item. This module recomputes the trivial and operational baselines in that unit,
on the same temporal split, so the comparison with the model stays honest.

Metrics reported:
  - WAPE (weighted absolute percentage error = sum|error| / sum|actual|): the demand
    forecasting standard when items have very different scales; robust to low-volume
    items and directly meaningful for stock (total units off / total units).
  - MAPE (unweighted mean across items): reported too, but it inflates with rare,
    small-volume items, so WAPE is our primary number.

Item scope: antimicrobials present in every month (dense series). This follows the
ABC-curve logic observed in the data (top items concentrate most of the volume).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import INCOMPLETE_MONTH, TEST_MONTHS


def build_item_month_matrix(series: pd.DataFrame, drop_month: str = INCOMPLETE_MONTH) -> pd.DataFrame:
    """Pivot the (principio_ativo, mes_inicio) series into item x month demand."""
    s = series[series["mes_inicio"] != drop_month].copy()
    piv = (s.pivot_table(index="principio_ativo", columns="mes_inicio",
                         values="qtd_total", aggfunc="sum"))
    piv = piv.reindex(sorted(piv.columns), axis=1)
    return piv


def select_dense_items(piv: pd.DataFrame, min_coverage: float = 1.0) -> pd.DataFrame:
    """Keep items present in at least `min_coverage` fraction of the months."""
    coverage = piv.notna().mean(axis=1)
    kept = piv[coverage >= min_coverage].copy()
    kept = kept.fillna(0.0)
    return kept


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def mape_mean(y_true_by_item, y_pred_by_item) -> float:
    """Unweighted mean of per-item MAPE (reported for comparison)."""
    mapes = []
    for yt, yp in zip(y_true_by_item, y_pred_by_item):
        yt = np.asarray(yt, float); yp = np.asarray(yp, float)
        nz = yt != 0
        if nz.sum() == 0:
            continue
        mapes.append(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100)
    return float(np.mean(mapes)) if mapes else float("nan")


def run_item_baselines(series: pd.DataFrame, n_test: int = TEST_MONTHS,
                       min_coverage: float = 1.0) -> dict:
    """Trivial (per-item train mean) and operational (per-item persistence) baselines."""
    piv = select_dense_items(build_item_month_matrix(series), min_coverage)
    months = list(piv.columns)
    train_months, test_months = months[:-n_test], months[-n_test:]

    train = piv[train_months]
    test = piv[test_months]

    # --- trivial: each item's own training mean, repeated over the test horizon ---
    item_mean = train.mean(axis=1)
    pred_trivial = np.tile(item_mean.values.reshape(-1, 1), (1, n_test))

    # --- operational: persistence (previous month's value) ---
    # prediction for test month t = actual of month t-1 (within the full pivot)
    pred_persist = piv[months[-n_test - 1:-1]].values

    y_true = test.values

    # per-item arrays for the unweighted MAPE
    yt_by_item = [y_true[i] for i in range(y_true.shape[0])]
    ytriv_by_item = [pred_trivial[i] for i in range(pred_trivial.shape[0])]
    ypers_by_item = [pred_persist[i] for i in range(pred_persist.shape[0])]

    return {
        "n_items": int(piv.shape[0]),
        "n_months": len(months),
        "train_span": (train_months[0], train_months[-1], len(train_months)),
        "test_span": (test_months[0], test_months[-1], len(test_months)),
        "baseline_trivial_mean": {
            "WAPE": wape(y_true.ravel(), pred_trivial.ravel()),
            "MAPE_mean": mape_mean(yt_by_item, ytriv_by_item),
        },
        "baseline_operational_persistence": {
            "WAPE": wape(y_true.ravel(), pred_persist.ravel()),
            "MAPE_mean": mape_mean(yt_by_item, ypers_by_item),
        },
    }


if __name__ == "__main__":
    import json, warnings
    warnings.filterwarnings("ignore")
    import huforecast as hf
    art = hf.run_pipeline()
    res = run_item_baselines(art["series"])
    print(json.dumps(res, indent=2, default=str, ensure_ascii=False))
