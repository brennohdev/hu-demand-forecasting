from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import INCOMPLETE_MONTH, TEST_MONTHS
from ..features.timeseries import total_monthly_demand


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def mape(y, yhat) -> float:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    nz = y != 0
    return float(np.mean(np.abs((y[nz] - yhat[nz]) / y[nz])) * 100)


def drop_incomplete_month(m: pd.DataFrame, month: str = INCOMPLETE_MONTH) -> pd.DataFrame:
    return m[m["mes_inicio"] != month].reset_index(drop=True)


def temporal_split(m: pd.DataFrame, n_test: int = TEST_MONTHS):
    return m.iloc[:-n_test].copy(), m.iloc[-n_test:].copy()


def run_baselines(series: pd.DataFrame, n_test: int = TEST_MONTHS) -> dict:
    m = drop_incomplete_month(total_monthly_demand(series))
    train, test = temporal_split(m, n_test=n_test)

    # trivial: train mean repeated over the test horizon
    mean_hat = np.repeat(train["qtd_total"].mean(), len(test))
    trivial = {"MAE": mae(test["qtd_total"], mean_hat),
               "MAPE": mape(test["qtd_total"], mean_hat)}

    # operational: persistence (value of the previous month)
    values = m["qtd_total"].values
    start = len(m) - n_test
    y = values[start:]
    yhat = values[start - 1:len(m) - 1]
    persistence = {"MAE": mae(y, yhat), "MAPE": mape(y, yhat)}

    return {
        "n_months": len(m),
        "train_span": (train["mes_inicio"].iloc[0], train["mes_inicio"].iloc[-1], len(train)),
        "test_span": (test["mes_inicio"].iloc[0], test["mes_inicio"].iloc[-1], len(test)),
        "baseline_trivial_mean": trivial,
        "baseline_operational_persistence": persistence,
    }
