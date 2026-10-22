from __future__ import annotations
import pandas as pd


def monthly_demand_by_drug(clean: pd.DataFrame) -> pd.DataFrame:
    base = clean.dropna(subset=["mes_inicio", "principio_ativo"]).copy()
    series = (base.groupby(["principio_ativo", "mes_inicio"])
              .agg(tratamentos=("principio_ativo", "size"),
                   qtd_total=("qtd_total", "sum"))
              .reset_index())
    return series


def total_monthly_demand(series: pd.DataFrame) -> pd.DataFrame:
    m = series.groupby("mes_inicio")["qtd_total"].sum().reset_index()
    m = m[m["mes_inicio"] != "NaT"].sort_values("mes_inicio").reset_index(drop=True)
    return m
