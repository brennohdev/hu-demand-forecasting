from __future__ import annotations
import pandas as pd
import numpy as np

RENAME = {
    "SETOR": "setor", "SITUAÇÃO": "situacao", "DIAS": "dias_tratamento",
    "POSO": "posologia", "QTD/DIA": "qtd_dia", "TOTAL": "qtd_total",
    "RESTA": "qtd_resta", "SUSPENDER": "suspender", "INÍCIO": "data_inicio",
    "TÉRMINO": "data_termino", "ANTIMICROBIANO": "antimicrobiano", "QTD": "qtd_dispensada",
}
NEAR_EMPTY = ["qtd_resta", "qtd_dispensada"]
DATE_COLS = ["data_inicio", "data_termino"]
NUM_COLS = ["dias_tratamento", "qtd_dia", "qtd_total", "qtd_resta", "qtd_dispensada"]
CAT_COLS = ["setor", "situacao", "antimicrobiano"]


def clean_treatments(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=RENAME).copy()

    df = df[df["antimicrobiano"].notna()].copy()

    for c in DATE_COLS:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in NUM_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_COLS:
        if c in df:
            df[c] = (df[c].astype(str).str.strip().str.upper().replace({"NAN": np.nan}))
    if "setor" in df:
        df["setor"] = df["setor"].str.replace(r"\s+", " ", regex=True).str.strip()

    # binary flag before dropping the mostly-empty source column
    df["suspender_flag"] = df["suspender"].notna().astype(int) if "suspender" in df else 0

    df = df.drop(columns=[c for c in NEAR_EMPTY if c in df])

    # derived features used downstream
    df["principio_ativo"] = df["antimicrobiano"].str.split(",").str[0].str.strip()
    df["mes_inicio"] = df["data_inicio"].dt.to_period("M").astype(str)

    df = df.drop_duplicates()
    df = df[df["data_inicio"].notna()]

    return df.reset_index(drop=True)
