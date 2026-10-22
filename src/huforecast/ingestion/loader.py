from __future__ import annotations
import pandas as pd
from pathlib import Path

from ..config import TREATMENTS_WORKBOOK, TREATMENTS_SHEET, TREATMENTS_HEADER_ROW

RAW_COLUMNS = [
    "SETOR", "SITUAÇÃO", "DIAS", "POSO", "QTD/DIA", "TOTAL", "RESTA",
    "SUSPENDER", "INÍCIO", "TÉRMINO", "ANTIMICROBIANO", "QTD",
]


def load_raw_treatments(workbook: str | Path | None = None) -> pd.DataFrame:
    path = Path(workbook) if workbook else TREATMENTS_WORKBOOK
    if not path.exists():
        raise FileNotFoundError(
            f"Raw workbook not found: {path}\n"
            "Place the files received from the HU pharmacy in data/raw/ "
            "(see data/raw/README.md)."
        )
    df = pd.read_excel(path, sheet_name=TREATMENTS_SHEET, header=TREATMENTS_HEADER_ROW)
    keep = [c for c in RAW_COLUMNS if c in df.columns]
    return df[keep].copy()
