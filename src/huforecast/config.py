from __future__ import annotations
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` until a directory containing pyproject.toml is found."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # fallback: three levels up from this file (src/huforecast/config.py -> repo)
    return Path(__file__).resolve().parents[2]


REPO_ROOT = find_repo_root()

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = REPO_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figuras"

# Source workbook (treatment sheet) received from the HU pharmacy.
TREATMENTS_WORKBOOK = RAW_DIR / "Planilha CCIH 2026 Atualizado.xlsx"
TREATMENTS_SHEET = "Planilha"
TREATMENTS_HEADER_ROW = 5  # real header lives on row index 5 (0-based)

# Modeling constants
INCOMPLETE_MONTH = "2026-10"  # collection on 2026-10-22, so this month is partial
TEST_MONTHS = 4               # temporal hold-out size


def ensure_dirs() -> None:
    """Create output directories if missing (safe to call repeatedly)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
