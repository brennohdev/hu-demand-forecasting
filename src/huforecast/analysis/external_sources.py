"""External public-data sources for the demand regression (Deliverable 5, read-only).

This module REALLY downloads public data and aligns it to the monthly modeling
window (2025-03 .. 2026-09, João Pessoa / Paraíba, IBGE geocode 2507507). It does
not fabricate anything: if a source is unavailable it records the URL and the
error and returns an empty frame, so the investigation can proceed honestly with
whatever downloaded.

Sources attempted
-----------------
1. InfoDengue (Fiocruz) — https://info.dengue.mat.br/
   Weekly municipal series for João Pessoa. Besides arbovirus case counts it
   bundles the local climate (temp min/med/max, humidity min/med/max) derived
   from satellite/station data, which doubles as our CLIMATE source. Reliable
   JSON API, geocode-specific. This is the source that actually feeds the model.

2. INMET (climate) — https://bdmep.inmet.gov.br/ and https://apitempo.inmet.gov.br/
   Attempted for the João Pessoa automatic station (A320). The public apitempo
   endpoint returned empty bodies for recent dates (it has moved behind
   tokens / portal exports), so INMET was NOT used; InfoDengue's bundled climate
   replaces it. Documented, not fabricated.

3. InfoGripe / Fiocruz SRAG — https://gitlab.procc.fiocruz.br/mave/repo
   Attempted for respiratory syndrome (SARI) by UF=PB. The GitLab raw CSV
   endpoints timed out repeatedly (>120s), so InfoGripe was NOT used. Documented.

Caching: downloaded public aggregates are saved under data/external/ (small,
monthly, non-sensitive — may be versioned). The hospital data in data/raw/ is
never touched here.

All functions are deterministic given the upstream data; no randomness.
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config

# João Pessoa / PB, IBGE municipality geocode.
JOAO_PESSOA_GEOCODE = 2507507
_UA = {"User-Agent": "Mozilla/5.0 (huforecast academic research)"}
_TIMEOUT = 60


@dataclass
class SourceResult:
    """Outcome of one external-source attempt (success or documented failure)."""
    name: str
    url: str
    ok: bool
    monthly: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# InfoDengue (epidemiology + bundled climate)
# ---------------------------------------------------------------------------
def _infodengue_url(disease: str, ey_start: int = 2025, ey_end: int = 2026) -> str:
    return (
        "https://info.dengue.mat.br/api/alertcity"
        f"?geocode={JOAO_PESSOA_GEOCODE}&disease={disease}&format=json"
        f"&ew_start=1&ew_end=53&ey_start={ey_start}&ey_end={ey_end}"
    )


def _fetch_json(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode())


def fetch_infodengue(disease: str) -> SourceResult:
    """Weekly InfoDengue series for João Pessoa, aggregated to month.

    Weekly rows carry `data_iniSE` (epi-week start, epoch ms). We map each week to
    the calendar month of its start date and aggregate: cases summed, climate
    averaged. Columns kept: casos (+casos_est), tempmed/min/max, umidmed/min/max.
    """
    url = _infodengue_url(disease)
    try:
        data = _fetch_json(url)
    except Exception as e:  # network / HTTP / JSON
        return SourceResult(f"infodengue_{disease}", url, False,
                            error=f"{type(e).__name__}: {e}")
    if not data:
        return SourceResult(f"infodengue_{disease}", url, False,
                            error="empty response (no rows for this disease/period)")

    df = pd.DataFrame(data)
    # epi-week start date -> month
    df["week_start"] = pd.to_datetime(df["data_iniSE"], unit="ms")
    df["mes"] = df["week_start"].dt.to_period("M").astype(str)

    num = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    df["_casos"] = num("casos")
    df["_casos_est"] = num("casos_est")
    for c in ["tempmed", "tempmin", "tempmax", "umidmed", "umidmin", "umidmax"]:
        df[c] = num(c)
    # Epidemiological-pressure signals carried by InfoDengue (weekly -> month).
    # `nivel` is the municipal alert level (1=green .. 4=red); monthly value is the
    # worst (max) level reached in the month. `Rt` is the reproduction number
    # (mean). `p_inc100k` is incidence per 100k (summed). These are the aggregate
    # pressure signals we test as honest lag-1 external features.
    for c in ["nivel", "Rt", "p_inc100k", "receptivo", "transmissao"]:
        df[c] = num(c)

    agg = (df.groupby("mes")
           .agg(casos=("_casos", "sum"),
                casos_est=("_casos_est", "sum"),
                n_semanas=("mes", "size"),
                tempmed=("tempmed", "mean"),
                tempmin=("tempmin", "mean"),
                tempmax=("tempmax", "mean"),
                umidmed=("umidmed", "mean"),
                umidmin=("umidmin", "mean"),
                umidmax=("umidmax", "mean"),
                nivel=("nivel", "max"),
                Rt=("Rt", "mean"),
                inc100k=("p_inc100k", "sum"),
                receptivo=("receptivo", "max"),
                transmissao=("transmissao", "max"))
           .reset_index())
    agg = agg.rename(columns={
        "casos": f"casos_{disease}",
        "casos_est": f"casos_est_{disease}",
        "nivel": f"nivel_{disease}",
        "Rt": f"Rt_{disease}",
        "inc100k": f"inc100k_{disease}",
        "receptivo": f"receptivo_{disease}",
        "transmissao": f"transmissao_{disease}",
    })
    return SourceResult(f"infodengue_{disease}", url, True, monthly=agg,
                        note=f"{len(df)} weeks -> {len(agg)} months")


def fetch_climate_from_infodengue(base: SourceResult) -> SourceResult:
    """Extract the climate block (shared across diseases) from a dengue pull."""
    if not base.ok:
        return SourceResult("climate_infodengue", base.url, False,
                            error="dengue pull failed; no climate available")
    clim = base.monthly[["mes", "tempmed", "tempmin", "tempmax",
                         "umidmed", "umidmin", "umidmax"]].copy()
    return SourceResult("climate_infodengue", base.url, True, monthly=clim,
                        note="climate bundled in InfoDengue (JP geocode)")


# ---------------------------------------------------------------------------
# INMET (documented attempt — public API returns empty for recent dates)
# ---------------------------------------------------------------------------
def fetch_inmet_attempt() -> SourceResult:
    """Attempt the INMET automatic-station API for João Pessoa (A320).

    Kept as a real, timeboxed attempt so the report documents the URL and error
    rather than claiming the source was never tried. Returns a documented failure
    if the body is empty / endpoint unavailable.
    """
    url = "https://apitempo.inmet.gov.br/estacao/2025-03-01/2025-03-31/A320"
    try:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=25) as r:
            body = r.read()
    except Exception as e:
        return SourceResult("inmet_A320", url, False, error=f"{type(e).__name__}: {e}")
    if not body.strip():
        return SourceResult("inmet_A320", url, False,
                            error="empty body (apitempo serves no data for this "
                                  "station/period without a token/portal export)")
    try:
        data = json.loads(body.decode())
    except Exception as e:
        return SourceResult("inmet_A320", url, False, error=f"parse: {e}")
    return SourceResult("inmet_A320", url, True, monthly=pd.DataFrame(data),
                        note="unexpectedly non-empty; not used (InfoDengue climate preferred)")


# ---------------------------------------------------------------------------
# InfoGripe / SRAG (respiratory) — the clinically most plausible external signal
# for antibiotics. The Fiocruz data center (gitlab.procc.fiocruz.br and
# info.gripe.fiocruz.br) refuses TCP connections from here (http=000), so we use
# the OFFICIAL GitHub mirror (github.com/infogripe/Boletim_InfoGripe), which is
# alive and updated daily. We download the per-state SARI estimates series and
# keep Paraíba (PB), aggregating epi-weeks to calendar months.
# ---------------------------------------------------------------------------
UF_PARAIBA = "PB"
_INFOGRIPE_GITLAB = (  # primary Fiocruz source, documented as down from here
    "https://gitlab.procc.fiocruz.br/mave/repo/-/raw/master/Dados/"
    "InfoGripe/serie_temporal_com_estimativas_recentes.csv.gz"
)
_INFOGRIPE_GITHUB = (  # official GitHub mirror that actually works
    "https://raw.githubusercontent.com/infogripe/Boletim_InfoGripe/main/"
    "Dados/InfoGripe/estados_e_pais_serie_estimativas_tendencia_sem_filtro_febre.csv"
)


def _epiweek_to_month(ano: int, se: int) -> str:
    """Map an ISO-ish epidemiological week to the calendar month of its Wednesday."""
    from datetime import date, timedelta
    jan4 = date(int(ano), 1, 4)
    week1_mon = jan4 - timedelta(days=jan4.weekday())
    d = week1_mon + timedelta(weeks=int(se) - 1, days=3)
    return f"{d.year}-{d.month:02d}"


def fetch_infogripe_srag(uf: str = UF_PARAIBA) -> SourceResult:
    """SRAG (respiratory) monthly series for `uf` from the InfoGripe GitHub mirror.

    The Fiocruz GitLab is unreachable from here (connection refused), so this uses
    the official GitHub mirror. If the mirror also fails, the GitLab attempt is
    recorded as the documented failure. Values use European decimals (comma).
    """
    # 1) document the Fiocruz GitLab attempt (expected to fail from here)
    gitlab_err = ""
    try:
        req = urllib.request.Request(_INFOGRIPE_GITLAB, headers=_UA)
        with urllib.request.urlopen(req, timeout=15) as r:
            r.read(1)
        gitlab_err = "unexpectedly reachable"
    except Exception as e:
        gitlab_err = f"{type(e).__name__}: {e}"

    # 2) real download from the working GitHub mirror
    try:
        req = urllib.request.Request(_INFOGRIPE_GITHUB, headers=_UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
    except Exception as e:
        return SourceResult(
            "infogripe_srag", _INFOGRIPE_GITHUB, False,
            error=f"GitHub mirror failed: {type(e).__name__}: {e}; "
                  f"Fiocruz GitLab also down: {gitlab_err}")

    import io
    df = pd.read_csv(io.BytesIO(raw), sep=";", decimal=",", encoding="utf-8")
    df = df[(df["DS_UF_SIGLA"] == uf) & (df["escala"] == "casos")].copy()
    df["ano"] = pd.to_numeric(df["Ano epidemiológico"], errors="coerce")
    df["se"] = pd.to_numeric(df["Semana epidemiológica"], errors="coerce")
    df["srag"] = pd.to_numeric(
        df["Casos semanais reportados até a última atualização"], errors="coerce")
    df["srag_mm"] = pd.to_numeric(df["média móvel"], errors="coerce")
    df = df.dropna(subset=["ano", "se"])
    df["mes"] = [_epiweek_to_month(a, s) for a, s in zip(df["ano"], df["se"])]

    agg = (df.groupby("mes")
           .agg(srag_casos=("srag", "sum"),
                srag_mm=("srag_mm", "mean"),
                n_semanas_srag=("se", "size"))
           .reset_index())
    return SourceResult(
        "infogripe_srag", _INFOGRIPE_GITHUB, True, monthly=agg,
        note=f"SRAG {uf} from GitHub mirror ({len(df)} weeks -> {len(agg)} months); "
             f"Fiocruz GitLab down: {gitlab_err}")


# ---------------------------------------------------------------------------
# Orchestration: build the monthly external table + cache it
# ---------------------------------------------------------------------------
def build_external_monthly(save: bool = True) -> tuple[pd.DataFrame, list[SourceResult]]:
    """Download everything, merge into ONE monthly table keyed by `mes`.

    Returns (monthly_df, source_results). `monthly_df` has one row per calendar
    month present in any source, with climate + arbovirus columns.
    """
    results: list[SourceResult] = []

    dengue = fetch_infodengue("dengue")
    results.append(dengue)
    chik = fetch_infodengue("chikungunya")
    results.append(chik)
    climate = fetch_climate_from_infodengue(dengue)
    results.append(climate)
    # SRAG respiratory (via GitHub mirror) — real download, clinically the most
    # plausible external driver for antibiotics.
    srag = fetch_infogripe_srag()
    results.append(srag)
    # documented attempt (kept honest, not used downstream unless it succeeds)
    results.append(fetch_inmet_attempt())

    # Merge the successful monthly frames on `mes`.
    frames = []
    if dengue.ok:
        frames.append(dengue.monthly)
    if chik.ok:
        frames.append(chik.monthly[["mes", "casos_chikungunya", "casos_est_chikungunya"]])
    if srag.ok:
        frames.append(srag.monthly)

    if not frames:
        merged = pd.DataFrame(columns=["mes"])
    else:
        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on="mes", how="outer")
        merged = merged.sort_values("mes").reset_index(drop=True)

    if save and not merged.empty:
        config.ensure_dirs()
        out = config.EXTERNAL_DIR / "joaopessoa_clima_epidemia_mensal.csv"
        merged.to_csv(out, index=False)
        meta = {
            "geocode": JOAO_PESSOA_GEOCODE,
            "municipio": "João Pessoa/PB",
            "fontes": [{"name": r.name, "url": r.url, "ok": r.ok,
                        "error": r.error, "note": r.note} for r in results],
            "colunas": list(merged.columns),
            "n_meses": int(len(merged)),
        }
        (config.EXTERNAL_DIR / "FONTES.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return merged, results


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    merged, results = build_external_monthly(save=True)
    for r in results:
        status = "OK " if r.ok else "FAIL"
        print(f"[{status}] {r.name}: {r.note or r.error}")
    print("\nMerged monthly external table:")
    print(merged.to_string(index=False))
