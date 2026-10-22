# Raw data — how to obtain

The original workbooks are **not versioned** because they contain operational data
from HU-UFPB. To reproduce the pipeline, place the files received from the hospital
pharmacy in this folder:

- `Planilha CCIH 2026 Atualizado.xlsx` — antimicrobial treatment sheet (one row per
  treatment: sector, status, days, posology, quantities, start/end dates). The patient
  name already comes anonymized (empty) in the file we received.
- `02 - Planilha de abastecimento do estoque geral.xlsx` — stock position (AGHU export).

Source: HU-UFPB pharmacy, shared for academic use during the stakeholder interview
(Deliverable 3). Do not redistribute.

Once the files are here, run:

```bash
make notebook      # or: uv run jupyter nbconvert --execute ...
```
