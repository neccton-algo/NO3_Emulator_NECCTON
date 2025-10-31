# NECCTON Nitrate Emulator
**ML-derived Surface Nitrate Bias Correction for the Northwest European Shelf (1993–2020)**

Contents:
- `model/auto.py` — AutoKeras training
- `expdir/files_tobe/pred_NO3_optm_1core.py` — year-wise prediction (NetCDF)
- `expdir/copy_run.sh` — SLURM staging
- `notebooks/` — baseline + demo
- `docs/` — data sources & metrics
- `environment.yml`, `LICENSE`, `CODEOWNERS`, `.github/workflows/ci.yml`

See `docs/` for data & metrics. Create env with:
```bash
conda env create -f environment.yml && conda activate neccton_no3
