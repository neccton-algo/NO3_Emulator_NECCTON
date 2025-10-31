# ML-derived Nitrate for bias correction
This repository contains the workflow and codes for the NECCTON nitrate emulator, which reconstructs daily, 7 km resolution surface nitrate fields (1993–2020) over the North-West European Shelf using machine learning.

model/auto.py – trains an AutoKeras neural-network regressor using ICES nitrate observations combined with Copernicus reanalysis, ERA5 forcing, and riverine inputs.

expdir/files_tobe/pred_NO3_optm_1core.py – runs the trained model on HPC to generate gap-free nitrate predictions saved as NetCDF files (year-wise).

copy_run.sh – quick setup script to copy everything and auto-create yearly run directories with SLURM job files.
