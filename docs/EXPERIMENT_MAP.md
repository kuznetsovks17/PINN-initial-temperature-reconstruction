# Correspondence between the manuscript and the code

Source: English LaTeX `JAMTP_inverse_reader.tex` in the user-supplied bilingual manuscript archive.

| Paper section / figure | Code / notebooks | Saved evidence | Comment |
|---|---|---|---|
| Problem formulation, PINN and loss | `inverse_study/pinn.py`, `common.py`, `nonlinear2d.py`, `pinn_nonlinear2d.py` | per-run config values under `published_results/neural/`, `published_results/nonlinear2d/neural/` | Same computational modules as author archive |
| Classical QR/Tikhonov/TSVD | `inverse_study/qr_audit.py`, `qr_nonlinear2d.py`, `numerics.py`, `workflows.py`; `02_...` | `results/classical/candidate`, `results/nonlinear/reduced`, `results/nonlinear2d/qr_mixed` | Nonlinear 2D uses QR, not spectral TSVD |
| Linear 2D and modal amplitudes; main error and mode tables | `01_...`, `02_...`, `03_...`, `07_...` | `published_results/reduced_report/synthetic_descriptive_summary.csv`, `classical_descriptive_summary.csv`, related individual records | Three neural repetitions; classical fits per observation vector |
| Noise-level study (0%, 1%, 5%, 10%, 15%, 20%); main table and Supplement S1 | FreeFEM `01_noise_sweep_2d.edp`, historical runs; `scripts/legacy_noise_grid_tables.py` | `data/legacy_runs/noise_sweep*`, `data/legacy_qr/fields/noise*` and shared case | Boundary-inclusive legacy observations: different protocol |
| Tensor-grid study (5x5 to 33x33); main table and Supplement S2 | FreeFEM `02_observation_count_2d.edp`, historical runs; `scripts/legacy_noise_grid_tables.py` | `data/legacy_runs/observation_sweep*`, `data/legacy_qr/fields/observation*` and shared case | 5% / 33x33 sample reused, not independent |
| Strongly nonlinear 2D QR versus PINN; reconstruction comparison | `08_...`, `09_...`, `10_...`; `inverse_study/nonlinear2d*.py`, `pinn_nonlinear2d.py`, `qr_nonlinear2d.py` | `published_results/nonlinear2d/report/nonlinear2d_qr_pinn_comparison.csv`; QR caches and observation files | Clean and 5% noise, single neural seed; QR discretization sensitivity remains |
| Nonlinear 1D discontinuity | `01_...`, `04_...`, `07_...`; historical FreeFEM `03_discontinuous_nonlinear_1d(2).edp` | `published_results/reduced_report/nonlinear_baseline_candidates.csv`, `synthetic_descriptive_summary.csv` | Nonlinear regularized 1D fit uses 17 knots |
| Brass experiment and physical-unit errors | `05_...`, `07_...`; `brass.py`, `brass_completion.py`; historical brass Python scripts | `data/brass/`, `published_results/reduced_report/brass_initial_descriptive_summary.csv` | One experiment, three neural initializations; calibration provenance caveat |

Original article graphics used in the README are included in `docs/readme_assets/`, with their manuscript and supplementary source mappings in `docs/readme_assets/README.md`. The generated nonlinear-2D graphics also remain under `published_results/nonlinear2d/figures`. This repository does not claim to re-create every hand-composed legacy figure from code contained in `heat(3).zip`.
