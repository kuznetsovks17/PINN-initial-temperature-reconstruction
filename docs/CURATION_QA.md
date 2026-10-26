# Repository-curation QA (2026-10-26)

**Input selection:** `heat(3).zip` (code/data/results), cross-referenced to English `JAMTP_inverse_reader.tex` in `JAMTP_EN_RU_full_revision(1).zip`.

## Checks completed without training

- 10 selected Jupyter notebooks pass `nbformat.validate`; source code cells remain unchanged and outputs/execution counters were cleared.
- Original Python and FreeFEM files retained byte-for-byte; `docs/SELECTION_MANIFEST.csv` contains source/repository SHA256 hashes.
- 136 entries in the filtered data source manifest verified using `python verify_installation.py` and `scripts/check_repository.py`.
- `verify_installation.py` reports passed core numerical checks and the 1D QR closure diagnostics, in the available CPU Python environment. These are software checks, not a new validation of 2D QR convergence.
- Notebooks **00**, **01**, **02** and **08** successfully executed as nontraining smoke/audit checks from the curated repository root.
- `scripts/paper_results.py` reads and validates saved cohort sizes: 18 smooth 2D, 12 nonlinear 1D, 9 brass, eight nonlinear-2D rows including QR.
- `scripts/legacy_noise_grid_tables.py` reproduced both original sweep tables from saved per-run metrics and QR sweeps and confirmed that the 5%-noise/33x33 observation row refers to the same source run.
- `scripts/reproduce_nonlinear2d_figure.py` successfully regenerated the **clean and 5% noisy** 2D QR-versus-Fourier reconstruction figures from real saved numerical arrays, using the original supplied plotting module without changing its style.
- The separate, previously unpublished forward Fourier-feature experiment, the composite case, 3D scalability files, notebook caches, OS Zone.Identifier sidecars, model checkpoints and automatic batch-start notebook were excluded.

## Boundaries of this QA

- No new neural full-budget training was conducted in this curation step. TensorFlow was **not** installed in the curation environment, and training notebooks 03,04,05,09 were **not** executed through their production branches.
- The archived 2D nonlinear QR results have significant mesh/time-step sensitivity, as documented with the source study. Curation does not repair or validate those scientific claims.
- Historical noise/sensor-grid PINN training code was not available as a standalone reproducer in the supplied `heat(3).zip`. Saved run outputs and the corresponding FreeFEM QR scripts are preserved.
- The saved legacy initial-error metrics differ from three manuscript noise-table values at the fourth decimal after ordinary rounding (1%, 10%, 20%); see `docs/DATA_PROVENANCE.md`.
- Brass measurements were digitized from a published experiment; redistribution rights and calibration metadata need author confirmation.
- The submitted files contained neither a chosen software license nor finalized repository citation metadata. No LICENSE was invented.

## Publication-ready scope

This is an *inspectable code-and-selected-data repository*, not a claim that a fresh clone can reproduce every original manuscript figure without the omitted full checkpoints/older code. Full new experiment reproduction requires the TensorFlow environment and training time documented in the notebooks. The README prominently warns before long execution.
