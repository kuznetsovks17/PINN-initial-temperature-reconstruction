# Initial Temperature Reconstruction Using Physics-Informed Neural Networks: Comparison with Classical Methods

**Reproducibility code and selected research data for the manuscript**  
*Initial Temperature Reconstruction Using Physics-Informed Neural Networks: Comparison with Classical Methods*  
(English/Russian manuscript versions supplied separately by the authors; the manuscript is **not** included in this code repository.)

The repository is a curated selection from the author-supplied `heat(3).zip`, matched to the manuscript in `JAMTP_EN_RU_full_revision(1).zip`. **It concerns the inverse problem only**. The separate Fourier-feature/forward-PINN article, composite-material case and removed 3D section are excluded.

## Contents and paper correspondence

| Manuscript experiment | Executable notebook(s) | Implementation / input data |
|---|---|---|
| Observation protocol, convergence and modal recoverability | `00_...`, `01_...` | `inverse_study/common.py`, `numerics.py`; stored 1D reference |
| Smooth linear 2D: three PINNs versus spectral Tikhonov/TSVD (three repetitions) | `02_...`, `03_...`, `07_...` | `pinn.py`, `completion.py`, `workflows.py`; shared interior observations |
| Smooth 2D: **historical** noise 0–20% and tensor grids 5²–33² | `scripts/legacy_noise_grid_tables.py` | `data/legacy_runs/`, `data/legacy_qr/`, `legacy_sources/freefem/01_*.edp` and `02_*.edp` |
| Strongly nonlinear 2D (six PINNs and nonlinear mixed QR) | `08_...`, `09_...`, `10_...` | `nonlinear2d.py`, `pinn_nonlinear2d.py`, `qr_nonlinear2d.py`, `nonlinear2d_figures.py` |
| Nonlinear 1D discontinuous profile and classical fit | `01_...`, `04_...`, `07_...` | `numerics.py`, `completion_checks.py`; original 1D FreeFEM script |
| Digitized brass temperature profiles | `05_...`, `07_...` | `brass.py`, `brass_completion.py`, `data/brass/`; historical Python scripts |
| Paper numerical result tables without training | `scripts/paper_results.py` | `published_results/` snapshots |

**Important:** the linear spectral TSVD/Tikhonov comparator is applicable directly to the **linear 2D** problem. The nonlinear 1D Tikhonov comparator uses a regularized initial-profile fit. The **nonlinear 2D** classical method is the mixed finite-element quasi-reversibility (QR) solver; nonlinear TSVD/Tikhonov are **not** represented as calculated results.

## Quick start

From a terminal in the extracted **repository root**:

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python verify_installation.py
python scripts/paper_results.py
python scripts/legacy_noise_grid_tables.py
# Optional: regenerate nonlinear-2D QR/PINN comparison figures in generated_figures/
python scripts/reproduce_nonlinear2d_figure.py
```

To run neural training, additionally install `requirements-training.txt` in a supported TensorFlow environment, then launch `jupyter lab` from the repository root. Use the numbered notebooks in the order **00, 01, 02, 03, 04, 05, 07, 08, 09, 10** as relevant. The 2D nonlinear QR notebook 10 requires the reference prepared by 08 and, for combined plots, trained outputs from 09.

> **WARNING — expensive execution:** In the supplied original executable notebooks, `RUN_PRODUCTION=True` and analogous production flags are enabled. **Running 03, 04, 05, or 09 with “Run All” starts substantial neural training** if checkpoints are absent. Notebook 10 can execute a QR gamma sweep. The *static* `scripts/paper_results.py` and `scripts/legacy_noise_grid_tables.py` report saved results **without retraining**. Inspect a notebook's controls before running it. The completed full training checkpoints are intentionally not redistributed here.

## Data and status

- `data/brass/`: digitized measurements and coefficient law, from the published Zhang and co-workers case as transcribed in the author-supplied archive. Confirm redistribution rights and the dataset citation before public release.
- `data/legacy_runs/`, `data/legacy_qr/`: retained historical results for 1D nonlinear and 2D noise/grid studies. The original PINN training workflows for these *boundary-inclusive* legacy sweeps were **not identified as executable standalone programs in the supplied heat archive**. Legacy FreeFEM QR scripts and saved run arrays are included. Do **not** present them as identical to the new interior-only study.
- `results/verification`, `results/observations`, `results/nonlinear2d/reference`, `results/nonlinear2d/observations`: bundled reusable verification data and frozen inputs. Additional classical QR/Tikhonov saved outputs are available under `results/`.
- `published_results/`: author-supplied computed tables, figures and selected per-run metrics/predictions; they do not populate the normal training caches. Do not treat these as full trainable/restorable checkpoints.
- `legacy_sources/`: author-supplied QR scripts and historical brass scripts, kept separate because their settings/data handling differ from the matched-comparison notebooks.

The nonlinear 2D QR results are **preliminary with respect to mesh/time-step convergence**: the saved check indicates approximately 11–12% changes. No fresh computation was used to change their values when selecting the repository contents.

## Reproducibility and provenance

See [EXPERIMENT_MAP](docs/EXPERIMENT_MAP.md), [DATA_PROVENANCE](docs/DATA_PROVENANCE.md), and [SELECTION_MANIFEST.csv](docs/SELECTION_MANIFEST.csv) and [CURATION_QA.md](docs/CURATION_QA.md). All `.py`/`.edp` source files are copied byte-for-byte from the supplied `heat(3).zip`; notebooks have only their outputs/execution metadata removed. `scripts/` and documentation are new packaging helpers, not the numerical methods used for the reported runs. Notebook source cells remain unchanged.

## Before making the repository public

1. Choose and add a code **LICENSE** (no rights are granted by this repository template).
2. Confirm author names/affiliations/citation metadata and experiment-data redistribution permission.
3. Archive full training checkpoints separately (e.g. an institutional or DOI-backed data repository), if you intend to provide exact restart/retraining state.
4. Confirm that publishing the repository before journal acceptance is compatible with the journal's policies.

Repository maintained by [kuznetsovks17](https://github.com/kuznetsovks17).
