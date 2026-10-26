# README figure sources

These figures were supplied by the manuscript authors in the `ENGLISH` package for *Initial Temperature Reconstruction Using Physics-Informed Neural Networks: Comparison with Classical Methods*. They are used as research illustrations in the repository README.

| Repository asset | Supplied source | Context |
|:--|:--|:--|
| `fourier-architecture.png` | `figs/affe.png` | Main article, Fig. 1: Fourier encoding and temperature network |
| `linear2d-noise-reconstructions.png` | `figs/exp2.png` | Main article, Fig. 2: historical boundary-inclusive noise study |
| `nonlinear2d-qr-fourier.png` | `figs/nonlinear2d_qr_vs_fourier_noisy5_original_style.pdf` | Main article, Fig. 3: nonlinear 2D, nominal 5% noise |
| `nonlinear1d-reconstruction.png` | `figs/exp3.png` | Main article, Fig. 4: representative nonlinear 1D APINN/QR profiles |
| `brass-experiment.png` | `figs/phys.png` | Main article, Fig. 5: brass profiles and withheld-time checks |
| `noise-and-grid-sensitivity.png` | `ESM_1_assets/ablation_analysis.png` | Supplement, Fig. S1: noise, modes and observation density |
| `measurement-grid-reconstructions.png` | `ESM_1_assets/exp2_.png` | Supplement, Fig. S2: sensor layouts and reconstructed fields |

Six PNG figures are copied byte-for-byte, with descriptive filenames. The nonlinear 2D PDF figure is rendered to PNG at a maximum dimension of 1700 pixels for inline GitHub display; its plotted data, labels and panel layout are unchanged. No graph was redrawn and no new numerical result was generated for README styling.

The historical noise/grid studies use complete boundary-inclusive observation grids. Their single-run results are distinct from the repeated interior-only comparisons. Representative APINN/QR profile figures do not depict every formulation included in the quantitative tables.

The original source/data hashes in `docs/SELECTION_MANIFEST.csv` and `data/SOURCE_MANIFEST.json` continue to refer to the curated numerical package. These documentation illustrations are separate additions.
