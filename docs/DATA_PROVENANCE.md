# Data and provenance notes

- Only the specified two archives were used for file selection: `heat(3).zip` for executable programs and numerical outputs; `JAMTP_EN_RU_full_revision(1).zip` for manuscript correspondence. The manuscript itself was not redistributed.
- `data/SOURCE_MANIFEST.json` is a **filtered subset** of the upstream manifest containing only files actually in this repository. It is not the original 176-entry manifest.
- File hashes and original paths appear in `docs/SELECTION_MANIFEST.csv`. Python/FreeFEM programs were copied without changing their bytes. Executable notebook cells were preserved; displayed outputs, execution counts and non-essential notebook metadata were removed to keep GitHub history clean.
- Historical 2D QR and PINN noise/grid sweeps have distinct boundary preprocessing compared with the modern interior-only observation contract. Their saved results may be analyzed, but not combined as statistically identical tests.
- Precomputed numerical results are snapshots copied from the author's submitted archive. The upload does not contain verifiable timestamps or signed run attestations. Re-running the full study requires the specified compute environment and budgets; no such production re-training was carried out during repository curation.
- Brass experimental measurements were digitized from a published study. Confirm dataset reuse/redistribution rights and source attribution. The provenance records do not establish independent calibration data.
- The QR strong nonlinear 2D solver evaluates diffusivity at the previous reversed-time step, with a mixed P1 FEM discretization. The available refinement changes its recovered initial field by about 11–12%; do not present this as a converged benchmark without further testing.
- The excluded 3D/scalability, forward Fourier ablation, composite material and optional broad regime-grid code are not needed for the present inverse paper. They remain in the user's original uploaded archive.

## Source-to-manuscript rounding notice

Comparing the English manuscript noise table with `data/legacy_runs/*/metrics.json` revealed minor differences in the **fourth displayed decimal** for the 1%, 10%, and 20% PINN entries: saved metrics round to `0.1351`, `0.2085`, `0.2210`, whereas the manuscript prints `0.1350`, `0.2086`, `0.2211`. The repository retains the actual archived metrics. **No manuscript results were silently edited** during repository curation. The corresponding QR values and grid-density entries agree at four decimal places.
