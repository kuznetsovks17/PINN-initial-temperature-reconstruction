"""Cached classical fits and independent replay for the reduced inverse study.

These additions never rewrite a saved neural checkpoint, observation vector,
reference solution, or imported evaluation.json. No training occurs here.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .common import (ROOT, RESULTS, array_hash, freeze_observations,
                     initial_jump, jump_metrics, load_observations,
                     metadata_hash, problem_spec, read_json, sha256_file,
                     tensor_grid, write_json)
from . import numerics as num
from . import workflows as flow


def _limit(value):
    if value is not None and (isinstance(value, bool) or int(value) != value or value < 0):
        raise ValueError("A job limit must be None or a nonnegative integer")
    return None if value is None else int(value)


def _source_identity():
    return {"numerics_sha256": sha256_file(ROOT / "inverse_study/numerics.py"),
            "checks_sha256": sha256_file(Path(__file__))}


def _save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _read_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def _cached_record(folder, name, signature, array_name):
    path, arrays = folder / name, folder / array_name
    if not path.exists():
        return None
    record = read_json(path)
    if record.get("signature") != signature:
        raise RuntimeError(f"Cached configuration differs: {path}")
    if not arrays.is_file() or sha256_file(arrays) != record.get("arrays_sha256"):
        raise RuntimeError(f"Cached arrays are missing or changed: {arrays}")
    return record


def nonlinear_baseline_cases(plan=None):
    """One case per observed vector: clean plus three independent noise draws."""
    from .completion import compact_nonlinear_plan, verify_protocol_lock
    reference, gate = verify_protocol_lock(require_reference=True)
    plan = compact_nonlinear_plan() if plan is None else plan
    if len(plan) and not plan["case"].eq("nonlinear1d").all():
        raise ValueError("The nonlinear comparator accepts nonlinear1d cases only")
    rows = []
    keys = ["case", "exposure", "obs_per_dim", "noise_level", "noise_seed"]
    for _, row in plan.drop_duplicates(keys).iterrows():
        p = problem_spec("nonlinear1d")
        folder = freeze_observations(p, int(row.obs_per_dim), float(row.noise_level),
                                     int(row.noise_seed), reference)
        _, obs, meta = load_observations(folder)
        rows.append(dict(problem=p, observations=obs, metadata=meta,
                         dataset_folder=str(folder.relative_to(ROOT)),
                         reference=reference, reference_gate=gate))
    return rows


def _baseline_config(case, nx, n_knots, alphas, max_nfev):
    alphas = np.asarray(alphas, dtype=float)
    if alphas.ndim != 1 or not len(alphas) or not np.isfinite(alphas).all() or np.any(alphas < 0):
        raise ValueError("alphas must be a nonempty one-dimensional finite nonnegative grid")
    alphas = np.unique(alphas)
    for name, value in (("nx", nx), ("n_knots", n_knots), ("max_nfev", max_nfev)):
        if isinstance(value, bool) or not np.isfinite(value) or int(value) != value:
            raise ValueError(f"{name} must be an integer")
    if int(n_knots) < 4 or int(nx) < int(n_knots) or int(max_nfev) < 1:
        raise ValueError("Require nx >= n_knots >= 4 and max_nfev >= 1")
    return dict(protocol="reduced-nonlinear-tikhonov-v1", **_source_identity(),
                dataset_id=case["metadata"]["dataset_id"],
                observation_sha256=case["metadata"]["observation_sha256"],
                reference_sha256=case["reference_gate"]["reference_sha256"],
                problem=case["problem"], nx=int(nx), n_knots=int(n_knots),
                alphas=alphas.tolist(),
                max_nfev=int(max_nfev), rtol=1e-8, atol=1e-10,
                max_step=.005, tau=1.1, value_bounds=[0., 2.],
                floor_rule="max(reference gate floor, twice coarse-to-reference sensor difference, 1e-10)",
                initialization="same data-derived mean for every alpha; no initial truth")


def nonlinear_baseline_inventory(plan=None, nx=201, n_knots=17,
                                 alphas=None, max_nfev=200):
    alphas = np.logspace(-9, -1, 13) if alphas is None else alphas
    rows = []
    for case in nonlinear_baseline_cases(plan):
        cfg = _baseline_config(case, nx, n_knots, alphas, max_nfev)
        signature = metadata_hash(cfg)
        folder = RESULTS / "nonlinear/reduced" / cfg["dataset_id"] / signature[:12]
        saved = _cached_record(folder, "candidate.json", signature, "candidate.npz")
        finished = 0
        for alpha in cfg["alphas"]:
            a_dir = folder / "alphas" / f"{alpha:.12e}"
            a_sig = metadata_hash({"parent_signature": signature, "alpha": alpha})
            if _cached_record(a_dir, "fit.json", a_sig, "fit.npz") is not None:
                finished += 1
        rows.append(dict(case="nonlinear1d", dataset_id=cfg["dataset_id"],
                         noise_level=case["metadata"]["noise_level"],
                         noise_seed=case["metadata"]["noise_seed"],
                         status="completed" if saved else "missing",
                         completed_alphas=finished, planned_alphas=len(cfg["alphas"]),
                         run_folder=str(folder.relative_to(ROOT)), signature=signature))
    return pd.DataFrame(rows)


def run_nonlinear_baselines(plan=None, max_new_cases=None, nx=201, n_knots=17,
                            alphas=None, max_nfev=200):
    """Fit each retained dataset once; each completed alpha also survives a restart.

    The alpha fits are independent in numerics.nonlinear_tikhonov. Calling it
    one alpha at a time preserves its initialization and selection semantics,
    while making interruption recovery possible without repeating solved fits.
    Optimizer nonconvergence is retained as a computed result, never discarded.
    """
    limit = _limit(max_new_cases)
    alphas = np.logspace(-9, -1, 13) if alphas is None else alphas
    records, launched = [], 0
    for case in nonlinear_baseline_cases(plan):
        p, obs, meta = case["problem"], case["observations"], case["metadata"]
        cfg = _baseline_config(case, nx, n_knots, alphas, max_nfev)
        signature = metadata_hash(cfg)
        folder = RESULTS / "nonlinear/reduced" / meta["dataset_id"] / signature[:12]
        saved = _cached_record(folder, "candidate.json", signature, "candidate.npz")
        if saved is not None:
            records.append(saved | {"execution": "cached"})
            print("Reused nonlinear baseline:", meta["dataset_id"], flush=True)
            continue
        if limit is not None and launched >= limit:
            continue
        launched += 1
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "config.json", cfg | {"signature": signature})
        calibration_file = folder / "numerical_floor.json"
        if calibration_file.exists():
            calibration = read_json(calibration_file)
            if calibration.get("signature") != signature:
                raise RuntimeError("Nonlinear numerical-floor configuration changed")
            numerical_floor = calibration["numerical_floor"]
        else:
            x = np.linspace(0, 1, int(nx))
            forward = num.forward_1d(p, x, initial_jump(x), [p["T"]],
                                     rtol=cfg["rtol"], atol=cfg["atol"], max_step=cfg["max_step"])
            coarse = np.interp(obs["X_obs"][:, 0], x, forward["terminal"])
            fine = np.interp(obs["X_obs"][:, 0], case["reference"]["x"], case["reference"]["terminal"])
            numerical_floor = max(case["reference_gate"]["suggested_numerical_floor"],
                                  2 * float(np.linalg.norm(coarse - fine)), 1e-10)
            write_json(calibration_file, dict(signature=signature, numerical_floor=numerical_floor,
                       coarse_reference_sensor_difference=float(np.linalg.norm(coarse - fine)),
                       scope="synthetic forward discretization calibration before inverse fitting"))
        paths, candidates, target = [], [], None
        started = perf_counter()
        for alpha in cfg["alphas"]:
            a_dir = folder / "alphas" / f"{alpha:.12e}"
            a_sig = metadata_hash({"parent_signature": signature, "alpha": alpha})
            cached = _cached_record(a_dir, "fit.json", a_sig, "fit.npz")
            if cached is None:
                print(f"Nonlinear baseline {meta['dataset_id']}: alpha={alpha:g}", flush=True)
                fit = num.nonlinear_tikhonov(p, obs, n_knots=n_knots, nx=nx,
                        alphas=[alpha], numerical_noise_floor=numerical_floor,
                        rtol=cfg["rtol"], atol=cfg["atol"], max_step=cfg["max_step"],
                        max_nfev=max_nfev, value_bounds=tuple(cfg["value_bounds"]), tau=cfg["tau"])
                arrays = {k: fit[k] for k in ["x", "knots", "coefficients", "initial", "terminal", "y_fit"]}
                if not all(np.isfinite(v).all() for v in arrays.values()):
                    raise FloatingPointError("Nonfinite classical fit; no completed cache was written")
                _save_arrays(a_dir / "fit.npz", **arrays)
                cached = dict(signature=a_sig, alpha=alpha, path=fit["path"][0],
                              target_residual=fit["target_residual"],
                              arrays_sha256=sha256_file(a_dir / "fit.npz"))
                write_json(a_dir / "fit.json", cached)
            else:
                arrays = _read_arrays(a_dir / "fit.npz")
            if target is not None and not np.isclose(target, cached["target_residual"], rtol=1e-12):
                raise RuntimeError("Inconsistent discrepancy targets between cached alpha fits")
            target = cached["target_residual"]
            paths.append(cached["path"])
            candidates.append(arrays)
        eligible = [i for i, r in enumerate(paths) if r["optimizer_success"] and r["residual_norm"] <= target]
        converged = [i for i, r in enumerate(paths) if r["optimizer_success"]]
        if eligible:
            chosen = eligible[-1]
            status = ("discrepancy_attained" if chosen < len(paths) - 1 else
                      "discrepancy_attained_at_strongest_grid_edge_expand_alpha")
        elif converged:
            chosen = min(converged, key=lambda i: paths[i]["residual_norm"])
            status = "discrepancy_unattainable_among_converged_candidates"
        else:
            chosen = int(np.argmin([r["residual_norm"] for r in paths]))
            status = "no_optimizer_candidate_converged"
        best = candidates[chosen]
        _save_arrays(folder / "candidate.npz", **best)
        pd.DataFrame(paths).to_csv(folder / "parameter_path.csv", index=False)
        scores = jump_metrics(best["x"], best["initial"])
        record = dict(case="nonlinear1d", method="nonlinear_piecewise_linear_tikhonov",
                      profile="candidate", dataset_id=meta["dataset_id"],
                      observation_sha256=meta["observation_sha256"],
                      noise_level=meta["noise_level"], noise_seed=meta["noise_seed"],
                      run_folder=str(folder.relative_to(ROOT)), signature=signature,
                      **scores, selection_status=status, selected_parameter=paths[chosen]["parameter"],
                      independent_observation_residual=paths[chosen]["residual_norm"],
                      target_residual=target, numerical_floor=numerical_floor,
                      n_knots=n_knots, nx=nx, optimizer_success=paths[chosen]["optimizer_success"],
                      full_parameter_fit_wall_seconds=sum(r["wall_seconds"] for r in paths),
                      current_call_wall_seconds=perf_counter() - started,
                      arrays_sha256=sha256_file(folder / "candidate.npz"),
                      publication_eligible=False,
                      remaining_checks="knot/mesh sensitivity and parameter-grid adequacy; empirical reference floor")
        write_json(folder / "candidate.json", record)
        records.append(record | {"execution": "computed"})
    return pd.DataFrame(records)


def refine_neural_replays(plan, max_new_runs=None):
    """Refine independent forward solves for every completed selected checkpoint.

    This reuses final neural weights. Each replay grid is cached separately.
    A passed empirical mesh check does not validate initial-state uniqueness,
    physical model adequacy, or a large initial boundary mismatch.
    """
    from .completion import plan_inventory
    from .pinn import predict_saved
    limit = _limit(max_new_runs)
    inventory = plan_inventory(plan)
    if inventory.status.eq("blocked").any():
        raise RuntimeError("Resolve blocked training identities before independent replay")
    records, launched = [], 0
    for _, row in inventory[inventory.status.eq("completed")].iterrows():
        run = ROOT / row.run_folder
        study = read_json(run / "study_record.json")
        p, obs, meta = load_observations(ROOT / study["dataset_folder"])
        if int(p["dim"]) == 3:
            grids = [17, 25, 33]
        elif p["law"] == "constant" and int(p["dim"]) == 2:
            grids = [33, 65, 129]
        else:
            grids = [401, 801, 1601]
        cfg = dict(protocol="reduced-independent-replay-v1", **_source_identity(),
                   run_folder=row.run_folder, training_signature=row.signature,
                   weights_sha256=sha256_file(run / "weights_final.npz"),
                   observation_sha256=meta["observation_sha256"], grid_nodes=grids,
                   rtol=1e-8, atol=1e-10, max_step=.005,
                   empirical_tolerance_rule="max(1e-4*norm(y),0.05*known synthetic noise norm,1e-12)")
        signature = metadata_hash(cfg)
        folder = RESULTS / "reduced_validation/neural_replay" / run.name / signature[:12]
        cached = _cached_record(folder, "summary.json", signature, "sensor_comparison.npz")
        if cached is not None:
            records.append(cached | {"execution": "cached"})
            continue
        if limit is not None and launched >= limit:
            continue
        launched += 1
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "config.json", cfg | {"signature": signature})
        refinements, sensor_arrays = [], []
        for nx in grids:
            mesh_folder = folder / f"grid_{nx}"
            mesh_signature = metadata_hash({"parent": signature, "nx": nx})
            mesh_saved = _cached_record(mesh_folder, "metrics.json", mesh_signature, "solution.npz")
            if mesh_saved is None:
                print(f"Independent replay {run.name}: grid {nx}", flush=True)
                axes, X = tensor_grid(p["dim"], nx, bounds=p["bounds"])
                u0 = predict_saved(run, X, times=0.0).ravel()
                if p["law"] == "constant":
                    fw = num.forward_linear_grid(u0.reshape(tuple(len(a) for a in axes)),
                                                 axes, p["Fo"] * float(p.get("diffusivity_multiplier", 1.0)), p["T"])
                    predicted = RegularGridInterpolator(axes, fw["terminal"])(obs["X_obs"])
                    boundary_linf = fw["initial_boundary_linf"]
                else:
                    fw = num.forward_1d(p, axes[0], u0, [p["T"]],
                                         rtol=cfg["rtol"], atol=cfg["atol"], max_step=cfg["max_step"])
                    predicted = np.interp(obs["X_obs"][:, 0], axes[0], fw["terminal"])
                    boundary_linf = float(np.max(np.abs(fw["initial_boundary_correction"])))
                if not np.isfinite(predicted).all():
                    raise FloatingPointError("Nonfinite independent replay")
                _save_arrays(mesh_folder / "solution.npz", sensor_terminal=predicted,
                             X_eval=X, initial=u0, terminal=np.asarray(fw["terminal"]))
                mesh_saved = dict(signature=mesh_signature, nx=nx,
                                 independent_observation_residual=float(np.linalg.norm(predicted - obs["y_obs"])),
                                 initial_boundary_linf=boundary_linf,
                                 forward_wall_seconds=fw["wall_seconds"],
                                 arrays_sha256=sha256_file(mesh_folder / "solution.npz"))
                write_json(mesh_folder / "metrics.json", mesh_saved)
            arrays = _read_arrays(mesh_folder / "solution.npz")
            sensor_arrays.append(arrays["sensor_terminal"])
            refinements.append(mesh_saved)
        fine = sensor_arrays[-1]
        for mesh, values in zip(refinements, sensor_arrays):
            mesh["sensor_difference_to_finest_l2"] = float(np.linalg.norm(values - fine))
        difference = float(np.linalg.norm(sensor_arrays[-2] - fine))
        tolerance = max(1e-4 * float(np.linalg.norm(obs["y_obs"])),
                        .05 * float(obs["noise_norm"]), 1e-12)
        _save_arrays(folder / "sensor_comparison.npz", grid_nodes=np.asarray(grids),
                     sensor_terminal=np.asarray(sensor_arrays), y_observed=obs["y_obs"])
        pd.DataFrame(refinements).to_csv(folder / "refinement.csv", index=False)
        summary = dict(run_folder=row.run_folder, case=study["case"], method=study["method"],
                       noise_level=meta["noise_level"], noise_seed=meta["noise_seed"],
                       training_seed=study["training_seed"], signature=signature,
                       training_signature=row.signature, dataset_id=meta["dataset_id"],
                       grid_nodes=grids, sensor_difference_l2=difference,
                       fine_observation_residual=float(np.linalg.norm(fine - obs["y_obs"])),
                       empirical_tolerance=tolerance, empirical_gate_passed=bool(difference <= tolerance),
                       boundary_linf=refinements[-1]["initial_boundary_linf"],
                       scope="fixed saved checkpoint; empirical discretization check",
                       publication_eligible=False,
                       arrays_sha256=sha256_file(folder / "sensor_comparison.npz"))
        write_json(folder / "summary.json", summary)
        records.append(summary | {"execution": "computed"})
    return pd.DataFrame(records)
