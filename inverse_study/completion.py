"""A small, paired completion plan that reuses the received production runs.

This module changes scheduling only.  The numerical and training engines remain
byte-identical to the uploaded ``heat.zip``; their signatures and checkpoints
therefore remain valid.  Three repetitions are descriptive evidence, not a
well-powered estimate of failure probabilities or universal method superiority.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import pinn, workflows as flow
from .common import (ROOT, RESULTS, freeze_observations, load_observations,
                     problem_spec, read_json, sha256_file, write_json)

PLAN_ID = "compact_paired_v1"
TRAINING_SEEDS = (2026, 2027, 2028)
NOISY_PAIRS = ((3100, 2026), (3101, 2027), (3102, 2028))
PLAN_KEYS = ["case", "exposure", "obs_per_dim", "noise_level", "noise_seed",
             "training_seed", "method"]
_TRUTH_KEYS = {"modes", "amplitudes", "initial_left", "initial_right", "jump_x"}
_FROZEN_REFERENCE_SHA256 = "94042a59b17d24db9c16c4c38f04a0022cad77613ca42a7b9b955c182859a20a"
_ENGINE_HASHES = {
    "pinn.py": "fc2c4d7f6619eeb56f253cb09a81c98a792f43b3b9d5bf7fe83f8f4d64bf91b0",
    "common.py": "24c2fe8ac5ae17899d1a976b60d94eae6b2e3a2160785ec821b8a9ea0afee2e9",
    "numerics.py": "638ec221aea83d40090d0a3d4bbbd6a70fa0d7eb0cd06af5f4ed6fb63f43047e",
    "workflows.py": "18519ce3861a5c086de7d597bcf7f33bd279bc761a1654ab6b625f3dcc6efb3e",
    "brass.py": "713ab07063f6284aeb454e17356c498cb248f2ef60cd909cc9852af3c17560cc",
    "qr_audit.py": "25b87f8199a966495a319fbbb66539dbbf958e5d3b80c615b18958e992de8d67",
}


def verify_protocol_lock(require_reference=False):
    """Fail before queueing if the frozen scientific protocol has changed."""
    for name, expected in _ENGINE_HASHES.items():
        if sha256_file(ROOT / "inverse_study" / name) != expected:
            raise RuntimeError(f"Frozen engine changed: {name}. Keep the supplied engine "
                               "for this cohort; declare protocol changes as a separate study.")
    if require_reference:
        reference, gate = flow.load_reference(require_pass=True)
        if gate["reference_sha256"] != _FROZEN_REFERENCE_SHA256:
            raise RuntimeError("The nonlinear reference differs from the one used by the "
                               "received runs. Restore the supplied reference for this cohort.")
        return reference, gate
    return None


def _compact_plan(case, methods, exposure, obs_per_dim):
    rows = []
    for rho in (0.0, 0.05):
        pairs = [(0, seed) for seed in TRAINING_SEEDS] if rho == 0 else NOISY_PAIRS
        for rep, (noise_seed, training_seed) in enumerate(pairs, 1):
            for method in methods:
                rows.append(dict(stage=PLAN_ID, case=case, exposure=exposure,
                                 obs_per_dim=obs_per_dim, noise_level=rho,
                                 noise_seed=noise_seed, training_seed=training_seed,
                                 method=method, repetition=rep))
    return pd.DataFrame(rows)


def compact_linear_plan():
    """18 jobs: three methods, clean/noisy, three paired repetitions."""
    return _compact_plan("smooth2d", ("plain", "fourier_fixed", "fourier_adaptive"), .02, 33)


def compact_nonlinear_plan():
    """12 jobs: two methods, clean/noisy, three paired repetitions."""
    return _compact_plan("nonlinear1d", ("plain", "fourier_adaptive"), .1, 257)


def compact_core_plan():
    """30 synthetic jobs; the nine nominal brass jobs are a separate block."""
    return pd.concat([compact_linear_plan(), compact_nonlinear_plan()], ignore_index=True)


def _relative(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _identity(problem, observations, config, output_dir):
    """Mirror the unchanged engine's pre-TensorFlow identity calculation."""
    p, cfg, Xobs, yobs = pinn._prepare(problem, observations, config)
    Xeval = np.asarray(cfg.pop("X_eval"), dtype=float) if "X_eval" in cfg else None
    dim, T = int(p["dim"]), float(p["T"])
    bounds = np.asarray(p["bounds"], dtype=float)
    if Xeval is None:
        n = cfg["eval_per_dim"] or {1: 1001, 2: 101, 3: 33}[dim]
        axes = [np.linspace(a, b, int(n)) for a, b in bounds]
        Xeval = np.column_stack([v.ravel() for v in np.meshgrid(*axes, indexing="ij")])
    if (Xeval.ndim != 2 or Xeval.shape[1] != dim or not len(Xeval)
            or not np.isfinite(Xeval).all() or np.any(Xeval < bounds[:, 0])
            or np.any(Xeval > bounds[:, 1])):
        raise ValueError("X_eval must be a finite (n,dim) array inside the physical box")
    eval_times = np.asarray(cfg.get("eval_times", [0.0, T]), dtype=float)
    if (eval_times.ndim != 1 or not len(eval_times) or not np.isfinite(eval_times).all()
            or np.any(eval_times < 0) or np.any(eval_times > T)):
        raise ValueError("eval_times must lie in [0,T]")
    identity_cfg = {k: v for k, v in cfg.items() if k not in {"resume", "print_every", "run_id"}}
    identity = dict(engine=pinn._ENGINE_VERSION,
                    source_sha256=hashlib.sha256(Path(pinn.__file__).read_bytes()).hexdigest(),
                    problem=p, config=identity_cfg, X_obs_sha256=pinn._hash_array(Xobs),
                    y_obs_sha256=pinn._hash_array(yobs), X_eval_sha256=pinn._hash_array(Xeval))
    signature = pinn._metadata_hash(identity)
    name = str(cfg.get("run_id", f"{p.get('name', 'inverse')}__{cfg['method']}__seed{cfg['training_seed']}"))
    name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    run = Path(output_dir).expanduser().resolve() / f"{name}__{signature[:12]}"
    return p, cfg, identity, signature, run


def _same_unit(saved, run, p, cfg, identity, logical_key):
    old_cfg, old_p = saved.get("config", {}), saved.get("problem", {})
    if (old_cfg.get("method") != cfg["method"]
            or old_cfg.get("training_seed") != cfg["training_seed"]
            or old_p.get("name") != p.get("name")):
        return False
    if logical_key is not None and (run / "study_record.json").exists():
        record = read_json(run / "study_record.json")
        if all(record.get(k) == v for k, v in logical_key.items()):
            return True
    old_identity = saved.get("identity", {})
    return (old_identity.get("X_obs_sha256") == identity["X_obs_sha256"]
            and old_identity.get("y_obs_sha256") == identity["y_obs_sha256"])


def _artifact_check(run, saved, status, cfg, signature, identity):
    needed = ["metrics.json", "predictions.npz", "weights_final.npz", "config.json",
              "observations.npz", "evaluation_grid.npz"]
    missing = [name for name in needed if not (run / name).is_file()]
    if missing:
        return "completed_artifacts_missing: " + ", ".join(missing)
    metrics = read_json(run / "metrics.json")
    if not (saved.get("signature") == status.get("signature") == metrics.get("signature") == signature):
        return "completed_signature_mismatch"
    if pinn._metadata_hash(saved.get("identity", {})) != signature:
        return "saved_identity_hash_mismatch"
    if saved.get("identity") != identity:
        return "saved_identity_differs_from_request"
    stored_identity_cfg = {k: v for k, v in saved.get("config", {}).items()
                           if k not in {"resume", "print_every", "run_id"}}
    if (saved.get("problem") != identity["problem"]
            or stored_identity_cfg != identity["config"]):
        return "saved_configuration_not_consistent_with_identity"
    if (status.get("profile") != cfg["profile"] or metrics.get("profile") != cfg["profile"]
            or saved.get("config", {}).get("profile") != cfg["profile"]):
        return "completed_profile_mismatch"
    if (int(status.get("epochs_completed", -1)) != int(cfg["epochs"])
            or int(metrics.get("epochs_completed", -1)) != int(cfg["epochs"])):
        return "completed_training_budget_mismatch"
    if metrics.get("method") != cfg["method"] or metrics.get("training_seed") != cfg["training_seed"]:
        return "completed_method_or_seed_mismatch"
    with np.load(run / "observations.npz", allow_pickle=False) as z:
        if (pinn._hash_array(z["X_obs"]) != identity["X_obs_sha256"]
                or pinn._hash_array(z["y_obs"]) != identity["y_obs_sha256"]):
            return "saved_observations_hash_mismatch"
    with np.load(run / "evaluation_grid.npz", allow_pickle=False) as z:
        if pinn._hash_array(z["X_eval"]) != identity["X_eval_sha256"]:
            return "saved_evaluation_grid_hash_mismatch"
        expected_times = np.asarray(cfg.get("eval_times", [0.0, float(identity["problem"]["T"])]), float)
        if not np.array_equal(z["eval_times"], expected_times):
            return "saved_evaluation_times_mismatch"
    # Verify that final arrays are readable; reconstruction accuracy is never a
    # reuse criterion.  Poor scientific outcomes must remain in the cohort.
    for filename in ("predictions.npz", "weights_final.npz"):
        with np.load(run / filename, allow_pickle=False) as z:
            if not z.files or any(not np.isfinite(z[key]).all() for key in z.files):
                return "unreadable_or_nonfinite_completed_arrays: " + filename
            if filename == "predictions.npz":
                if pinn._hash_array(z["X_eval"]) != identity["X_eval_sha256"]:
                    return "prediction_evaluation_grid_hash_mismatch"
                if pinn._hash_array(z["X_obs"]) != identity["X_obs_sha256"]:
                    return "prediction_observation_grid_hash_mismatch"
    return None


def inspect_training_job(problem, observations, config, output_dir, *, logical_key=None):
    """Inspect one exact job without importing TensorFlow or altering its files.

    ``completed`` means intact training artifacts at the declared final budget;
    it does not mean scientifically validated.  Missing downstream evaluation
    is indicated separately, so it never triggers another neural training.
    Alternative configurations for the same statistical unit are blocked.
    """
    verify_protocol_lock()
    p, cfg, identity, signature, expected_run = _identity(problem, observations, config, output_dir)
    result = dict(status="missing", run_dir=str(expected_run), run_folder=_relative(expected_run),
                  signature=signature, reason="not_yet_started", epochs_completed=0,
                  needs_evaluation=True, conflict_run_dirs=[])
    candidates, conflicts = [], []
    for path in sorted(Path(output_dir).glob("*/config.json")):
        try:
            saved = read_json(path)
            if _same_unit(saved, path.parent, p, cfg, identity, logical_key):
                if saved.get("signature") == signature:
                    candidates.append((path.parent, saved))
                else:
                    conflicts.append(str(path.parent.resolve()))
        except (OSError, ValueError, KeyError) as exc:
            if path.parent.resolve() == expected_run:
                result.update(status="blocked", reason=f"unreadable_existing_config: {exc}")
                return result
    if conflicts:
        result.update(status="blocked", reason="alternative_configuration_for_same_statistical_unit",
                      conflict_run_dirs=conflicts)
        return result
    if len(candidates) > 1:
        result.update(status="blocked", reason="duplicate_matching_run_directories",
                      conflict_run_dirs=[str(p) for p, _ in candidates])
        return result
    if not candidates:
        if expected_run.exists() and any(expected_run.iterdir()):
            result.update(status="blocked", reason="existing_run_directory_without_valid_configuration")
        return result
    run, saved = candidates[0]
    result.update(run_dir=str(run.resolve()), run_folder=_relative(run))
    try:
        status = read_json(run / "status.json")
        if status.get("signature") != signature:
            result.update(status="blocked", reason="status_signature_mismatch")
            return result
        result["epochs_completed"] = int(status.get("epochs_completed", 0))
        if status.get("status") == "completed":
            error = _artifact_check(run, saved, status, cfg, signature, identity)
            if error:
                result.update(status="blocked", reason=error)
            else:
                result.update(status="completed", reason="matching_final_checkpoint_reused",
                              needs_evaluation=not (run / "evaluation.json").is_file())
            return result
        stored_cfg = {k: v for k, v in saved.get("config", {}).items()
                      if k not in {"resume", "print_every", "run_id"}}
        if (pinn._metadata_hash(saved.get("identity", {})) != signature
                or saved.get("identity") != identity
                or saved.get("problem") != identity["problem"]
                or stored_cfg != identity["config"]):
            result.update(status="blocked", reason="incomplete_run_configuration_or_identity_mismatch")
            return result
        if run.resolve() != expected_run:
            result.update(status="blocked", reason="incomplete_matching_job_moved_or_renamed")
            return result
        checkpoint_dir = run / "checkpoints"
        state_file = checkpoint_dir / "checkpoint"
        match = re.search(r'^model_checkpoint_path:\s*"([^"\n]+)"', state_file.read_text(), re.M) if state_file.is_file() else None
        # TensorFlow-generated checkpoint prefixes are relative to this folder.
        # Fail closed on moved absolute paths instead of silently starting at 0.
        prefix = match.group(1) if match else None
        usable_checkpoint = bool(prefix and Path(prefix).name == prefix
                                 and (checkpoint_dir / (prefix + ".index")).is_file()
                                 and list(checkpoint_dir.glob(prefix + ".data-*")))
        if usable_checkpoint:
            with np.load(run / "observations.npz", allow_pickle=False) as z:
                correct_obs = (pinn._hash_array(z["X_obs"]) == identity["X_obs_sha256"]
                               and pinn._hash_array(z["y_obs"]) == identity["y_obs_sha256"])
            with np.load(run / "evaluation_grid.npz", allow_pickle=False) as z:
                correct_grid = pinn._hash_array(z["X_eval"]) == identity["X_eval_sha256"]
            if not correct_obs or not correct_grid:
                result.update(status="blocked", reason="incomplete_run_saved_data_mismatch")
            elif not cfg.get("resume", True):
                result.update(status="blocked", reason="existing_checkpoint_but_resume_disabled")
            else:
                result.update(status="resumable", reason="matching_saved_training_checkpoint")
        elif result["epochs_completed"] > 0 or (checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir())):
            result.update(status="blocked", reason="recorded_training_progress_without_resume_checkpoint")
        else:
            result.update(status="missing", reason="started_without_saved_training_checkpoint")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status="blocked", reason=f"existing_artifact_validation_failed: {exc}")
    return result


def prepare_plan_row(row, overrides=None):
    """Create only missing observation files, preserving all existing data."""
    row = dict(row)
    p = problem_spec(row["case"], exposure=float(row["exposure"]) if row["case"] != "nonlinear1d" else None)
    locked = verify_protocol_lock(require_reference=p["law"] != "constant")
    reference = locked[0] if locked else None
    folder = freeze_observations(p, int(row["obs_per_dim"]), float(row["noise_level"]),
                                 int(row["noise_seed"]), reference)
    _, obs, meta = load_observations(folder)
    obs["noise_seed"] = meta["noise_seed"]
    options = {}
    if p["dim"] == 1:
        options.update(N_r=12000, N_b=200, batch_r=4096, batch_b=200, eval_per_dim=1001)
    elif p["dim"] == 3:
        options.update(N_r=500000, N_b=60000, batch_r=16384, batch_b=4096, eval_per_dim=33)
    seed = int(row["training_seed"])
    cfg = pinn.training_config("production", method=row["method"], training_seed=seed,
                               feature_seed=seed + 70000, collocation_seed=seed + 80000,
                               residual_scale=float(p["T"]), **options)
    cfg.update(overrides or {})
    if cfg["profile"] != "production":
        raise ValueError("The compact completion cohort uses production configurations only")
    training_problem = {k: v for k, v in p.items() if k not in _TRUTH_KEYS}
    return dict(problem=p, training_problem=training_problem, observations=obs,
                metadata=meta, dataset_folder=folder, config=cfg)


def plan_inventory(plan, overrides=None):
    """All declared jobs, including completed jobs and their exact identities."""
    plan = pd.DataFrame(plan).copy()
    if plan.duplicated(PLAN_KEYS).any():
        raise ValueError("The requested plan contains duplicate statistical units")
    records = []
    for _, row in plan.iterrows():
        try:
            job = prepare_plan_row(row, overrides)
            state = inspect_training_job(job["training_problem"], job["observations"], job["config"],
                                         RESULTS / "neural/production",
                                         logical_key={k: row[k] for k in PLAN_KEYS})
            records.append({**row.to_dict(), **state, "dataset_id": job["metadata"]["dataset_id"],
                            "dataset_folder": _relative(job["dataset_folder"]),
                            "observation_sha256": job["metadata"]["observation_sha256"]})
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            records.append({**row.to_dict(), "status": "blocked", "reason": str(exc),
                            "run_dir": None, "run_folder": None, "signature": None,
                            "epochs_completed": None, "needs_evaluation": False})
    return pd.DataFrame(records)


def pending_plan(plan, max_new_jobs=None, overrides=None):
    """Return the next missing/resumable jobs; completed jobs never use the cap."""
    if max_new_jobs is not None and (not isinstance(max_new_jobs, (int, np.integer)) or max_new_jobs < 0):
        raise ValueError("max_new_jobs must be None or a nonnegative integer")
    inventory = plan_inventory(plan, overrides=overrides)
    blocked = inventory[inventory.status == "blocked"]
    if len(blocked):
        reasons = blocked[["case", "method", "training_seed", "reason"]].to_dict("records")
        raise RuntimeError(f"Resolve blocked jobs before training; no job was started: {reasons}")
    pending = inventory[inventory.status.isin(["missing", "resumable"])]
    if max_new_jobs is not None:
        pending = pending.head(int(max_new_jobs))
    return pending.loc[:, pd.DataFrame(plan).columns].reset_index(drop=True)


def _repair_evaluation(row, state, overrides=None):
    job = prepare_plan_row(row, overrides)
    run = Path(state["run_dir"])
    if not (run / "study_record.json").is_file():
        write_json(run / "study_record.json", {**dict(row),
                   "dataset_id": job["metadata"]["dataset_id"],
                   "dataset_folder": _relative(job["dataset_folder"]),
                   "observation_sha256": job["metadata"]["observation_sha256"],
                   "run_folder": _relative(run), "profile": "production",
                   "protocol": "matched-interior-v1-unit-time-residual",
                   "endpoint": "final_fixed_budget", "publication_eligible": False})
    if not (run / "evaluation.json").is_file():
        if job["problem"]["law"] == "constant":
            flow.linear_run_evaluation(run, job["problem"], job["observations"], closure=True)
        else:
            flow.evaluate_nonlinear_run(run, job["problem"], job["observations"])


def run_missing(plan, max_new_jobs=None, overrides=None):
    """Reuse completed runs and train/resume only the next requested missing jobs.

    A cap of 1 means the next unfinished job, even when earlier plan rows are
    complete.  Evaluation-only repair does not count as a new training.  No
    existing completed run is rewritten when its evaluation already exists.
    """
    plan = pd.DataFrame(plan).copy()
    todo = pending_plan(plan, max_new_jobs=max_new_jobs, overrides=overrides)
    before = plan_inventory(plan, overrides=overrides)
    print(f"Reusing {int((before.status == 'completed').sum())} completed jobs; "
          f"queueing {len(todo)} missing/resumable jobs.")
    for _, state in before[before.status == "completed"].iterrows():
        if state["needs_evaluation"] or not (Path(state["run_dir"]) / "study_record.json").is_file():
            _repair_evaluation({k: state[k] for k in plan.columns}, state, overrides)
    if len(todo):
        flow.run_pinn_plan(todo, profile="production", max_runs=None, overrides=overrides)
    after = plan_inventory(plan, overrides=overrides)
    destination = RESULTS / "reduced_plan"
    destination.mkdir(parents=True, exist_ok=True)
    cohort = "_".join(sorted(plan.case.unique()))
    plan.to_csv(destination / f"{cohort}_declared_plan.csv", index=False)
    after.to_csv(destination / f"{cohort}_inventory.csv", index=False)
    return after
