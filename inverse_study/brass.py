"""Brass data contract and physical-unit assessment for the completion notebooks.

The five supplied CSVs are digitized data, not newly measured temperatures.
Only terminal INTERIOR values and endpoint histories enter inverse training.
Initial/intermediate interiors are held out from the objective, but originate
from the same experimental run and do not establish independent calibration.
No digitization uncertainty or new scientific result is manufactured here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LENGTH_M = 0.034
FINAL_TIME_S = 9.0
OFFSET_K = 520.0
SCALE_K = 190.0
B0 = 0.287265051903114
B1 = 0.011656401384083
FILES = (
    "dirichlet_boundary_history.csv", "hidden_initial_truth_t0.csv",
    "terminal_observations_t9.csv", "thermal_diffusivity_model.csv",
    "withheld_intermediate_validation.csv",
)


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(table, columns, name):
    if not set(columns).issubset(table.columns):
        raise ValueError(f"{name} lacks columns {set(columns)-set(table.columns)}")
    if not np.isfinite(table[list(columns)].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} has missing/nonfinite physical data")


def _interior(table):
    return table[(table.x_m > 1e-12) & (table.x_m < LENGTH_M - 1e-12)].copy()


def load_brass(data_dir, uncertainty_K=None):
    """Return (problem, observations, heldout_dataframe, provenance).

    X_obs has shape (48,1); y_obs has shape (48,). Heldout has 240 rows: the
    48 interiors at t=0,1.8,3.6,5.4,7.2 s. All endpoints are excluded from
    validation because those same values prescribe the boundary histories.

    The default observations['noise_norm'] is None: the supplied files do
    not quantify digitization/measurement uncertainty. If uncertainty_K is
    supplied, it is explicitly interpreted as an ASSUMED per-point absolute
    error bound, not a measured standard deviation or confidence interval.
    """
    data_dir = Path(data_dir)
    missing = [name for name in FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing supplied brass CSVs in {data_dir}: {missing}")
    frames = {name: pd.read_csv(data_dir / name) for name in FILES}
    boundary = frames[FILES[0]].sort_values("time_s").reset_index(drop=True)
    initial = frames[FILES[1]].sort_values("x_m").reset_index(drop=True)
    terminal = frames[FILES[2]].sort_values("x_m").reset_index(drop=True)
    law = frames[FILES[3]]
    intermediate = frames[FILES[4]].sort_values(["time_s", "x_m"]).reset_index(drop=True)
    _finite(boundary, ["time_s", "T_left_K", "T_right_K"], FILES[0])
    for name, table in ((FILES[1], initial), (FILES[2], terminal), (FILES[4], intermediate)):
        cols = ["x_m", "temperature_K"] + (["time_s"] if "time_s" in table else [])
        _finite(table, cols, name)
        if np.any(table.x_m < -1e-12) or np.any(table.x_m > LENGTH_M + 1e-12):
            raise ValueError(f"{name} lies outside the declared 34 mm observation segment")
    if len(law) != 1:
        raise ValueError("Expected exactly one reported brass diffusivity law")
    _finite(law, ["A0_m2_s", "A1_m2_s_K"], FILES[3])
    a0, a1 = float(law.A0_m2_s.iloc[0]), float(law.A1_m2_s_K.iloc[0])
    calculated_b0 = FINAL_TIME_S * (a0 + a1 * OFFSET_K) / LENGTH_M**2
    calculated_b1 = FINAL_TIME_S * a1 * SCALE_K / LENGTH_M**2
    if not np.allclose([calculated_b0, calculated_b1], [B0, B1], rtol=2e-13, atol=1e-14):
        raise ValueError("CSV diffusivity law disagrees with the archived nondimensional B0/B1")
    times = boundary.time_s.to_numpy(float)
    if len(times) != 6 or not np.allclose(times, np.linspace(0, FINAL_TIME_S, 6)):
        raise ValueError("Expected the six supplied boundary-history times 0...9 s")
    if len(initial) != 50 or len(terminal) != 50 or len(intermediate) != 200:
        raise ValueError("Expected supplied profile counts 50 initial, 50 terminal, 200 intermediate")
    if not np.allclose(terminal.time_s, FINAL_TIME_S):
        raise ValueError("Training observations must be the terminal 9 s profile")
    initial = initial.assign(time_s=0.0)
    all_profiles = pd.concat([initial, intermediate, terminal], ignore_index=True)
    consistency = []
    for t, group in all_profiles.groupby("time_s", sort=True):
        group = group.sort_values("x_m")
        if len(group) != 50 or group.x_m.duplicated().any():
            raise ValueError(f"Profile at t={t:g} s must contain 50 unique spatial samples")
        if not np.allclose(group.x_m.iloc[[0, -1]], [0.0, LENGTH_M], atol=1e-12):
            raise ValueError("Every source profile must include the known endpoints")
        if not np.allclose(group.x_m, terminal.x_m, atol=1e-12):
            raise ValueError("Source profiles must share the same sensor coordinates")
        boundary_row = boundary[np.isclose(boundary.time_s, t)]
        if len(boundary_row) != 1:
            raise ValueError(f"No unique boundary history at {t:g} s")
        endpoint_error = group.temperature_K.iloc[[0, -1]].to_numpy() - boundary_row[["T_left_K", "T_right_K"]].to_numpy().ravel()
        if np.max(np.abs(endpoint_error)) > 2e-6:
            raise ValueError("Profile endpoints disagree with the separately stored boundary history")
        consistency.append(float(np.max(np.abs(endpoint_error))))

    problem = dict(name="brass", dim=1, bounds=[[0.0, 1.0]], T=1.0, Fo=1.0,
                   law="brass", B0=B0, B1=B1,
                   boundary_times=(times / FINAL_TIME_S).tolist(),
                   boundary_left=((boundary.T_left_K.to_numpy() - OFFSET_K) / SCALE_K).tolist(),
                   boundary_right=((boundary.T_right_K.to_numpy() - OFFSET_K) / SCALE_K).tolist(),
                   physical_length_m=LENGTH_M, physical_final_time_s=FINAL_TIME_S,
                   temperature_offset_K=OFFSET_K, temperature_scale_K=SCALE_K)
    train = _interior(terminal)
    observations = dict(X_obs=(train.x_m.to_numpy(float) / LENGTH_M)[:, None],
                        y_obs=(train.temperature_K.to_numpy(float) - OFFSET_K) / SCALE_K,
                        noise_norm=None, noise_status="unknown_not_reported_in_supplied_CSVs",
                        dataset_id="brass_terminal48_" + _file_hash(data_dir / FILES[2])[:12],
                        boundary_policy="measured_endpoint_histories_separate_from_interior_terminal_data")
    heldout = _interior(pd.concat([initial, intermediate], ignore_index=True)).sort_values(["time_s", "x_m"]).reset_index(drop=True)
    heldout["xi"] = heldout.x_m / LENGTH_M
    heldout["tau"] = heldout.time_s / FINAL_TIME_S
    heldout["theta"] = (heldout.temperature_K - OFFSET_K) / SCALE_K
    heldout["role"] = np.where(heldout.time_s == 0, "heldout_initial_interior", "heldout_same_run_intermediate_interior")
    provenance = dict(
        files={name: {"sha256": _file_hash(data_dir / name), "rows": len(frames[name])} for name in FILES},
        data_kind="supplied_digitized_profiles_not_original_digital_instrument_records",
        n_terminal_training=48, n_heldout_initial=48, n_heldout_intermediate=192,
        n_boundary_values=12, n_distinct_experimental_runs=1,
        maximum_endpoint_consistency_error_K=max(consistency),
        coefficient_source_note=str(law["note"].iloc[0]),
        coefficient_note_status="copied_from_supplied_CSV_not_independently_verified_against_original_article",
        calibration_independence="unresolved; reported 3.6-5.4 s calibration overlaps the same-run heldout time range",
        digitization_uncertainty_K=None,
        uncertainty_status="unknown; no uncertainty bars should be claimed from these CSVs alone",
        release_status="data_loaded_for_completion_work_not_publication_approval",
        pending_checks=[
            "Verify the source article's coefficient-identification data and whether initial/profile data enter calibration.",
            "Obtain original digital profiles, or document repeat digitization and axis calibration.",
            "Document uncertainty of temperatures, positions, time nodes and boundary interpolation.",
            "Interpret intermediate interiors as same-run withheld data, not an independent experiment.",
            "Check plausible lateral losses, diffusivity and boundary-history sensitivity before assigning a bias cause.",
            "Confirm attribution and reuse status of the adapted experimental schematic.",
        ],
    )
    if uncertainty_K is not None:
        observations = baseline_observations(observations, uncertainty_K)
        provenance["assumed_pointwise_error_bound_K"] = float(uncertainty_K)
        provenance["uncertainty_status"] = "user-specified sensitivity assumption; not measured uncertainty"
    return problem, observations, heldout, provenance


def baseline_observations(observations, uncertainty_K=None, interpretation="pointwise_bound"):
    """Add a declared L2 discrepancy scale for a classical experimental fit.

    For a per-point absolute bound b_K, ||epsilon||_2 <= sqrt(n)*b_K/190.
    'rms_scenario' uses the same numeric conversion as an ASSUMED RMS scale;
    it is not a high-probability bound and never creates a measured error bar.
    Both interpretations require provenance outside the current source files.
    """
    if uncertainty_K is None or not np.isfinite(uncertainty_K) or float(uncertainty_K) <= 0:
        raise ValueError("Brass uncertainty is unknown. Supply a documented positive uncertainty_K before discrepancy tuning.")
    if interpretation not in {"pointwise_bound", "rms_scenario"}:
        raise ValueError("interpretation must be 'pointwise_bound' or 'rms_scenario'")
    result = deepcopy(observations)
    result.update(noise_norm=float(np.sqrt(len(result["y_obs"])) * float(uncertainty_K) / SCALE_K),
                  assumed_uncertainty_K=float(uncertainty_K), noise_interpretation=interpretation,
                  noise_status="declared_assumption_not_measured_digitization_uncertainty")
    return result


def physical_metrics(predtheta, true_theta):
    """Unweighted errors over the supplied evaluation points, in kelvin.

    shifted_relative_L2 is the historical normalization using (T-520)/190;
    it must never be described as a percentage error in absolute Kelvin.
    Range normalization uses the range of THIS heldout reference profile.
    """
    pred = np.asarray(predtheta, dtype=float).reshape(-1)
    ref = np.asarray(true_theta, dtype=float).reshape(-1)
    if pred.shape != ref.shape or not len(ref) or not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise ValueError("Prediction/reference arrays must be finite, nonempty and equal in shape")
    error = SCALE_K * (pred - ref)
    rms = float(np.sqrt(np.mean(error**2)))
    span = float(SCALE_K * np.ptp(ref))
    denominator = float(np.linalg.norm(ref))
    return dict(n_points=len(ref), RMSE_K=rms, MAE_K=float(np.mean(np.abs(error))),
                bias_K=float(np.mean(error)), maxabs_K=float(np.max(np.abs(error))),
                reference_range_K=span, range_normalized_RMSE=rms / span if span > 0 else None,
                shifted_relative_L2=float(np.linalg.norm(pred-ref) / denominator) if denominator > 0 else None)


def evaluate_heldout(predictor, heldout):
    """Evaluate frozen predictions: predictor(X_space, paired_times)->theta.

    Returns (per_time_metrics, point_predictions). This function neither trains
    nor selects a candidate. Heldout errors must not be fed back into tuning.
    """
    needed = {"xi", "tau", "theta", "time_s", "temperature_K", "role"}
    if not needed.issubset(heldout.columns):
        raise ValueError(f"heldout lacks {needed-set(heldout.columns)}")
    if np.any(heldout.xi <= 0) or np.any(heldout.xi >= 1) or np.any(heldout.tau >= 1):
        raise ValueError("Only initial/intermediate INTERIORS belong in withheld evaluation")
    points = heldout.copy()
    predictions = np.asarray(predictor(points[["xi"]].to_numpy(), points.tau.to_numpy()), dtype=float).ravel()
    if len(predictions) != len(points) or not np.isfinite(predictions).all():
        raise ValueError("predictor returned nonfinite values or the wrong number of predictions")
    points["prediction_theta"] = predictions
    points["prediction_K"] = OFFSET_K + SCALE_K * predictions
    points["error_K"] = points.prediction_K - points.temperature_K
    rows = []
    for t, group in points.groupby("time_s", sort=True):
        rows.append(dict(time_s=float(t), tau=float(group.tau.iloc[0]),
                         role=str(group.role.iloc[0]), **physical_metrics(group.prediction_theta, group.theta)))
    return pd.DataFrame(rows), points


def evaluate_saved_pinn(run_dir, heldout, checkpoint="final"):
    """Physical-unit heldout metrics from an already trained saved network."""
    from .pinn import predict_saved
    return evaluate_heldout(lambda X, times: predict_saved(run_dir, X, times=times, checkpoint=checkpoint), heldout)


def replay_initial(problem, initial_predictor, observations, heldout, nx=401,
                   rtol=1e-8, atol=1e-10, max_step=0.02):
    """Independently evolve a fixed recovered initial trace with the FD/BDF solver.

    initial_predictor takes spatial shape (nx,1), returns theta(t=0).
    No heldout temperature enters forward propagation. Endpoint compatibility
    corrections are reported. The terminal comparison is a fit to TRAINING
    observations and must not be labelled heldout validation.
    """
    from .numerics import forward_1d
    if problem.get("law") != "brass" or problem.get("dim") != 1:
        raise ValueError("Use this physical-unit helper for the brass case only")
    x = np.linspace(0.0, 1.0, int(nx))
    initial = np.asarray(initial_predictor(x[:, None]), dtype=float).ravel()
    if initial.shape != x.shape or not np.isfinite(initial).all():
        raise ValueError("initial_predictor must return one finite value per grid point")
    times = np.unique(np.r_[0.0, heldout.tau.to_numpy(float), float(problem["T"])])
    forward = forward_1d(problem, x, initial, times=times, rtol=rtol, atol=atol, max_step=max_step)

    def interpolated(X, paired_times):
        values = np.empty(len(X))
        for t in np.unique(paired_times):
            index = np.flatnonzero(np.isclose(times, t, rtol=0, atol=1e-12))
            if len(index) != 1:
                raise ValueError("Forward output does not uniquely contain a required validation time")
            mask = np.isclose(paired_times, t, rtol=0, atol=1e-12)
            values[mask] = np.interp(X[mask, 0], x, forward["u"][index[0]])
        return values

    metrics, points = evaluate_heldout(interpolated, heldout)
    terminal_prediction = np.interp(np.asarray(observations["X_obs"])[:, 0], x, forward["terminal"])
    terminal_metrics = physical_metrics(terminal_prediction, observations["y_obs"])
    terminal_metrics.update(role="terminal_training_observation_consistency_not_heldout",
                            absolute_observation_residual_norm_theta=float(np.linalg.norm(terminal_prediction - observations["y_obs"])))
    return dict(nx=int(nx), forward=forward, heldout_metrics=metrics, heldout_points=points,
                terminal_prediction=terminal_prediction, terminal_metrics=terminal_metrics,
                initial_boundary_correction_K=SCALE_K * np.asarray(forward["initial_boundary_correction"]),
                evidence_status="independent_forward_replay_pending_mesh_tolerance_and_provenance_checks")


def replay_refinement(problem, initial_predictor, observations, heldout,
                      meshes=(101, 201, 401), rtol=1e-8, atol=1e-10, max_step=0.02):
    """Spatial sensitivity at fixed tolerances; no automatic convergence verdict.

    The last (finest) mesh is the comparison grid, not exact truth. Repeat
    with stricter rtol/atol/max_step to assess time-integration error separately.
    Differences in terminal theta units may inform a numerical-noise-floor
    study, but are not automatically a certified error bound.
    """
    meshes = tuple(int(n) for n in meshes)
    if len(meshes) < 2 or any(n < 3 for n in meshes) or any(b <= a for a, b in zip(meshes, meshes[1:])):
        raise ValueError("Provide at least two strictly increasing mesh sizes")
    replays = [replay_initial(problem, initial_predictor, observations, heldout, n, rtol, atol, max_step) for n in meshes]
    finest = replays[-1]
    rows = []
    for run in replays:
        diff = np.asarray(run["terminal_prediction"]) - finest["terminal_prediction"]
        point_diff_K = SCALE_K * (run["heldout_points"].prediction_theta.to_numpy() - finest["heldout_points"].prediction_theta.to_numpy())
        rows.append(dict(nx=run["nx"], rtol=rtol, atol=atol, max_step=max_step,
                         terminal_fit_RMSE_K=run["terminal_metrics"]["RMSE_K"],
                         terminal_l2_difference_to_finest_theta=float(np.linalg.norm(diff)),
                         terminal_RMSE_difference_to_finest_K=float(SCALE_K*np.sqrt(np.mean(diff**2))),
                         heldout_max_difference_to_finest_K=float(np.max(np.abs(point_diff_K))),
                         boundary_correction_maxabs_K=float(np.max(np.abs(run["initial_boundary_correction_K"]))),
                         wall_seconds=run["forward"]["wall_seconds"],
                         comparison="difference_to_finest_numerical_grid_not_exact_error"))
    return pd.DataFrame(rows)


def sensitivity_scenarios(problem, diffusivity_shifts=(-0.10, -0.05, 0.05, 0.10),
                          boundary_offsets_K=(-1.0, 1.0)):
    """Declared model scenarios, not measured uncertainties or confidence intervals.

    Each perturbation is varied separately. Fo multiplies the entire brass law
    (both B0 and B1) consistently in the PINN and independent forward solver.
    Boundary offsets affect both endpoint histories; terminal interior data stay
    unchanged, isolating uncertainty in the imposed histories.
    """
    if problem.get("law") != "brass":
        raise ValueError("Expected the brass problem")
    rows = [dict(scenario="nominal", assumption="reported coefficients and supplied endpoint histories", problem=deepcopy(problem))]
    for shift in diffusivity_shifts:
        shift = float(shift)
        if not np.isfinite(shift) or shift <= -1:
            raise ValueError("Diffusivity shifts must be finite and exceed -1")
        p = deepcopy(problem)
        p["Fo"] = float(problem["Fo"]) * (1 + shift)
        name = f"diffusivity_{shift:+.0%}".replace("+", "plus").replace("-", "minus").replace("%", "pct")
        p["name"] = "brass_" + name
        rows.append(dict(scenario=name, assumption=f"ASSUMED {shift:+.0%} multiplier of the entire diffusivity law; no measured probability", problem=p))
    for offset in boundary_offsets_K:
        offset = float(offset)
        if not np.isfinite(offset):
            raise ValueError("Boundary offsets must be finite")
        p = deepcopy(problem)
        for key in ("boundary_left", "boundary_right"):
            p[key] = (np.asarray(problem[key], float) + offset / SCALE_K).tolist()
        name = f"boundary_{offset:+g}K".replace("+", "plus").replace("-", "minus")
        p["name"] = "brass_" + name
        rows.append(dict(scenario=name, assumption=f"ASSUMED common {offset:+g} K offset of both endpoint histories; not a measured error bar", problem=p))
    return rows


def retraining_plan(problem, training_seeds=(2026, 2027, 2028, 2029, 2030),
                    methods=("plain", "fourier_fixed", "fourier_adaptive"),
                    include_sensitivity=False, profile="production"):
    """Return jobs only; no training runs start here and no heldout data enter.

    Nominal production retains the archived brass architecture (3 x 60 tanh),
    16 Fourier frequencies/coordinate, 60,000 Adam steps, and 12,000/200
    residual/boundary points. The full-batch budgets are explicit. Sensitivity
    retraining is optional after nominal fits and independent replay checks.
    """
    from .pinn import training_config, METHODS
    if any(method not in METHODS for method in methods):
        raise ValueError(f"Methods must belong to {METHODS}")
    scenarios = sensitivity_scenarios(problem) if include_sensitivity else sensitivity_scenarios(problem)[:1]
    jobs = []
    for scenario in scenarios:
        for method in methods:
            for seed in training_seeds:
                settings = dict(method=method, training_seed=int(seed), eval_times=np.linspace(0, 1, 6).tolist())
                if profile == "production":
                    settings.update(hidden_width=60, hidden_depth=3, epochs=60000,
                                    N_r=12000, N_b=200, batch_r=12000, batch_b=200,
                                    features_per_coordinate=16, eval_per_dim=1001)
                config = training_config(profile=profile, **settings)
                config["run_id"] = f"brass__{scenario['scenario']}__{method}__seed{int(seed)}"
                jobs.append(dict(scenario=scenario["scenario"], assumption=scenario["assumption"],
                                 problem=deepcopy(scenario["problem"]), config=config,
                                 evidence_status="planned_only_not_run"))
    return jobs


def freeze_candidate(run_dir, destination, checkpoint="final", selection_note="final fixed-budget checkpoint; no heldout-error selection"):
    """Write a candidate identity before inspecting heldout errors.

    This records the author's declared selection rule and hashes; it is not
    publication approval or proof that a caller has never inspected holdouts.
    A pre-existing destination must match rather than be silently overwritten.
    """
    run_dir, destination = Path(run_dir), Path(destination)
    if checkpoint not in {"final", "best_truth_free"}:
        raise ValueError("Unsupported checkpoint")
    files = [run_dir / "config.json", run_dir / f"weights_{checkpoint}.npz"]
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("The requested trained candidate configuration/weights are not saved")
    record = dict(run_dir=str(run_dir.resolve()), checkpoint=checkpoint,
                  files={path.name: _file_hash(path) for path in files},
                  declared_selection_rule=str(selection_note),
                  evidence_status="candidate_frozen_for_withheld_evaluation_not_publication_approval")
    if destination.exists():
        old = json.loads(destination.read_text())
        if any(old.get(key) != value for key, value in record.items()):
            raise RuntimeError("Frozen candidate differs; choose a new destination instead of overwriting the record")
        return old
    record["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2) + "\n")
    return record
