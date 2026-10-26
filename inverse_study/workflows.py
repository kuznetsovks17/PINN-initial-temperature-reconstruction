"""Small orchestration functions used by the numbered notebooks.

No job is a publication result merely because it completed. Smoke, legacy and
prospective candidate outputs have separate locations and explicit metadata.
"""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .common import (ROOT, RESULTS, DATA, read_json, write_json, metadata_hash,
                     sha256_file, array_hash, problem_spec, tensor_grid,
                     exact_linear, initial_jump, relative_l2, jump_metrics,
                     freeze_observations, load_observations, linear_run_evaluation)


def linear_plan(stage="screen"):
    """Predeclared paired study, with broad extensions kept out of the core pilot."""
    methods = ["plain", "fourier_fixed", "fourier_adaptive"]
    rows = []
    if stage == "screen":
        settings = [(0.0, 33, [0], [2026], 0.02), (0.05, 33, [3100], [2026], 0.02)]
    elif stage == "pilot":
        settings = [(0.0, 33, [0], range(2026, 2031), 0.02),
                    (0.05, 33, range(3100, 3105), range(2026, 2031), 0.02)]
    elif stage == "confirmatory":
        # Independent noise and training seeds, disjoint from the pilot.
        settings = [(0.05, 33, range(4100, 4110), range(4026, 4036), 0.02)]
    elif stage == "extensions":
        methods = ["plain", "fourier_adaptive"]
        settings = [(rho, 33, range(3100, 3103), range(2026, 2029), 0.02) for rho in [.01, .10, .15, .20]]
        settings += [(0.05, n, range(3100, 3103), range(2026, 2029), 0.02) for n in [5, 9, 17]]
        settings += [(0.05, 33, range(3100, 3103), range(2026, 2029), exposure) for exposure in [.005, .08]]
    else:
        raise ValueError("stage must be screen, pilot, confirmatory, or extensions")
    for rho, n, noise_seeds, training_seeds, exposure in settings:
        for ns in noise_seeds:
            for ts in training_seeds:
                for method in methods:
                    rows.append(dict(stage=stage, case="smooth2d", exposure=exposure,
                                     obs_per_dim=n, noise_level=rho, noise_seed=ns,
                                     training_seed=ts, method=method))
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def nonlinear_plan(stage="screen"):
    if stage not in {"screen", "pilot"}:
        raise ValueError("Choose screen or pilot; add further settings after the reference gate")
    methods = ["plain", "fourier_adaptive"]
    ts = [2026] if stage == "screen" else list(range(2026, 2029))
    rows = []
    for rho in [0.0, 0.05]:
        noise_seeds = [0] if rho == 0 else ([3100] if stage == "screen" else range(3100, 3103))
        for ns in noise_seeds:
            for seed in ts:
                for method in methods:
                    rows.append(dict(stage=stage, case="nonlinear1d", exposure=0.1,
                                     obs_per_dim=257, noise_level=rho, noise_seed=ns,
                                     training_seed=seed, method=method))
    return pd.DataFrame(rows)


def dimensional_plan():
    rows = []
    for d, n in [(1, 257), (2, 33), (3, 17)]:
        for rho in ([0.0, 0.05] if d == 3 else [0.0]):
            rows.append(dict(stage="feasibility", case=f"dimension{d}d", exposure=0.02,
                             obs_per_dim=n, noise_level=rho, noise_seed=0 if rho == 0 else 3100,
                             training_seed=2026, method="fourier_adaptive"))
    return pd.DataFrame(rows)


def reference_convergence(meshes=(201, 401, 801), max_steps=(0.02, 0.01, 0.005),
                          rtol=1e-9, atol=1e-11, acceptance_relative=1e-3):
    """Space refinement then time control at the finest spatial resolution.

    The last mesh difference is an empirical error indicator, not a rigorous
    bound. Its norm at 255 sensors is stored for discrepancy sensitivity.
    """
    from .numerics import forward_1d
    meshes = tuple(int(n) for n in meshes)
    max_steps = tuple(float(step) for step in max_steps)
    if len(meshes) < 2 or meshes[0] < 5 or any(b <= a for a, b in zip(meshes, meshes[1:])):
        raise ValueError("Use at least two strictly increasing spatial node counts, each >=5")
    if not max_steps or not all(np.isfinite(step) and step > 0 for step in max_steps):
        raise ValueError("max_steps must contain positive finite step caps")
    if not all(np.isfinite(v) and v > 0 for v in (rtol, atol, acceptance_relative)):
        raise ValueError("Tolerances and acceptance_relative must be positive and finite")
    p = problem_spec("nonlinear1d")
    out = RESULTS / "verification"
    out.mkdir(parents=True, exist_ok=True)
    _, Xobs = tensor_grid(1, 257, interior=True)
    rows, solutions = [], []
    for nx in meshes:
        x = np.linspace(0, 1, int(nx))
        solution = forward_1d(p, x, initial_jump(x), [0, .25, .5, .75, 1.0],
                              rtol=rtol, atol=atol, max_step=min(max_steps))
        solutions.append(solution)
        rows.append(dict(kind="space", nx=nx, max_step=min(max_steps), rtol=rtol, atol=atol,
                         wall_seconds=solution["wall_seconds"]))
    finest = solutions[-1]
    fine_obs = np.interp(Xobs[:, 0], finest["x"], finest["terminal"])
    for row, solution in zip(rows, solutions):
        values = np.interp(Xobs[:, 0], solution["x"], solution["terminal"])
        row["terminal_difference_l2"] = float(np.linalg.norm(values - fine_obs))
        row["terminal_difference_relative_l2"] = relative_l2(values, fine_obs)
    time_values = []
    for step in max_steps:
        sol = forward_1d(p, finest["x"], initial_jump(finest["x"]), [1.0],
                         rtol=rtol, atol=atol, max_step=float(step))
        values = np.interp(Xobs[:, 0], sol["x"], sol["terminal"])
        time_values.append(values)
        rows.append(dict(kind="time", nx=len(finest["x"]), max_step=step, rtol=rtol, atol=atol,
                         terminal_difference_l2=float(np.linalg.norm(values - fine_obs)),
                         terminal_difference_relative_l2=relative_l2(values, fine_obs),
                         wall_seconds=sol["wall_seconds"]))
    # Tighten BDF tolerances independently of max_step.
    tight = forward_1d(p, finest["x"], initial_jump(finest["x"]), [1.0],
                       rtol=rtol / 10, atol=atol / 10, max_step=min(max_steps))
    tight_values = np.interp(Xobs[:, 0], tight["x"], tight["terminal"])
    tolerance_difference = float(np.linalg.norm(tight_values - fine_obs))
    max_step_difference = max(float(np.linalg.norm(values - fine_obs)) for values in time_values)
    temporal_difference = max(tolerance_difference, max_step_difference)
    rows.append(dict(kind="tolerance", nx=len(finest["x"]), max_step=min(max_steps), rtol=rtol/10, atol=atol/10,
                     terminal_difference_l2=tolerance_difference,
                     terminal_difference_relative_l2=relative_l2(tight_values, fine_obs),
                     wall_seconds=tight["wall_seconds"]))
    spatial_difference = rows[len(meshes)-2]["terminal_difference_l2"]
    indicator = max(spatial_difference, temporal_difference, 1e-12)
    passed = indicator / max(np.linalg.norm(fine_obs), 1e-15) < acceptance_relative
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "nonlinear_reference_convergence.csv", index=False)
    np.savez_compressed(out / "nonlinear_reference.npz", x=finest["x"], t=finest["t"],
                        u=finest["u"], terminal=finest["terminal"])
    write_json(out / "nonlinear_reference_check.json", dict(
        passed=bool(passed), status="empirical_reference_gate" if passed else "refine_before_production",
        meshes=list(meshes), max_steps=list(max_steps), rtol=rtol, atol=atol,
        relative_acceptance=acceptance_relative, sensor_difference_l2=indicator,
        spatial_sensor_difference_l2=spatial_difference,
        time_step_sensor_difference_l2=max_step_difference,
        time_tolerance_sensor_difference_l2=tolerance_difference,
        convergence_scope="terminal field at 255 interior observation locations",
        suggested_numerical_floor=2 * indicator,
        caveat="Empirical two-grid estimate, not a rigorous bound; rerun with finer grid for clean inverse claims.",
        problem_sha256=metadata_hash(p),
        forward_source_sha256=sha256_file(ROOT / "inverse_study/numerics.py"),
        reference_sha256=sha256_file(out / "nonlinear_reference.npz")))
    return frame, read_json(out / "nonlinear_reference_check.json")


def load_reference(require_pass=True):
    file = RESULTS / "verification/nonlinear_reference.npz"
    meta = read_json(file.with_name("nonlinear_reference_check.json"))
    if require_pass and not meta["passed"]:
        raise RuntimeError("Reference convergence gate failed; refine notebook 01 before production")
    if sha256_file(file) != meta["reference_sha256"]:
        raise RuntimeError("Reference hash mismatch")
    if meta.get("problem_sha256") != metadata_hash(problem_spec("nonlinear1d")):
        raise RuntimeError("Reference problem definition changed; rerun notebook 01")
    if meta.get("forward_source_sha256") != sha256_file(ROOT / "inverse_study/numerics.py"):
        raise RuntimeError("Forward solver source changed; rerun notebook 01 before reusing its reference")
    with np.load(file, allow_pickle=False) as z:
        reference = {key: z[key].copy() for key in z.files}
    return reference, meta


def run_pinn_plan(plan, profile="production", max_runs=None, overrides=None):
    from .pinn import train_inverse, training_config
    records = []
    for _, row in plan.head(max_runs if max_runs is not None else len(plan)).iterrows():
        p = problem_spec(row["case"], exposure=float(row["exposure"]) if row["case"] != "nonlinear1d" else None)
        reference = load_reference(require_pass=(profile == "production"))[0] if p["law"] != "constant" else None
        folder = freeze_observations(p, int(row.obs_per_dim), float(row.noise_level), int(row.noise_seed), reference)
        _, obs, meta = load_observations(folder)
        obs["noise_seed"] = meta["noise_seed"]
        case_cfg = {}
        if profile == "production" and p["dim"] == 1:
            case_cfg.update(N_r=12000, N_b=200, batch_r=4096, batch_b=200, eval_per_dim=1001)
        elif profile == "production" and p["dim"] == 3:
            case_cfg.update(N_r=500000, N_b=60000, batch_r=16384, batch_b=4096, eval_per_dim=33)
        cfg = training_config(profile, method=row.method, training_seed=int(row.training_seed),
                              feature_seed=int(row.training_seed) + 70000,
                              collocation_seed=int(row.training_seed) + 80000,
                              residual_scale=float(p["T"]), **case_cfg)
        cfg.update(overrides or {})
        if profile == "smoke":
            cfg["profile"] = "smoke"
        training_problem = {k: v for k, v in p.items() if k not in {"modes", "amplitudes", "initial_left", "initial_right", "jump_x"}}
        result = train_inverse(training_problem, obs, cfg, RESULTS / "neural" / profile)
        run = Path(result["run_dir"])
        record = dict(row.to_dict(), dataset_id=meta["dataset_id"],
                      dataset_folder=str(folder.relative_to(ROOT)),
                      observation_sha256=meta["observation_sha256"],
                      run_folder=str(run.relative_to(ROOT)), profile=profile,
                      protocol="matched-interior-v1-unit-time-residual",
                      endpoint="final_fixed_budget", publication_eligible=False)
        write_json(run / "study_record.json", record)
        if p["law"] == "constant":
            record.update(linear_run_evaluation(run, p, obs, closure=True))
        else:
            record.update(evaluate_nonlinear_run(run, p, obs))
        records.append(record)
    frame = pd.DataFrame(records)
    if len(frame):
        destination = RESULTS / "plans"
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / f"latest_{profile}_completed.csv", index=False)
    return frame


def evaluate_nonlinear_run(run, problem, observations, nx=801):
    from .numerics import forward_1d
    from .pinn import predict_saved
    x = np.linspace(0, 1, int(nx))
    u0 = predict_saved(run, x[:, None], times=0.0).ravel()
    replay = forward_1d(problem, x, u0, [problem["T"]], rtol=1e-8, atol=1e-10, max_step=.005)
    at_sensors = np.interp(observations["X_obs"][:, 0], x, replay["terminal"])
    ans = jump_metrics(x, u0)
    ans.update(independent_observation_residual=float(np.linalg.norm(at_sensors-observations["y_obs"])),
               initial_boundary_linf=float(np.max(np.abs(replay["initial_boundary_correction"]))),
               forward_replay_nx=nx, forward_replay_requires_refinement=True)
    write_json(Path(run) / "evaluation.json", ans)
    return ans


def run_spectral_case(folder, kmax=12, methods=("tikhonov", "tsvd"), profile="candidate", prior_order=0):
    from .numerics import solve_spectral, evaluate_spectral
    p, obs, meta = load_observations(folder)
    if p["law"] != "constant":
        raise ValueError("Use the nonlinear forward-observation comparator for nonlinear cases")
    _, Xeval = tensor_grid(p["dim"], 101 if p["dim"] <= 2 else 25, bounds=p["bounds"])
    rows = []
    for method in methods:
        signature = metadata_hash(dict(dataset=meta, method=method, kmax=kmax, prior_order=prior_order,
                                       source_sha256=sha256_file(ROOT / "inverse_study/numerics.py")))
        run = RESULTS / "classical" / profile / f"{method}_{signature[:16]}"
        run.mkdir(parents=True, exist_ok=True)
        if (run / "metrics.json").exists():
            rows.append(read_json(run / "metrics.json"))
            continue
        fit = solve_spectral(p, obs, kmax=kmax, method=method, prior_order=prior_order)
        u0 = evaluate_spectral(fit, Xeval, 0, p)
        uT = evaluate_spectral(fit, Xeval, p["T"], p)
        record = dict(case=p["name"], method=method, profile=profile, dataset_id=meta["dataset_id"],
                      noise_level=meta["noise_level"], noise_seed=meta["noise_seed"],
                      obs_per_dim=meta["obs_per_dim"], exposure=p["Fo"] * p["T"],
                      observation_sha256=meta["observation_sha256"], kmax=int(kmax),
                      initial_relative_l2=relative_l2(u0, exact_linear(p, Xeval)),
                      independent_terminal_relative_l2=relative_l2(uT, exact_linear(p, Xeval, p["T"])),
                      independent_observation_residual=fit["residual_norm"], target_residual=fit["target_residual"],
                      selection_status=fit["selection_status"], selected_parameter=fit["selected_parameter"],
                      wall_seconds=fit["wall_seconds"], numerical_rank=fit["numerical_rank"],
                      publication_eligible=False, run_folder=str(run.relative_to(ROOT)))
        np.savez_compressed(run / "solution.npz", X_eval=Xeval, theta_initial=u0, theta_terminal=uT,
                            modes=fit["modes"], coefficients=fit["coefficients"], singular_values=fit["singular_values"])
        pd.DataFrame(fit["path"]).to_csv(run / "parameter_path.csv", index=False)
        write_json(run / "metrics.json", record)
        write_json(run / "config.json", dict(problem=p, dataset_folder=str(Path(folder).relative_to(ROOT)),
                                             signature=signature, prior_order=prior_order,
                                             numerical_floor_source=fit["numerical_floor_source"]))
        rows.append(record)
    return pd.DataFrame(rows)


def collect_candidates():
    """Collect explicit candidates without silently multiplying statistical units.

    Multiple configurations/source versions for one dataset/method/training
    seed are blocked together. The analyst must predeclare one protocol or
    report the variants separately; the collector never selects the best error.
    """
    rows, blocked = [], []
    for path in sorted((RESULTS / "neural/production").glob("*/study_record.json")):
        run = path.parent
        record = read_json(path)
        status = read_json(run / "status.json")
        if status["status"] != "completed" or not (run / "evaluation.json").exists():
            blocked.append(dict(run_folder=str(run.relative_to(ROOT)), reason="incomplete_training_or_evaluation"))
            continue
        p, obs, meta = load_observations(ROOT / record["dataset_folder"])
        with np.load(run / "observations.npz", allow_pickle=False) as z:
            actual_hash = array_hash(z["X_obs"], z["y_obs"])
        if (actual_hash != meta["observation_sha256"]
                or record.get("observation_sha256") != meta["observation_sha256"]
                or record.get("dataset_id") != meta["dataset_id"]):
            blocked.append(dict(run_folder=str(run.relative_to(ROOT)), reason="observation_mismatch"))
            continue
        metrics = read_json(run / "metrics.json")
        saved_config = read_json(run / "config.json")
        if (metrics.get("profile") != "production" or record.get("profile") != "production"
                or status.get("profile") != "production"
                or saved_config["config"].get("profile") != "production"):
            blocked.append(dict(run_folder=str(run.relative_to(ROOT)), reason="smoke_output"))
            continue
        if not (status.get("signature") == metrics.get("signature") == saved_config.get("signature")):
            blocked.append(dict(run_folder=str(run.relative_to(ROOT)), reason="run_signature_mismatch"))
            continue
        if (metrics.get("method") != record["method"]
                or metrics.get("training_seed") != record["training_seed"]):
            blocked.append(dict(run_folder=str(run.relative_to(ROOT)), reason="method_or_seed_metadata_mismatch"))
            continue
        rows.append({**record, **metrics, **read_json(run / "evaluation.json"),
                     "training_seed": record["training_seed"], "noise_seed": meta["noise_seed"],
                     "method": record["method"], "status": "candidate_requires_gates"})
    for file in sorted((RESULTS / "classical/candidate").glob("*/metrics.json")):
        record = read_json(file)
        config = read_json(file.with_name("config.json"))
        _, _, meta = load_observations(ROOT / config["dataset_folder"])
        if (record.get("observation_sha256") != meta["observation_sha256"]
                or record.get("dataset_id") != meta["dataset_id"]):
            blocked.append(dict(run_folder=str(file.parent.relative_to(ROOT)), reason="classical_observation_mismatch"))
            continue
        rows.append({**record, "training_seed": None, "status": "candidate_requires_gates"})
    frame = pd.DataFrame(rows)
    if len(frame):
        duplicates = frame.duplicated(["dataset_id", "method", "training_seed"], keep=False)
        for _, row in frame[duplicates].iterrows():
            blocked.append(dict(run_folder=row["run_folder"], reason="duplicate_statistical_unit_requires_predeclared_variant",
                                dataset_id=row["dataset_id"], method=row["method"], training_seed=row["training_seed"]))
        frame = frame[~duplicates].reset_index(drop=True)
    return frame, pd.DataFrame(blocked)
