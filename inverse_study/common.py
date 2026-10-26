"""Shared observation contract, metadata, evaluation and run planning.

The trainer receives only the PDE, boundary histories and terminal observations.
Initial/clean synthetic truth is written separately for post-training evaluation.
Original archived runs are never silently relabelled as the corrected protocol.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(*arrays):
    digest = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        digest.update(str(a.shape).encode())
        digest.update(a.dtype.str.encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def metadata_hash(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def problem_spec(name="smooth2d", exposure=None):
    if name == "smooth2d":
        p = dict(name=name, dim=2, Fo=2e-4, T=100.0, law="constant", a=1.0,
                 modes=[[1, 1], [2, 3]], amplitudes=[1.0, 0.25])
    elif name == "nonlinear1d":
        p = dict(name=name, dim=1, Fo=0.1, T=1.0, law="exp_minus_theta_minus_one",
                 initial_left=1.0, initial_right=0.8, jump_x=0.5)
    elif name.startswith("dimension") and name[-2:] in {"1d", "2d", "3d"}:
        d = int(name[-2])
        p = dict(name=name, dim=d, Fo=4e-4, T=50.0, law="constant", a=1.0,
                 modes=[[1] * d, [2] + [1] * (d - 1)], amplitudes=[1.0, 0.2])
    else:
        raise ValueError(f"Unsupported study case: {name}")
    p["bounds"] = [[0.0, 1.0]] * p["dim"]
    if exposure is not None:
        if p["law"] != "constant" or float(exposure) <= 0:
            raise ValueError("Exposure override requires a positive linear Fo*T")
        p["Fo"] = float(exposure) / p["T"]
    return p


def tensor_grid(dim, n, interior=False, bounds=None):
    bounds = np.asarray(bounds if bounds is not None else [[0, 1]] * dim, float)
    axes = [np.linspace(lo, hi, int(n))[1:-1] if interior else np.linspace(lo, hi, int(n)) for lo, hi in bounds]
    if not axes or min(map(len, axes)) < 1:
        raise ValueError("A grid must contain at least one sample per axis")
    return axes, np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)


def exact_linear(problem, X, t=0.0):
    if problem["law"] != "constant":
        raise ValueError("This expression applies only to the specified linear modal benchmarks")
    X = np.asarray(X, float)
    bounds = np.asarray(problem["bounds"], float)
    span = bounds[:, 1] - bounds[:, 0]
    Z = (X - bounds[:, 0]) / span
    modes = np.asarray(problem["modes"], int)
    basis = np.prod(np.sin(np.pi * Z[:, None, :] * modes[None, :, :]), axis=2)
    eigenvalues = np.pi**2 * np.sum((modes / span)**2, axis=1)
    if float(problem.get("a", 1.0)) != 1.0:
        raise ValueError("Use diffusivity_multiplier so neural and numerical solvers share one convention")
    coefficient = float(problem["Fo"]) * float(problem.get("diffusivity_multiplier", 1.0))
    return basis @ (np.asarray(problem["amplitudes"]) * np.exp(-coefficient * float(t) * eigenvalues))


def initial_jump(x):
    return np.where(np.asarray(x) <= 0.5, 1.0, 0.8)


def freeze_observations(problem, obs_per_dim=33, noise_level=0.05, noise_seed=3100,
                        reference=None, destination=None):
    """Freeze iid Gaussian INTERIOR noise and a separate evaluation-only truth file.

    sigma = noise_level * RMS(clean terminal interior samples).
    noise_norm is the actual synthetic ||epsilon||_2, not noise_level itself.
    This known synthetic norm is used by the discrepancy selector. The original
    study's noisy boundary observations are deliberately not reproduced here.
    """
    destination = Path(destination or RESULTS / "observations")
    noise_level = float(noise_level)
    if noise_level < 0 or not np.isfinite(noise_level):
        raise ValueError("noise_level must be finite and nonnegative")
    noise_seed = 0 if noise_level == 0 else int(noise_seed)
    _, X = tensor_grid(problem["dim"], obs_per_dim, interior=True, bounds=problem["bounds"])
    if problem["law"] == "constant":
        clean = exact_linear(problem, X, problem["T"])
        reference_hash = "analytic-modal-truth"
    else:
        if reference is None:
            raise ValueError("Nonlinear observations require a verified forward reference")
        clean = np.interp(X[:, 0], reference["x"], reference["terminal"])
        reference_hash = array_hash(np.asarray(reference["x"]), np.asarray(reference["terminal"]))
    sigma = noise_level * float(np.sqrt(np.mean(clean**2)))
    noise = np.random.default_rng(noise_seed).normal(0, sigma, size=clean.shape)
    y = clean + noise
    meta = dict(problem=problem, obs_per_dim=int(obs_per_dim), n_interior=len(X),
                noise_level=noise_level, noise_seed=noise_seed, noise_model="iid_gaussian_interior",
                sigma=sigma, noise_norm=float(np.linalg.norm(noise)),
                selection_noise_norm_source="known realised synthetic noise norm",
                boundary_policy="exact_known_Dirichlet_data_in_separate_boundary_loss",
                reference_sha256=reference_hash, observation_sha256=array_hash(X, y),
                protocol="matched-interior-v1")
    dataset_id = f"{problem['name']}_n{obs_per_dim}_noise{noise_level:g}_s{noise_seed}_" + metadata_hash(meta)[:10]
    meta["dataset_id"] = dataset_id
    folder = destination / dataset_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "metadata.json").exists():
        if read_json(folder / "metadata.json") != jsonable(meta):
            raise RuntimeError("Frozen dataset differs from requested metadata; use a new destination")
        with np.load(folder / "observations.npz", allow_pickle=False) as old:
            if array_hash(old["X_obs"], old["y_obs"]) != meta["observation_sha256"]:
                raise RuntimeError("Frozen observations were changed on disk")
    else:
        np.savez_compressed(folder / "observations.npz", X_obs=X, y_obs=y,
                            noise_norm=meta["noise_norm"], sigma=sigma)
        np.savez_compressed(folder / "evaluation_only.npz", y_clean=clean, noise=noise)
        write_json(folder / "metadata.json", meta)
    return folder


def load_observations(folder):
    folder = Path(folder)
    meta = read_json(folder / "metadata.json")
    with np.load(folder / "observations.npz", allow_pickle=False) as z:
        obs = {k: z[k].copy() for k in z.files}
    if array_hash(obs["X_obs"], obs["y_obs"]) != meta["observation_sha256"]:
        raise RuntimeError("Observation content does not match its frozen manifest")
    obs["dataset_id"] = meta["dataset_id"]
    obs["noise_norm"] = float(obs["noise_norm"])
    return meta["problem"], obs, meta


def relative_l2(pred, truth):
    pred, truth = np.asarray(pred).reshape(-1), np.asarray(truth).reshape(-1)
    den = np.linalg.norm(truth)
    return float(np.linalg.norm(pred - truth) / den) if den > 0 else None


def jump_metrics(x, u):
    """Evaluate the ORIGINAL 0.2 jump; no extrapolated percentage-of-baseline claim."""
    x, u = np.asarray(x).reshape(-1), np.asarray(u).reshape(-1)
    idx = np.argsort(x)
    x, u = x[idx], u[idx]
    left = float(np.mean(u[(x >= 0.25) & (x <= 0.40)]))
    right = float(np.mean(u[(x >= 0.60) & (x <= 0.75)]))
    height = left - right
    def crossing(level):
        hits = np.where((u[:-1] - level) * (u[1:] - level) <= 0)[0]
        hits = [i for i in hits if 0.25 <= x[i] <= 0.75 and u[i + 1] != u[i]]
        if not hits:
            return None
        positions = [x[i] + (level - u[i]) * (x[i + 1] - x[i]) / (u[i + 1] - u[i]) for i in hits]
        return float(min(positions, key=lambda z: abs(z - 0.5)))
    mid = crossing(0.9)
    p90, p10 = crossing(0.98), crossing(0.82)
    band = (x >= 0.25) & (x <= 0.75)
    truth = initial_jump(x)
    return dict(initial_relative_l2=relative_l2(u, truth),
                rmse_over_true_jump=float(np.sqrt(np.mean((u - truth)**2)) / 0.2),
                plateau_left=left, plateau_right=right, jump_height=height,
                jump_height_relative_error=abs(height - 0.2) / 0.2,
                jump_location=mid, jump_location_absolute_error=abs(mid - 0.5) if mid is not None else None,
                transition_width_10_90=abs(p10 - p90) if p10 is not None and p90 is not None else None,
                overshoot_over_jump=float(max(0, np.max(u[band]) - 1.0) / 0.2),
                undershoot_over_jump=float(max(0, 0.8 - np.min(u[band])) / 0.2))


def archived_inventory():
    rows = []
    for d in sorted((DATA / "legacy_runs").glob("*")):
        if not (d / "metrics.json").exists():
            continue
        metrics = read_json(d / "metrics.json")
        cfg = read_json(d / "config.json")
        with np.load(d / "terminal_observations.npz", allow_pickle=False) as a:
            X = a["X_f"][:, :int(cfg["spatial_dim"])]
            clean, noisy = a["theta_clean"].ravel(), a["theta_noisy"].ravel()
        interior = np.all((X > 1e-12) & (X < 1 - 1e-12), axis=1)
        rows.append(dict(run_id=d.name, spatial_dim=int(cfg["spatial_dim"]),
                         original_initial_relative_l2=metrics.get("initial_relative_l2"),
                         n_all=len(X), n_interior=int(interior.sum()),
                         boundary_noise_norm=float(np.linalg.norm((noisy-clean)[~interior])),
                         observation_sha256=array_hash(X, noisy),
                         protocol="legacy_source_reported", independent_replication=False))
    return pd.DataFrame(rows)


def paired_crossed_bootstrap(frame, method_a, method_b, value="initial_relative_l2", repeats=2000, seed=8301,
                             expected_noise_seeds=None, expected_training_seeds=None):
    """Paired CI with independent resampling of noise and training-seed axes.

    Call only for ONE fixed case/noise/density/exposure. Incomplete factorials
    are returned as blocked, not silently pooled as independent observations.
    Classical deterministic fits should be reported across noise seeds alone.
    Pass the two expected seed lists from the frozen plan to detect a seed
    missing from BOTH methods; without them, completeness is checked over the
    union of observed seeds and cannot establish planned-sample completeness.
    """
    key = ["noise_seed", "training_seed"]
    selected = frame[frame.method.isin([method_a, method_b])]
    for condition in ("case", "noise_level", "obs_per_dim", "exposure", "beta"):
        if condition in selected and selected[condition].nunique(dropna=False) > 1:
            raise ValueError(f"Bootstrap requires one fixed {condition}; group the data first")
    a = selected[selected.method == method_a][key + [value]].rename(columns={value: "a"})
    b = selected[selected.method == method_b][key + [value]].rename(columns={value: "b"})
    if a.duplicated(key).any() or b.duplicated(key).any():
        raise ValueError("Duplicate statistical units: group by case/noise/density/exposure first")
    if a[key].isna().any().any() or b[key].isna().any().any():
        return dict(status="blocked_missing_seed_identity", n_pairs=0)
    if not np.isfinite(a["a"].to_numpy(dtype=float)).all() or not np.isfinite(b["b"].to_numpy(dtype=float)).all():
        return dict(status="blocked_nonfinite_metric", n_pairs=0)
    noise_seeds = set(expected_noise_seeds) if expected_noise_seeds is not None else set(a.noise_seed) | set(b.noise_seed)
    training_seeds = set(expected_training_seeds) if expected_training_seeds is not None else set(a.training_seed) | set(b.training_seed)
    expected = {(n, t) for n in noise_seeds for t in training_seeds}
    keys_a, keys_b = set(a[key].itertuples(index=False, name=None)), set(b[key].itertuples(index=False, name=None))
    if not expected or keys_a != expected or keys_b != expected:
        return dict(status="blocked_incomplete_pairs", n_pairs=len(keys_a & keys_b),
                    expected_pairs=len(expected), available_a=len(keys_a), available_b=len(keys_b))
    paired = a.merge(b, on=key, validate="one_to_one")
    pivot = paired.assign(delta=paired.a - paired.b).pivot(index="noise_seed", columns="training_seed", values="delta")
    if pivot.empty or pivot.isna().any().any():
        return dict(status="blocked_incomplete_pairs", n_pairs=len(paired))
    values = pivot.to_numpy()
    if min(values.shape) < 2:
        return dict(status="insufficient_independent_noise_or_training_seeds", n_pairs=len(paired),
                    mean_delta=float(values.mean()), ci_low=None, ci_high=None)
    rng = np.random.default_rng(seed)
    if int(repeats) < 1:
        raise ValueError("repeats must be a positive integer")
    boot = np.empty(int(repeats))
    for i in range(int(repeats)):
        rows = rng.integers(0, values.shape[0], values.shape[0])
        cols = rng.integers(0, values.shape[1], values.shape[1])
        boot[i] = values[np.ix_(rows, cols)].mean()
    return dict(status="exploratory_crossed_bootstrap", mean_delta=float(values.mean()),
                ci_low=float(np.quantile(boot, 0.025)), ci_high=float(np.quantile(boot, 0.975)),
                n_pairs=len(paired), n_noise=values.shape[0], n_training=values.shape[1],
                convention=f"{method_a} minus {method_b}; negative favors {method_a}")


def linear_run_evaluation(run_dir, problem, observations, closure=True):
    """Independent continuation of the reconstructed initial trace, not network(T)."""
    from .numerics import forward_linear_grid
    run_dir = Path(run_dir)
    with np.load(run_dir / "predictions.npz", allow_pickle=False) as z:
        X, u0, uT = z["X_eval"], z["theta_initial"].ravel(), z["theta_terminal"].ravel()
        obs_T = z["obs_terminal"].ravel()
    ans = dict(initial_relative_l2=relative_l2(u0, exact_linear(problem, X)),
               network_terminal_relative_l2=relative_l2(uT, exact_linear(problem, X, problem["T"])),
               network_observation_residual=float(np.linalg.norm(obs_T - observations["y_obs"])))
    if closure:
        axes = [np.unique(X[:, i]) for i in range(problem["dim"])]
        expected = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, problem["dim"])
        if X.shape != expected.shape or not np.allclose(X, expected):
            raise ValueError("Independent continuation requires the saved regular ij-order grid")
        replay = forward_linear_grid(u0.reshape(tuple(map(len, axes))), axes,
                                     problem["Fo"] * problem.get("diffusivity_multiplier", 1.0), problem["T"])
        replay_values = np.asarray(replay["terminal"])
        sensor_replay = RegularGridInterpolator(tuple(axes), replay_values)(observations["X_obs"])
        ans.update(independent_terminal_relative_l2=relative_l2(replay_values.ravel(), exact_linear(problem, X, problem["T"])),
                   independent_observation_residual=float(np.linalg.norm(sensor_replay-observations["y_obs"])),
                   network_vs_independent_terminal_relative_l2=relative_l2(uT, replay_values.ravel()),
                   initial_boundary_linf=replay["initial_boundary_linf"],
                   forward_replay_spatial_points=list(map(len, axes)),
                   forward_replay_requires_refinement=True)
    write_json(run_dir / "evaluation.json", ans)
    return ans


def environment():
    import importlib.metadata as md
    names = ["numpy", "scipy", "pandas", "matplotlib", "tensorflow-cpu", "tensorflow", "nbformat", "nbclient"]
    versions = {}
    for name in names:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions)
