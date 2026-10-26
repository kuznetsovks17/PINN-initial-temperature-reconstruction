"""Isolated two-dimensional, strongly nonlinear extension of the inverse PINN engine.

Adapted from the user-supplied ``epinn_revised_core.py`` (architecture,
conservative flux differentiation, zero final layer, and bounded scalar
weight ascent). The experiment driver is new: shared interior observations,
separate randomness, matched Fourier maps, truth-free diagnostics, provenance,
and restartable fixed-budget training. This separate engine intentionally does not change the fingerprint of completed older trainings. Composite materials are not supported.

TensorFlow is imported lazily. Inspection, data validation, and configuration
planning therefore work in a NumPy/SciPy-only environment. This file does not
contain any completed study result.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

METHODS = ("plain", "fourier_fixed", "fourier_adaptive", "raw_matched")
_ENGINE_VERSION = "strong-nonlinear-2d-1"


def _json_value(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    return value


def _write_json(path: Path, value: dict):
    """Atomic metadata write; incomplete jobs cannot appear completed."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _hash_array(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.shape).encode() + value.dtype.str.encode() + value.tobytes()).hexdigest()


def _metadata_hash(value):
    return hashlib.sha256(json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tensorflow_available():
    """A lightweight dependency check; does not import TensorFlow."""
    import importlib.util
    return importlib.util.find_spec("tensorflow") is not None


def _tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "PINN training requires TensorFlow. Install requirements-training.txt "
            "in the notebook environment, restart its kernel, and rerun. "
            "The data/reference/baseline notebooks do not require TensorFlow."
        ) from exc
    return tf


def training_config(profile="production", **overrides):
    """Explicit profiles. Smoke output is never publication evidence.

    Production preserves the archived synthetic 4 x 30 tanh architecture,
    16 frequencies per coordinate, Adam rates, and 30,000-step budget.
    Data/observation counts are not changed by either profile.
    """
    if profile not in {"production", "smoke"}:
        raise ValueError("profile must be 'production' or 'smoke'")
    cfg = dict(
        profile=profile, method="fourier_adaptive", training_seed=2026,
        feature_seed=1729, collocation_seed=2718, hidden_width=30,
        hidden_depth=4, features_per_coordinate=16, fourier_scale=1.0,
        dtype="float32", epochs=30000, N_r=60000, N_b=8000,
        batch_r=8192, batch_b=2048, learning_rate=2e-4,
        weight_learning_rate=5e-4, beta_1=0.99, eta_min=1e-2,
        eta_max=1e2, residual_scale=1.0, output_shift="terminal_mean",
        history_every=100, validate_every=500, checkpoint_every=1000,
        validation_r=2048, validation_b=512, print_every=1000,
        eval_per_dim=None, prediction_batch=8192, deterministic=True,
        resume=True,
    )
    if profile == "smoke":
        cfg.update(epochs=10, N_r=256, N_b=128, batch_r=128, batch_b=64,
                   hidden_width=16, hidden_depth=2, features_per_coordinate=4,
                   history_every=1, validate_every=5, checkpoint_every=5,
                   validation_r=64, validation_b=64, print_every=5,
                   eval_per_dim=17)
    cfg.update(overrides)
    return cfg


def _prepare(problem, observations, config):
    problem = dict(problem)
    cfg = training_config(**config)
    dim = int(problem["dim"])
    if dim not in (1, 2, 3):
        raise ValueError("Only one-, two-, and three-dimensional boxes are supported.")
    bounds = np.asarray(problem.get("bounds", [[0.0, 1.0]] * dim), dtype=float)
    if bounds.shape != (dim, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("bounds must be dim x 2 with increasing finite endpoints")
    problem["bounds"] = bounds.tolist()
    if not np.isfinite(problem["T"]) or float(problem["T"]) <= 0:
        raise ValueError("T must be positive and finite")
    if not np.isfinite(problem["Fo"]) or float(problem["Fo"]) <= 0:
        raise ValueError("Fo must be positive and finite")
    if problem.get("law", "constant") not in {"constant", "exp_minus_theta_minus_one", "brass", "sigmoid20_2d"}:
        raise ValueError("Unsupported diffusivity law; composite materials are excluded")
    problem.setdefault("law", "constant")
    if problem.get("boundary_type", "dirichlet") not in {"dirichlet", "dirichlet0"}:
        raise ValueError("This engine supports Dirichlet boundary data only")
    if any(k in problem for k in ("source", "f")):
        for key in ("source", "f"):
            if key in problem and (not np.isscalar(problem[key]) or float(problem[key]) != 0):
                raise ValueError("Nonzero heat sources require an explicit extension; this engine uses f=0")
    if "beta" in problem and problem["law"] not in {"exp_minus_theta_minus_one", "sigmoid20_2d"}:
        raise ValueError("beta is only meaningful for exp_minus_theta_minus_one")
    if any(k in problem for k in ("B0", "B1")) and problem["law"] != "brass":
        raise ValueError("B0/B1 are only meaningful for the brass diffusivity law")
    if "a" in problem and float(problem["a"]) != 1.0:
        raise ValueError("Use diffusivity_multiplier instead of a so PINN and numerical solvers share one convention")
    multiplier = float(problem.get("diffusivity_multiplier", 1.0))
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("diffusivity_multiplier must be positive and finite")
    if not np.isfinite(float(problem.get("beta", 1.0))):
        raise ValueError("Nonlinearity beta must be finite")
    if problem["law"] in {"exp_minus_theta_minus_one", "brass"} and dim != 1:
        raise ValueError("This completion suite implements nonlinear and brass cases in 1D only")
    if problem["law"] == "sigmoid20_2d":
        if dim != 2:
            raise ValueError("sigmoid20_2d expects two spatial dimensions")
        if float(problem.get("a_lo",0)) <= 0 or float(problem.get("a_hi",0)) <= float(problem.get("a_lo",0)):
            raise ValueError("sigmoid20_2d requires 0<a_lo<a_hi")
        if float(problem.get("beta",0)) <= 0 or not np.isfinite(float(problem["beta"])):
            raise ValueError("sigmoid20_2d requires a positive finite steepness")
    if cfg["method"] not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if cfg["dtype"] not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    for key in ("epochs", "N_r", "N_b", "batch_r", "batch_b", "history_every", "validate_every", "checkpoint_every", "validation_r", "validation_b", "hidden_width", "hidden_depth", "features_per_coordinate", "prediction_batch"):
        if int(cfg[key]) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < float(cfg["eta_min"]) <= 1 <= float(cfg["eta_max"]):
        raise ValueError("Adaptive coefficient bounds must contain 1 and be strictly positive")
    if not np.isfinite(cfg["residual_scale"]) or float(cfg["residual_scale"]) <= 0:
        raise ValueError("residual_scale must be positive and finite")
    X = np.asarray(observations["X_obs"], dtype=np.float64)
    y = np.asarray(observations["y_obs"], dtype=np.float64).reshape(-1)
    if X.ndim != 2 or X.shape != (len(y), dim) or len(y) == 0:
        raise ValueError("X_obs must have shape (number of observations, dim)")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Observed coordinates and values must be finite")
    if np.any(X <= bounds[:, 0]) or np.any(X >= bounds[:, 1]):
        raise ValueError("Primary comparison requires STRICTLY INTERIOR terminal observations; do not add noisy boundary nodes")
    if len(np.unique(X, axis=0)) != len(X):
        raise ValueError("Duplicate observation coordinates require an explicit covariance model")
    if "boundary_values" in problem:
        bv = np.asarray(problem["boundary_values"], dtype=float)
        if dim != 1 or bv.shape != (2,) or not np.isfinite(bv).all():
            raise ValueError("boundary_values must be a finite [left,right] pair for a 1D problem")
    if problem["law"] == "brass" or "boundary_times" in problem:
        if dim != 1:
            raise ValueError("Tabulated boundary histories are supported in one spatial dimension")
        for key in ("boundary_times", "boundary_left", "boundary_right"):
            if key not in problem:
                raise ValueError(f"Tabulated boundary histories require {key}, in nondimensional coordinates/temperatures")
        bt = np.asarray(problem["boundary_times"], dtype=float)
        if bt.ndim != 1 or len(bt) < 2 or np.any(np.diff(bt) <= 0) or bt[0] > 0 or bt[-1] < problem["T"]:
            raise ValueError("boundary_times must increase and cover [0,T]")
        for key in ("boundary_left", "boundary_right"):
            b = np.asarray(problem[key], dtype=float)
            if b.shape != bt.shape or not np.isfinite(b).all():
                raise ValueError(f"{key} must match finite boundary_times")
    return problem, cfg, X, y


def validate_inputs(problem, observations, config=None):
    """Validate the observation contract without importing TensorFlow."""
    p, cfg, X, y = _prepare(problem, observations, config or {})
    return {"n_observations": len(y), "dim": int(p["dim"]),
            "interior_only": True, "observation_sha256": _hash_array(np.column_stack([X, y])),
            "profile": cfg["profile"], "method": cfg["method"]}


def fourier_features_numpy(Z, bounds, final_time, seed=1729, features=16, scale=1.0, raw=False):
    """Inspect the matched feature map without TensorFlow.

    Fourier normalization is [0,1], whereas the plain network receives [-1,1].
    The raw path uses reciprocal spans and subtracts the lower-bound phase.
    Equal underlying draws therefore give the same map on nonzero-based boxes.
    """
    Z = np.asarray(Z, dtype=float)
    lower = np.r_[np.asarray(bounds, dtype=float)[:, 0], 0.0]
    span = np.r_[np.diff(np.asarray(bounds, dtype=float), axis=1).ravel(), final_time]
    B = np.random.default_rng(seed).normal(size=(len(span), int(features))) * scale
    if raw:
        phase = 2 * np.pi * (Z[:, :, None] - lower[None, :, None]) * (B / span[:, None])[None, :, :]
    else:
        phase = 2 * np.pi * ((Z - lower) / span)[:, :, None] * B[None, :, :]
    return np.concatenate([np.sin(phase), np.cos(phase)], axis=2).reshape(len(Z), -1)


def _sobol(n, dim, seed):
    # random_base2 avoids silently discarding a non-power-of-two warning.
    n = int(n)
    points = qmc.Sobol(dim, scramble=True, seed=int(seed)).random_base2(int(math.ceil(math.log2(max(1, n)))))[:n]
    return np.clip(points, np.finfo(float).eps, 1.0 - np.finfo(float).eps)


def _boundary_value(problem, Z):
    dim, T = int(problem["dim"]), float(problem["T"])
    bounds = np.asarray(problem["bounds"], dtype=float)
    left = np.isclose(Z[:, 0], bounds[0, 0], rtol=0, atol=1e-12)
    times = Z[:, dim]
    if "boundary_times" in problem:
        left_values = np.interp(times, problem["boundary_times"], problem["boundary_left"])
        right_values = np.interp(times, problem["boundary_times"], problem["boundary_right"])
        return np.where(left, left_values, right_values)[:, None]
    if problem["law"] == "exp_minus_theta_minus_one":
        return np.where(left, 1.0 - 0.5 * times / T, 0.8 - 0.3 * times / T)[:, None]
    if dim == 1 and "boundary_values" in problem:
        values = problem["boundary_values"]
        return np.where(left, values[0], values[1])[:, None]
    return np.full((len(Z), 1), float(problem.get("dirichlet_value", 0.0)))


def _sample_points(problem, nr, nb, seed):
    dim, T = int(problem["dim"]), float(problem["T"])
    bounds = np.asarray(problem["bounds"], dtype=float)
    lower = np.r_[bounds[:, 0], 0.0]
    span = np.r_[bounds[:, 1] - bounds[:, 0], T]
    Xr = lower + _sobol(nr, dim + 1, seed) * span
    Xb = lower + _sobol(nb, dim + 1, seed + 1) * span
    for i in range(len(Xb)):
        face = i % (2 * dim)
        Xb[i, face // 2] = bounds[face // 2, face % 2]
    return Xr, Xb, _boundary_value(problem, Xb)


def _make_model(tf, problem, cfg, output_shift):
    dtype = tf.as_dtype(cfg["dtype"])
    dim = int(problem["dim"])
    bounds = np.asarray(problem["bounds"], dtype=float)
    lower_np = np.r_[bounds[:, 0], 0.0]
    span_np = np.r_[bounds[:, 1] - bounds[:, 0], float(problem["T"])]
    B = np.random.default_rng(int(cfg["feature_seed"])).normal(size=(dim + 1, int(cfg["features_per_coordinate"]))) * float(cfg["fourier_scale"])

    class FeaturePINN(tf.keras.Model):
        def __init__(self):
            super().__init__(dtype=dtype, name="inverse_temperature_pinn")
            self.lower = tf.constant(lower_np, dtype=dtype)
            self.span = tf.constant(span_np, dtype=dtype)
            self.B = tf.constant(B, dtype=dtype)
            self.B_raw = tf.constant(B / span_np[:, None], dtype=dtype)
            self.shift = tf.constant(float(output_shift), dtype=dtype)
            self.hidden = [tf.keras.layers.Dense(
                int(cfg["hidden_width"]), activation="tanh", dtype=dtype,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(cfg["training_seed"]) + 101 * (i + 1)),
                bias_initializer="zeros", name=f"hidden_{i}",
            ) for i in range(int(cfg["hidden_depth"]))]
            self.out = tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros", dtype=dtype, name="theta")

        def call(self, Z, training=False):
            Z = tf.cast(Z, dtype)
            unit = (Z - self.lower) / self.span
            if cfg["method"] == "plain":
                h = 2 * unit - 1
            else:
                # Canonicalize b*(Z-lower)/span to one identical operation
                # sequence in both Fourier branches. This exact identity
                # control deliberately has no floating-point/order confound.
                phase = tf.constant(2 * np.pi, dtype=dtype) * unit[:, :, None] * self.B[None, :, :]
                h = tf.reshape(tf.concat([tf.sin(phase), tf.cos(phase)], axis=2), [-1, 2 * (dim + 1) * int(cfg["features_per_coordinate"])])
            for layer in self.hidden:
                h = layer(h, training=training)
            return self.out(h, training=training) + self.shift

    model = FeaturePINN()
    model(tf.zeros((1, dim + 1), dtype=dtype))
    return model, B


def _residual_function(tf, problem, model):
    dim = int(problem["dim"])
    dtype = tf.as_dtype(model.compute_dtype)
    Fo = tf.constant(float(problem["Fo"]) * float(problem.get("diffusivity_multiplier", 1.0)), dtype=dtype)
    law = problem["law"]

    def residual(Z, training=False):
        # Differentiate physical nondimensional coordinates, not pre-normalized
        # inputs. Internal input transformations remain inside both tapes, so
        # all length and time chain factors are included automatically.
        with tf.GradientTape(persistent=True) as outer:
            outer.watch(Z)
            with tf.GradientTape() as inner:
                inner.watch(Z)
                theta = model(Z, training=training)
            grad = inner.gradient(theta, Z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            if law == "constant":
                a = tf.ones_like(theta) * tf.constant(float(problem.get("a", 1.0)), dtype=dtype)
            elif law == "exp_minus_theta_minus_one":
                beta = tf.constant(float(problem.get("beta", 1.0)), dtype=dtype)
                a = tf.exp(-beta * theta - tf.constant(1.0, dtype=dtype))
                tf.debugging.assert_all_finite(a, "Nonlinear diffusivity overflowed")
            elif law == "sigmoid20_2d":
                low = tf.constant(float(problem["a_lo"]), dtype=dtype)
                width = tf.constant(float(problem["a_hi"])-float(problem["a_lo"]), dtype=dtype)
                beta = tf.constant(float(problem["beta"]), dtype=dtype)
                midpoint = tf.constant(float(problem["theta_mid"]), dtype=dtype)
                a = low + width * tf.math.sigmoid(beta * (theta - midpoint))
            else:
                a = tf.constant(float(problem.get("B0", 0.287265051903114)), dtype=dtype) + tf.constant(float(problem.get("B1", 0.011656401384083)), dtype=dtype) * theta
                tf.debugging.assert_positive(a, message="Brass diffusivity left its positive physical range")
            flux = Fo * a * grad[:, :dim]
            components = [flux[:, j:j + 1] for j in range(dim)]
        divergence = []
        for j, component in enumerate(components):
            derivative = outer.gradient(component, Z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            divergence.append(derivative[:, j:j + 1])
        del outer
        return grad[:, dim:dim + 1] - tf.add_n(divergence)
    return residual


def _prediction(model, Z, batch_size):
    return np.concatenate([np.asarray(model(Z[i:i + batch_size], training=False)).reshape(-1) for i in range(0, len(Z), batch_size)])


def _save_weights(path, model):
    np.savez_compressed(path, **{f"w_{i:03d}": w for i, w in enumerate(model.get_weights())})


def _load_weights(path, model):
    with np.load(path, allow_pickle=False) as data:
        model.set_weights([data[key] for key in sorted(data.files)])


def load_model(run_dir, checkpoint="final"):
    """Rebuild a saved network for post-training evaluation; no truth required."""
    if checkpoint not in {"final", "best_truth_free"}:
        raise ValueError("checkpoint must be final or best_truth_free")
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "config.json").read_text())
    tf = _tensorflow()
    model, _ = _make_model(tf, meta["problem"], meta["config"], meta["output_shift"])
    _load_weights(run_dir / f"weights_{checkpoint}.npz", model)
    return model


def predict_saved(run_dir, X_space, times=0.0, checkpoint="final"):
    """Predict at paired spatial positions and scalar/vector times."""
    model = load_model(run_dir, checkpoint)
    X = np.asarray(X_space, dtype=float)
    tt = np.broadcast_to(np.asarray(times, dtype=float), (len(X),))
    return _prediction(model, np.column_stack([X, tt]), 8192)


def train_inverse(problem: dict, observations: dict, config: dict, output_dir) -> dict:
    """Train a fixed-budget inverse reconstruction on frozen interior data.

    ``problem``: name, dim, Fo, T, law, bounds; see module documentation.
    ``observations``: X_obs (n,d), y_obs (n,), noise_norm optional. Other
    truth/reference arrays are ignored and NEVER used by loss/checkpointing.
    ``config``: training_config overrides, X_eval optional, eval_times optional.
    ``output_dir``: parent directory. A SHA256-keyed child is created; matching
    completed jobs are reused. Incomplete jobs resume their latest checkpoint.

    Primary predictions are the FINAL fixed-budget network. A second
    best_truth_free checkpoint minimizes an unweighted, fixed-point score:
    (residual_scale*PDE)^2 + boundary MSE + observed-terminal MSE. It is a
    diagnostic, not a ground-truth selection or independent validation set.
    """
    p, cfg, Xobs, yobs = _prepare(problem, observations, config)
    Xeval = np.asarray(cfg.pop("X_eval"), dtype=float) if "X_eval" in cfg else None
    dim, T = int(p["dim"]), float(p["T"])
    bounds = np.asarray(p["bounds"], dtype=float)
    if Xeval is None:
        n = cfg["eval_per_dim"] or {1: 1001, 2: 101, 3: 33}[dim]
        axes = [np.linspace(a, b, int(n)) for a, b in bounds]
        Xeval = np.column_stack([v.ravel() for v in np.meshgrid(*axes, indexing="ij")])
    if Xeval.ndim != 2 or Xeval.shape[1] != dim or not len(Xeval) or not np.isfinite(Xeval).all():
        raise ValueError("X_eval must have finite shape (n,dim)")
    if np.any(Xeval < bounds[:, 0]) or np.any(Xeval > bounds[:, 1]):
        raise ValueError("X_eval must lie inside the physical box")
    eval_times = np.asarray(cfg.get("eval_times", [0.0, T]), dtype=float)
    if eval_times.ndim != 1 or not len(eval_times) or not np.isfinite(eval_times).all() or np.any(eval_times < 0) or np.any(eval_times > T):
        raise ValueError("eval_times must lie in [0,T]")
    identity_cfg = {k: v for k, v in cfg.items() if k not in {"resume", "print_every", "run_id"}}
    identity = dict(engine=_ENGINE_VERSION, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    problem=p, config=identity_cfg, X_obs_sha256=_hash_array(Xobs),
                    y_obs_sha256=_hash_array(yobs), X_eval_sha256=_hash_array(Xeval))
    signature = _metadata_hash(identity)
    name = str(cfg.get("run_id", f"{p.get('name', 'inverse')}__{cfg['method']}__seed{cfg['training_seed']}"))
    name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    run_dir = Path(output_dir).expanduser().resolve() / f"{name}__{signature[:12]}"
    status_file = run_dir / "status.json"
    if status_file.exists():
        status = json.loads(status_file.read_text())
        if status.get("status") == "completed" and status.get("signature") == signature:
            needed = ["metrics.json", "predictions.npz", "weights_final.npz", "config.json"]
            if all((run_dir / item).is_file() for item in needed):
                return dict(run_dir=str(run_dir), status="cached", metrics=json.loads((run_dir / "metrics.json").read_text()),
                            predictions_file=str(run_dir / "predictions.npz"), signature=signature)
    call_started = time.perf_counter()
    tf = _tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(int(cfg["training_seed"]))
    if cfg.get("deterministic", True):
        tf.config.experimental.enable_op_determinism()
    dtype = tf.as_dtype(cfg["dtype"])
    adaptive = cfg["method"] in {"fourier_adaptive", "raw_matched"}
    shift = float(np.mean(yobs)) if cfg["output_shift"] == "terminal_mean" else float(cfg["output_shift"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", dict(problem=p, config=cfg, output_shift=shift, identity=identity, signature=signature))
    _write_json(status_file, dict(status="running", signature=signature, profile=cfg["profile"]))
    np.savez_compressed(run_dir / "observations.npz", X_obs=Xobs, y_obs=yobs)
    np.savez_compressed(run_dir / "evaluation_grid.npz", X_eval=Xeval, eval_times=eval_times)
    model, B = _make_model(tf, p, cfg, shift)
    np.save(run_dir / "fourier_draws.npy", B)
    residual = _residual_function(tf, p, model)
    Xr, Xb, yb = _sample_points(p, cfg["N_r"], cfg["N_b"], int(cfg["collocation_seed"]))
    Xvr, Xvb, yvb = _sample_points(p, cfg["validation_r"], cfg["validation_b"], int(cfg["collocation_seed"]) + 1000003)
    np.savez_compressed(run_dir / "training_points.npz", X_r=Xr, X_b=Xb, y_b=yb, X_validation_r=Xvr, X_validation_b=Xvb, y_validation_b=yvb)
    Xr, Xb, yb = [tf.constant(v, dtype=dtype) for v in (Xr, Xb, yb)]
    Xf, yf = tf.constant(np.column_stack([Xobs, np.full(len(Xobs), T)]), dtype=dtype), tf.constant(yobs[:, None], dtype=dtype)
    Xvr, Xvb, yvb = [tf.constant(v, dtype=dtype) for v in (Xvr, Xvb, yvb)]
    eta = tf.Variable([1.0, 1.0], dtype=dtype, trainable=adaptive, name="eta_pde_boundary")
    opt = tf.keras.optimizers.Adam(learning_rate=float(cfg["learning_rate"]), beta_1=float(cfg["beta_1"]))
    opt.build(model.trainable_variables)
    eta_opt = tf.keras.optimizers.Adam(learning_rate=float(cfg["weight_learning_rate"]), beta_1=float(cfg["beta_1"])) if adaptive else None
    if eta_opt is not None:
        eta_opt.build([eta])
    step = tf.Variable(0, dtype=tf.int64, trainable=False)
    best_score = tf.Variable(float("inf"), dtype=dtype, trainable=False)
    checkpoint_items = dict(model=model, optimizer=opt, eta=eta, step=step, best_score=best_score)
    if eta_opt is not None:
        checkpoint_items["eta_optimizer"] = eta_opt
    checkpoint = tf.train.Checkpoint(**checkpoint_items)
    manager = tf.train.CheckpointManager(checkpoint, str(run_dir / "checkpoints"), max_to_keep=2)
    if cfg.get("resume", True) and manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()
    start_step = int(step.numpy())
    history = []
    history_path = run_dir / "history.json"
    if start_step and history_path.is_file():
        history = [r for r in json.loads(history_path.read_text())["records"] if r["step"] <= start_step]
    br, bb = min(int(cfg["batch_r"]), len(Xr)), min(int(cfg["batch_b"]), len(Xb))
    scale = tf.constant(float(cfg["residual_scale"]), dtype=dtype)

    @tf.function(reduce_retracing=True)
    def train_step(index):
        # A stateless epoch-indexed sampler makes a restored run continue the
        # same mini-batch sequence. Shared training seeds pair method budgets.
        seed = tf.stack([tf.cast(int(cfg["training_seed"]) % (2**31 - 1), tf.int32), tf.cast(index, tf.int32)])
        ir = tf.random.stateless_uniform([br], seed=seed, minval=0, maxval=tf.shape(Xr)[0], dtype=tf.int32)
        ib = tf.random.stateless_uniform([bb], seed=seed + tf.constant([1247, 811], tf.int32), minval=0, maxval=tf.shape(Xb)[0], dtype=tf.int32)
        with tf.GradientTape(persistent=True) as tape:
            lp = tf.reduce_mean(tf.square(scale * residual(tf.gather(Xr, ir), training=True)))
            lb = tf.reduce_mean(tf.square(model(tf.gather(Xb, ib), training=True) - tf.gather(yb, ib)))
            lf = tf.reduce_mean(tf.square(model(Xf, training=True) - yf))
            objective = eta[0] ** 2 * lp + eta[1] ** 2 * lb + lf
            tf.debugging.assert_all_finite(objective, "Nonfinite training objective")
        grads = tape.gradient(objective, model.trainable_variables)
        opt.apply_gradients([(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None])
        if adaptive:
            eg = tape.gradient(objective, eta)
            eta_opt.apply_gradients([(-eg, eta)])
            eta.assign(tf.clip_by_value(eta, float(cfg["eta_min"]), float(cfg["eta_max"])))
        del tape
        step.assign_add(1)
        return objective, lp, lb, lf

    @tf.function(reduce_retracing=True)
    def validation_losses():
        lp = tf.reduce_mean(tf.square(scale * residual(Xvr, training=False)))
        lb = tf.reduce_mean(tf.square(model(Xvb, training=False) - yvb))
        lf = tf.reduce_mean(tf.square(model(Xf, training=False) - yf))
        return lp + lb + lf, lp, lb, lf

    started = time.perf_counter()
    previous_elapsed = 0.0
    timing_path = run_dir / "timing.json"
    if start_step and timing_path.is_file():
        previous_elapsed = float(json.loads(timing_path.read_text()).get("training_wall_time_s", 0.0))
    try:
        for i in range(start_step, int(cfg["epochs"])):
            values = train_step(tf.constant(i, tf.int64))
            k = i + 1
            if k % int(cfg["history_every"]) == 0 or k == 1 or k == cfg["epochs"]:
                row = dict(step=k, objective=float(values[0].numpy()), pde_scaled_mse=float(values[1].numpy()),
                           boundary_mse=float(values[2].numpy()), terminal_mse=float(values[3].numpy()),
                           eta_pde=float(eta[0].numpy()), eta_boundary=float(eta[1].numpy()))
                history.append(row)
            if k % int(cfg["validate_every"]) == 0 or k == 1 or k == cfg["epochs"]:
                v = validation_losses()
                score = float(v[0].numpy())
                if not np.isfinite(score):
                    raise FloatingPointError("Nonfinite truth-free validation score")
                if score < float(best_score.numpy()):
                    best_score.assign(score)
                    _save_weights(run_dir / "weights_best_truth_free.npz", model)
                    _write_json(run_dir / "best_truth_free.json", dict(step=k, score=score, rule="fixed independent collocation + exact boundary + observed terminal MSE; no initial truth"))
            if cfg["print_every"] and (k % int(cfg["print_every"]) == 0 or k == 1):
                print(f"{cfg['method']}: step {k}/{cfg['epochs']}, objective={float(values[0].numpy()):.4e}, terminal={float(values[3].numpy()):.4e}")
            if k % int(cfg["checkpoint_every"]) == 0 or k == cfg["epochs"]:
                manager.save(checkpoint_number=k)
                _write_json(history_path, dict(records=history))
                _write_json(timing_path, dict(training_wall_time_s=previous_elapsed + time.perf_counter() - started))
        elapsed = previous_elapsed + time.perf_counter() - started
        _save_weights(run_dir / "weights_final.npz", model)
        n = len(Xeval)
        batch = int(cfg["prediction_batch"])
        theta0 = _prediction(model, np.column_stack([Xeval, np.zeros(n)]), batch)
        thetaT = _prediction(model, np.column_stack([Xeval, np.full(n, T)]), batch)
        profiles = np.stack([_prediction(model, np.column_stack([Xeval, np.full(n, t)]), batch) for t in eval_times])
        obs_prediction = _prediction(model, np.column_stack([Xobs, np.full(len(Xobs), T)]), batch)
        np.savez_compressed(run_dir / "predictions.npz", X_eval=Xeval, theta_initial=theta0, theta_terminal=thetaT,
                            X_obs=Xobs, obs_terminal=obs_prediction, eval_times=eval_times, theta_profiles=profiles)
        final_v = [float(v.numpy()) for v in validation_losses()]
        discrepancy = float(np.linalg.norm(obs_prediction - yobs))
        noise_norm = observations.get("noise_norm", observations.get("noise_l2"))
        metrics = dict(status="smoke" if cfg["profile"] == "smoke" else "candidate", profile=cfg["profile"],
                       publication_eligible=False, verification_note="Downstream reference/closure and study gates must pass before publication",
                       method=cfg["method"], training_seed=int(cfg["training_seed"]),
                       feature_seed=int(cfg["feature_seed"]), collocation_seed=int(cfg["collocation_seed"]),
                       noise_seed=_json_value(observations.get("noise_seed")), n_observations=len(yobs),
                       epochs_completed=int(step.numpy()), training_wall_time_s=elapsed,
                       setup_wall_time_s_this_call=started - call_started,
                       total_wall_time_s_this_call=time.perf_counter() - call_started,
                       resumed_from_step=start_step, terminal_observation_discrepancy_l2=discrepancy,
                       terminal_observation_relative_l2=discrepancy / max(float(np.linalg.norm(yobs)), 1e-15),
                       discrepancy_to_noise=None if noise_norm is None or float(noise_norm) <= 0 else discrepancy / float(noise_norm),
                       independent_collocation_pde_rmse=math.sqrt(final_v[1]) / float(cfg["residual_scale"]),
                       boundary_rmse=math.sqrt(final_v[2]), truth_free_score=final_v[0],
                       primary_checkpoint="final fixed budget", independent_forward_check="pending downstream notebook",
                       initial_truth_used_in_training=False, signature=signature)
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(run_dir / "environment.json", dict(python=platform.python_version(), tensorflow=tf.__version__, numpy=np.__version__,
                    platform=platform.platform(), devices=[d.name for d in tf.config.list_physical_devices()],
                    engine=_ENGINE_VERSION, source_sha256=identity["source_sha256"]))
        _write_json(history_path, dict(records=history))
        _write_json(status_file, dict(status="completed", signature=signature, profile=cfg["profile"], epochs_completed=int(step.numpy())))
        return dict(run_dir=str(run_dir), status="completed", metrics=metrics,
                    predictions_file=str(run_dir / "predictions.npz"), signature=signature)
    except BaseException as exc:
        # A failed or interrupted run remains visible in the failure ledger.
        _write_json(status_file, dict(status="failed" if not isinstance(exc, KeyboardInterrupt) else "interrupted", signature=signature,
                    profile=cfg["profile"], epochs_completed=int(step.numpy()), error_type=type(exc).__name__, error=str(exc)))
        _write_json(history_path, dict(records=history))
        raise
