"""Independent classical and forward solvers for the inverse-study notebooks.

New verification code, not a reimplementation/validation of the archived FreeFEM
QR system. Units are those of ``problem``: u_t = Fo div(a(u) grad u).
Synthetic truth is never accepted by an inverse-parameter selection function.

The nonlinear case definition and brass coefficients come from the supplied
``epinn_revised_core.py`` and brass notebook. The finite-difference/BDF forward
solver and discrete-observation spectral solvers below are independent code.
"""
from __future__ import annotations

from itertools import product
from time import perf_counter
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import svd
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import expm_multiply


def _bounds(problem, dim=None):
    dim = int(problem.get("dim", 1) if dim is None else dim)
    b = np.asarray(problem.get("bounds", [[0.0, 1.0]] * dim), dtype=float)
    if b.shape != (dim, 2) or np.any(b[:, 1] <= b[:, 0]):
        raise ValueError("bounds must contain one increasing [min,max] pair per dimension")
    return b


def full_sine_modes(dim, kmax):
    """All positive tensor-product mode indices, not an oracle two-mode set."""
    dim, kmax = int(dim), int(kmax)
    if dim < 1 or kmax < 1:
        raise ValueError("dim and kmax must be positive")
    return np.asarray(list(product(range(1, kmax + 1), repeat=dim)), dtype=int)


def sine_matrix(X, modes, bounds=None):
    """Unnormalised product-sine basis. c=(1,.25) retains manuscript amplitudes."""
    X = np.asarray(X, dtype=float)
    modes = np.asarray(modes, dtype=int)
    if X.ndim == 1:
        X = X[:, None]
    if modes.ndim == 1:
        modes = modes[:, None]
    if X.ndim != 2 or modes.ndim != 2 or X.shape[1] != modes.shape[1]:
        raise ValueError("X must be (n,d), modes must be (m,d)")
    if np.any(modes < 1):
        raise ValueError("Dirichlet sine indices must be positive")
    b = _bounds({"dim": X.shape[1], "bounds": bounds}) if bounds is not None else _bounds({"dim": X.shape[1]})
    z = (X - b[:, 0]) / (b[:, 1] - b[:, 0])
    return np.prod(np.sin(np.pi * z[:, None, :] * modes[None, :, :]), axis=2)


def laplacian_eigenvalues(modes, bounds=None):
    modes = np.asarray(modes, dtype=int)
    if modes.ndim == 1:
        modes = modes[:, None]
    b = _bounds({"dim": modes.shape[1], "bounds": bounds}) if bounds is not None else _bounds({"dim": modes.shape[1]})
    return np.pi ** 2 * np.sum((modes / (b[:, 1] - b[:, 0])) ** 2, axis=1)


def _linear_coefficient(problem):
    if problem.get("law", "constant") != "constant":
        raise ValueError("Sine-basis heat propagation only supports constant diffusivity")
    coefficient = float(problem["Fo"]) * float(problem.get("diffusivity_multiplier", 1.0))
    if coefficient <= 0:
        raise ValueError("Fo*diffusivity_multiplier must be positive")
    return coefficient


def forward_matrix(problem, X, modes):
    """Map initial coefficients to actual point observations at time problem['T']."""
    b = _bounds(problem)
    eigenvalues = laplacian_eigenvalues(modes, b)
    attenuation = np.exp(-_linear_coefficient(problem) * float(problem["T"]) * eigenvalues)
    return sine_matrix(X, modes, b) * attenuation[None, :]


def _noise_target(y, observations, tau, numerical_noise_floor):
    delta = float(observations.get("noise_norm", 0.0))
    if not np.isfinite(delta) or delta < 0 or tau < 1:
        raise ValueError("noise_norm must be nonnegative and tau >= 1")
    # This floor is just floating-point arithmetic for the exact linear matrix;
    # callers must supply measured forward/discretisation error for numerical data.
    default_floor = 100 * np.finfo(float).eps * max(float(np.linalg.norm(y)), 1.0)
    floor = default_floor if numerical_noise_floor is None else float(numerical_noise_floor)
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("numerical_noise_floor must be positive")
    target = float(tau) * max(delta, floor)
    return target, floor, ("exact_matrix_roundoff_only" if numerical_noise_floor is None else "caller_supplied_numerical_floor")


def solve_spectral(problem, observations, kmax=12, method="tikhonov", tau=1.1,
                   numerical_noise_floor=None, prior_order=0.0, alphas=None,
                   modes=None, rcond=None):
    """Solve an inverse heat problem at its true finite observation locations.

    Minimise ||A c-y||_2^2 + alpha ||L c||_2^2, L_kk=(1+lambda_k)^(p/2),
    or truncate the SVD of A L^-1. Select the strongest regularisation whose
    *absolute* observation residual is <= tau*max(noise_norm,numerical_floor).
    For TSVD this is the smallest retained rank. No true initial state is used.

    ``modes`` is optional for explicit diagnostics only; using the two truth
    modes must be labelled an oracle restricted-family benchmark. Main runs
    should use ``kmax`` and inspect basis sensitivity.

    Returns arrays and Python scalars; save arrays with np.savez and scalar/path
    records with JSON/CSV. Status is never silently changed to 'success' when
    no candidate attains the discrepancy target.
    """
    start = perf_counter()
    X = np.asarray(observations["X_obs"], dtype=float)
    y = np.asarray(observations["y_obs"], dtype=float).ravel()
    if X.ndim == 1:
        X = X[:, None]
    if X.shape != (len(y), int(problem["dim"])) or not np.all(np.isfinite(y)):
        raise ValueError("Invalid observation shapes/values")
    b = _bounds(problem)
    if np.any(X <= b[:, 0]) or np.any(X >= b[:, 1]):
        raise ValueError("Primary comparison uses interior observations; boundary data are prescribed separately")
    explicit_modes = modes is not None
    modes = full_sine_modes(problem["dim"], kmax) if modes is None else np.asarray(modes, dtype=int)
    A = forward_matrix(problem, X, modes)
    eigenvalues = laplacian_eigenvalues(modes, b)
    penalty = (1.0 + eigenvalues) ** (float(prior_order) / 2.0)
    B = A / penalty[None, :]
    U, singular, Vh = svd(B, full_matrices=False, check_finite=True)
    cutoff = (max(B.shape) * np.finfo(float).eps if rcond is None else float(rcond)) * singular[0]
    valid = singular > cutoff
    rank = int(np.sum(valid))
    if rank == 0:
        raise ValueError("Forward matrix has zero numerical rank at this exposure")
    projected = U.T @ y
    target, floor, floor_source = _noise_target(y, observations, tau, numerical_noise_floor)
    coefficients, path = [], []
    if method == "tikhonov":
        if alphas is None:
            parameters = np.r_[0.0, singular[0] ** 2 * np.logspace(-18, 4, 113)]
        else:
            parameters = np.unique(np.asarray(alphas, dtype=float))
        if np.any(parameters < 0) or not np.all(np.isfinite(parameters)):
            raise ValueError("alphas must be finite and nonnegative")
        for alpha in parameters:
            weights = np.zeros_like(singular)
            if alpha == 0:
                weights[valid] = 1.0 / singular[valid]
            else:
                weights = singular / (singular ** 2 + alpha)
            c = (Vh.T @ (weights * projected)) / penalty
            residual = float(np.linalg.norm(A @ c - y))
            coefficients.append(c)
            path.append({"parameter": float(alpha), "residual_norm": residual,
                         "penalty_norm": float(np.linalg.norm(penalty * c)),
                         "effective_dof": float(np.sum(singular ** 2 / (singular ** 2 + alpha))) if alpha else float(rank)})
        eligible = [i for i, row in enumerate(path) if row["residual_norm"] <= target]
        selected = eligible[-1] if eligible else int(np.argmin([row["residual_norm"] for row in path]))
        status = "discrepancy_attained" if eligible else "discrepancy_unattainable_in_grid"
        if eligible and selected == len(path) - 1:
            status = "discrepancy_attained_at_strongest_grid_edge_expand_alpha"
    elif method == "tsvd":
        for kept in range(rank + 1):
            c = (Vh[:kept].T @ (projected[:kept] / singular[:kept])) / penalty if kept else np.zeros(A.shape[1])
            coefficients.append(c)
            path.append({"parameter": int(kept), "residual_norm": float(np.linalg.norm(A @ c - y)),
                         "penalty_norm": float(np.linalg.norm(penalty * c)), "effective_dof": float(kept)})
        eligible = [i for i, row in enumerate(path) if row["residual_norm"] <= target]
        selected = eligible[0] if eligible else rank
        status = "discrepancy_attained" if eligible else "discrepancy_unattainable_at_full_rank"
    else:
        raise ValueError("method must be 'tikhonov' or 'tsvd'")
    c = coefficients[selected]
    return {"method": method, "modes": modes, "coefficients": c,
            "singular_values": singular, "numerical_rank": rank,
            "number_of_modes": len(modes), "number_of_observations": len(y),
            "selected_index": int(selected), "selected_parameter": path[selected]["parameter"],
            "selection_status": status, "status": status, "path": path,
            "target_residual": target, "noise_norm": float(observations.get("noise_norm", 0.0)),
            "numerical_noise_floor": floor, "numerical_floor_source": floor_source,
            "prior_order": float(prior_order), "explicit_restricted_modes": explicit_modes,
            "y_fit": A @ c, "residual_norm": path[selected]["residual_norm"],
            "wall_seconds": float(perf_counter() - start)}


def evaluate_spectral(result, X, t, problem):
    """Evaluate a recovered initial state or its exact linear continuation."""
    modes = result["modes"]
    b = _bounds(problem)
    attenuation = np.exp(-_linear_coefficient(problem) * float(t) * laplacian_eigenvalues(modes, b))
    return sine_matrix(X, modes, b) @ (np.asarray(result["coefficients"]) * attenuation)


def forward_linear_grid(u0, axes, Fo, T):
    """Independent finite-difference forward replay on a 1D/2D/3D tensor grid.

    ``u0.shape == tuple(len(a) for a in axes)``; axes follow NumPy indexing='ij'.
    Prescribed zero Dirichlet boundaries are imposed and the correction is
    reported. Time integration is scipy's expm_multiply of the semidiscrete
    Laplacian, not the inverse solver's sine operator. Refine axes to verify
    spatial error. The returned array has the same axes and shape as u0.
    """
    start = perf_counter()
    u0 = np.asarray(u0, dtype=float)
    axes = [np.asarray(a, dtype=float) for a in axes]
    shape = tuple(len(a) for a in axes)
    if len(axes) not in (1, 2, 3) or u0.shape != shape or any(n < 3 for n in shape):
        raise ValueError("u0 must match 1-3 tensor axes, each with >=3 points")
    if Fo <= 0 or T < 0:
        raise ValueError("Fo must be positive and T nonnegative")
    operators = []
    for a in axes:
        dx = np.diff(a)
        if np.any(dx <= 0) or not np.allclose(dx, dx[0], rtol=1e-10, atol=1e-12):
            raise ValueError("Independent FD replay requires increasing uniform axes")
        n = len(a) - 2
        operators.append(diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1], format="csc") / dx[0] ** 2)
    inner_shape = tuple(n - 2 for n in shape)
    L = csc_matrix((int(np.prod(inner_shape)), int(np.prod(inner_shape))))
    for j, op in enumerate(operators):
        term = csc_matrix([[1.0]])
        for k, n in enumerate(inner_shape):
            term = kron(term, op if j == k else eye(n, format="csc"), format="csc")
        L = L + term
    interior = tuple(slice(1, -1) for _ in axes)
    boundary = np.ones(shape, dtype=bool)
    boundary[interior] = False
    terminal = np.zeros(shape)
    propagator = float(Fo) * float(T) * L
    terminal[interior] = expm_multiply(propagator, u0[interior].ravel(order="C"),
                                      traceA=float(propagator.diagonal().sum())).reshape(inner_shape, order="C")
    return {"terminal": terminal, "axes": axes,
            "initial_boundary_linf": float(np.max(np.abs(u0[boundary]))),
            "initial_boundary_l2": float(np.linalg.norm(u0[boundary])),
            "method": "centered_fd_laplacian_expm_multiply", "wall_seconds": float(perf_counter() - start)}


def boundary_values(problem, t):
    """Return known [left,right] Dirichlet values in the current time units."""
    t = float(t)
    if "boundary_times" in problem:
        times = np.asarray(problem["boundary_times"], dtype=float)
        left = np.asarray(problem["boundary_left"], dtype=float)
        right = np.asarray(problem["boundary_right"], dtype=float)
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0) or len(left) != len(times) or len(right) != len(times):
            raise ValueError("Boundary histories must have matching increasing time samples")
        if t < times[0] - 1e-12 or t > times[-1] + 1e-12:
            raise ValueError("Boundary history does not cover the requested time")
        return np.array([np.interp(t, times, left), np.interp(t, times, right)])
    if problem.get("law") == "exp_minus_theta_minus_one":
        return np.array([1.0 - 0.5 * t / float(problem["T"]), 0.8 - 0.3 * t / float(problem["T"])])
    return np.asarray(problem.get("boundary_values", [0.0, 0.0]), dtype=float)


def diffusivity(problem, u):
    """Dimensionless a(u); Fo is applied separately, exactly once."""
    u = np.asarray(u, dtype=float)
    law = problem.get("law", "constant")
    if law == "constant":
        a = np.ones_like(u)
    elif law == "exp_minus_theta_minus_one":
        beta = float(problem.get("beta", 1.0))
        a = np.exp(-beta * u - 1.0)
    elif law == "brass":
        a = float(problem.get("B0", 0.287265051903114)) + float(problem.get("B1", 0.011656401384083)) * u
    else:
        raise ValueError(f"Unsupported law: {law}")
    a = a * float(problem.get("diffusivity_multiplier", 1.0))
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("Diffusivity must remain finite and positive")
    return a


def _kirchhoff(problem, u):
    u = np.asarray(u, dtype=float)
    law = problem.get("law", "constant")
    if law == "constant":
        A = u
    elif law == "exp_minus_theta_minus_one":
        beta = float(problem.get("beta", 1.0))
        A = -np.exp(-beta * u - 1.0) / beta if beta != 0 else np.exp(-1.0) * u
    elif law == "brass":
        A = float(problem.get("B0", 0.287265051903114)) * u + 0.5 * float(problem.get("B1", 0.011656401384083)) * u ** 2
    else:
        raise ValueError(f"Unsupported law: {law}")
    return float(problem.get("diffusivity_multiplier", 1.0)) * A


def forward_1d(problem, x, u0, times=None, rtol=1e-7, atol=1e-9,
               max_step=np.inf, enforce_initial_boundary=True):
    """Conservative nonlinear 1D forward solve with an analytic sparse Jacobian.

    Face flux is [A(u[i+1])-A(u[i])]/dx, where A'=a. BDF integrates
    u_t=Fo*Delta_h A(u), using the known time-dependent Dirichlet data.
    This differs independently from the archived harmonic-flux/Picard scheme.
    It supplies numerical evidence only after spatial/time convergence checks.
    """
    start = perf_counter()
    x = np.asarray(x, dtype=float).ravel()
    u0 = np.asarray(u0, dtype=float).ravel().copy()
    if int(problem.get("dim", 1)) != 1 or len(x) < 3 or len(u0) != len(x):
        raise ValueError("1D x/u0 shapes must match and include both boundary nodes")
    dx = np.diff(x)
    if np.any(dx <= 0) or not np.allclose(dx, dx[0], rtol=1e-10, atol=1e-12):
        raise ValueError("forward_1d requires an increasing uniform grid")
    expected_bounds = _bounds(problem, 1)[0]
    if not np.allclose(x[[0, -1]], expected_bounds):
        raise ValueError("x must include both domain endpoints")
    times = np.asarray([0.0, float(problem["T"])] if times is None else times, dtype=float).ravel()
    if not len(times) or np.any(np.diff(times) <= 0) or times[0] < 0 or times[-1] > float(problem["T"]) + 1e-12:
        raise ValueError("times must be increasing within [0,T]")
    correction = u0[[0, -1]] - boundary_values(problem, 0.0)
    if not enforce_initial_boundary and np.max(np.abs(correction)) > 1e-10:
        raise ValueError("Initial and prescribed boundary values are incompatible")
    u0[[0, -1]] = boundary_values(problem, 0.0)
    diffusivity(problem, u0)
    factor = float(problem["Fo"]) / dx[0] ** 2
    if factor <= 0:
        raise ValueError("Fo must be positive")
    n = len(x) - 2

    def fill(t, z):
        u = np.empty(len(x))
        u[1:-1] = z
        u[[0, -1]] = boundary_values(problem, t)
        return u

    def rhs(t, z):
        u = fill(t, z)
        diffusivity(problem, u)
        A = _kirchhoff(problem, u)
        return factor * (A[2:] - 2 * A[1:-1] + A[:-2])

    def jac(t, z):
        a = diffusivity(problem, z)
        return factor * diags([a[:-1], -2 * a, a[1:]], [-1, 0, 1], shape=(n, n), format="csc")

    if times[-1] == 0:
        values = u0[None, :]
        diagnostics = {"nfev": 0, "njev": 0, "nlu": 0}
    else:
        sol = solve_ivp(rhs, (0.0, float(times[-1])), u0[1:-1], method="BDF", t_eval=times,
                        rtol=rtol, atol=atol, max_step=max_step, jac=jac)
        if not sol.success:
            raise RuntimeError(f"Forward BDF solve failed: {sol.message}")
        values = np.stack([fill(t, z) for t, z in zip(sol.t, sol.y.T)])
        diffusivity(problem, values)
        diagnostics = {"nfev": int(sol.nfev), "njev": int(sol.njev), "nlu": int(sol.nlu)}
    return {"x": x, "t": times, "u": values, "terminal": values[-1],
            "initial_boundary_correction": correction,
            "method": "kirchhoff_flux_centered_fd_BDF", "rtol": float(rtol), "atol": float(atol),
            "max_step": float(max_step), "minimum_temperature": float(values.min()),
            "maximum_temperature": float(values.max()), "diagnostics": diagnostics,
            "wall_seconds": float(perf_counter() - start)}


def nonlinear_tikhonov(problem, observations, n_knots=9, alphas=None, nx=81,
                       tau=1.1, numerical_noise_floor=None, rtol=2e-6, atol=2e-8,
                       max_step=np.inf, max_nfev=100, value_bounds=(0.0, 2.0),
                       initial_guess=None):
    """Optional slow 1D finite-parameter nonlinear Tikhonov comparator.

    The initial profile is piecewise linear on n_knots nodes; both endpoints
    are fixed by the known boundary histories. Interior parameters are fitted
    to the *same point observations* as the PINN. The penalty is the integrated
    squared second difference. Each alpha starts from the same data-derived
    guess; the truth is never used. A result is acceptable for discrepancy
    selection only if least_squares reports convergence. Mesh, knot count,
    tolerance, bounds and alpha-grid sensitivity remain required experiments.
    This new comparator does not repair or stand in for the archived QR code.
    """
    start = perf_counter()
    if int(problem.get("dim", 1)) != 1:
        raise ValueError("nonlinear_tikhonov is 1D only")
    if numerical_noise_floor is None:
        raise ValueError("Supply a positive numerical_noise_floor from nonlinear forward convergence")
    X = np.asarray(observations["X_obs"], dtype=float).reshape(-1)
    y = np.asarray(observations["y_obs"], dtype=float).ravel()
    b = _bounds(problem, 1)[0]
    if len(X) != len(y) or np.any(X <= b[0]) or np.any(X >= b[1]):
        raise ValueError("Use identical strictly interior point observations")
    if n_knots < 4 or nx < n_knots:
        raise ValueError("Require nx >= n_knots >= 4")
    knots = np.linspace(*b, int(n_knots))
    x = np.linspace(*b, int(nx))
    endpoints = boundary_values(problem, 0.0)
    target, floor, floor_source = _noise_target(y, observations, tau, numerical_noise_floor)
    if alphas is None:
        alphas = np.logspace(-8, 0, 9)
    alphas = np.unique(np.asarray(alphas, dtype=float))
    if np.any(alphas < 0) or not np.all(np.isfinite(alphas)):
        raise ValueError("alphas must be finite and nonnegative")
    lower, upper = map(float, value_bounds)
    if not lower < upper or np.any(endpoints < lower) or np.any(endpoints > upper):
        raise ValueError("Invalid profile bounds or incompatible endpoint data")
    if initial_guess is None:
        guess = np.full(int(n_knots) - 2, np.mean(y))
    else:
        guess = np.asarray(initial_guess, dtype=float).ravel()
        if len(guess) != n_knots - 2:
            raise ValueError("initial_guess supplies interior knot values only")
    guess = np.clip(guess, lower + 1e-8 * (upper - lower), upper - 1e-8 * (upper - lower))
    h = knots[1] - knots[0]
    candidates, path = [], []
    for alpha in alphas:
        candidate_start = perf_counter()
        cache = {}

        def simulate(z):
            key = np.asarray(z).tobytes()
            if cache.get("key") != key:
                c = np.r_[endpoints[0], z, endpoints[1]]
                initial = np.interp(x, knots, c)
                fw = forward_1d(problem, x, initial, [float(problem["T"])], rtol, atol, max_step)
                cache.update(key=key, c=c, initial=initial, fw=fw,
                             predicted=np.interp(X, x, fw["terminal"]))
            return cache

        def residual(z):
            values = simulate(z)
            curvature = np.diff(values["c"], n=2) / h ** 1.5
            return np.r_[values["predicted"] - y, np.sqrt(alpha) * curvature]

        opt = least_squares(residual, guess.copy(), bounds=(lower, upper), max_nfev=max_nfev,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8, diff_step=1e-4)
        values = dict(simulate(opt.x))
        values.pop("key", None)
        residual_norm = float(np.linalg.norm(values["predicted"] - y))
        path.append({"parameter": float(alpha), "residual_norm": residual_norm,
                     "penalty_norm": float(np.linalg.norm(np.diff(values["c"], n=2) / h ** 1.5)),
                     "optimizer_success": bool(opt.success), "optimizer_status": int(opt.status),
                     "optimizer_message": str(opt.message), "optimizer_nfev": int(opt.nfev),
                     "wall_seconds": float(perf_counter() - candidate_start)})
        candidates.append(values)
    eligible = [i for i, row in enumerate(path) if row["optimizer_success"] and row["residual_norm"] <= target]
    converged = [i for i, row in enumerate(path) if row["optimizer_success"]]
    if eligible:
        selected = eligible[-1]
        status = "discrepancy_attained" if selected < len(path) - 1 else "discrepancy_attained_at_strongest_grid_edge_expand_alpha"
    elif converged:
        selected = min(converged, key=lambda i: path[i]["residual_norm"])
        status = "discrepancy_unattainable_among_converged_candidates"
    else:
        selected = int(np.argmin([row["residual_norm"] for row in path]))
        status = "no_optimizer_candidate_converged"
    values = candidates[selected]
    return {"method": "nonlinear_piecewise_linear_tikhonov", "x": x, "knots": knots,
            "coefficients": values["c"], "initial": values["initial"],
            "terminal": values["fw"]["terminal"], "y_fit": values["predicted"],
            "selected_parameter": path[selected]["parameter"], "selected_index": int(selected),
            "selection_status": status, "status": status, "path": path,
            "target_residual": target, "residual_norm": path[selected]["residual_norm"],
            "numerical_noise_floor": floor, "numerical_floor_source": floor_source,
            "wall_seconds": float(perf_counter() - start)}


def qr_compatible_mode_diagnostic(modes, Fo, T, gammas, bounds=None):
    """Analytic QR filter ONLY for a Laplacian-compatible/Navier closure.

    This is NOT the supplied mixed FreeFEM system, whose unrestricted auxiliary
    test space imposes an additional natural normal-derivative closure for w.
    The returned attenuation is a diagnostic, never evidence that archived QR
    has been implemented correctly. gamma has the scaling of w_s=-Fo*Delta w
    -gamma*Delta^2 w in reverse time.
    """
    lam = laplacian_eigenvalues(modes, bounds)
    gammas = np.asarray(gammas, dtype=float).ravel()
    if Fo <= 0 or T < 0 or np.any(gammas < 0):
        raise ValueError("Invalid Fo/T/gammas")
    exponent = float(T) * (float(Fo) * lam[None, :] - gammas[:, None] * lam[None, :] ** 2)
    return {"eigenvalues": lam, "gammas": gammas,
            "relative_to_exact_inverse": np.exp(-gammas[:, None] * float(T) * lam[None, :] ** 2),
            "log_inverse_gain": exponent,
            "closure": "Laplacian-compatible/Navier diagnostic; NOT archived mixed FE closure"}


def run_self_checks():
    """Small actual numerical checks; these are software checks, not paper results."""
    report = {}
    p = {"name": "check", "dim": 2, "bounds": [[0, 1], [0, 1]], "Fo": 2e-4, "T": 100., "law": "constant"}
    grid = np.linspace(0, 1, 11)[1:-1]
    X = np.stack(np.meshgrid(grid, grid, indexing="ij"), axis=-1).reshape(-1, 2)
    modes = np.array([[1, 1], [2, 3]])
    clean = forward_matrix(p, X, modes) @ np.array([1., .25])
    obs = {"X_obs": X, "y_obs": clean, "noise_norm": 0.0}
    fit = solve_spectral(p, obs, kmax=5, numerical_noise_floor=1e-11)
    expected = sine_matrix(X, modes) @ np.array([1., .25])
    err = np.linalg.norm(evaluate_spectral(fit, X, 0., p) - expected) / np.linalg.norm(expected)
    assert err < 1e-6, f"Clean spectral reconstruction error {err}"
    report["clean_spectral_relative_error"] = float(err)
    rng = np.random.default_rng(4)
    noise = rng.normal(size=len(clean)); noise *= .05 * np.linalg.norm(clean) / np.linalg.norm(noise)
    noisy = {"X_obs": X, "y_obs": clean + noise, "noise_norm": float(np.linalg.norm(noise))}
    for method in ("tikhonov", "tsvd"):
        fit = solve_spectral(p, noisy, kmax=5, method=method, numerical_noise_floor=1e-11)
        assert fit["residual_norm"] <= fit["target_residual"]
        report[f"{method}_discrepancy_ratio"] = fit["residual_norm"] / fit["target_residual"]
    x = np.linspace(0, 1, 41)
    linear = {"dim": 1, "Fo": .1, "T": .05, "law": "constant", "bounds": [[0, 1]]}
    fw = forward_1d(linear, x, np.sin(np.pi * x), rtol=1e-9, atol=1e-11, max_step=.002)
    exact = np.sin(np.pi * x) * np.exp(-.1 * np.pi ** 2 * .05)
    error = float(np.max(np.abs(fw["terminal"] - exact)))
    assert error < 1e-4, f"Nonlinear-solver linear-limit error {error}"
    report["BDF_linear_limit_linf"] = error
    for dim in (1, 2, 3):
        axes = [np.linspace(0, 1, 9)] * dim
        XX = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        init = np.prod(np.sin(np.pi * XX), axis=-1)
        out = forward_linear_grid(init, axes, .1, .05)
        discrete_lam = dim * 4 * np.sin(np.pi / 16) ** 2 / (.125 ** 2)
        error = float(np.max(np.abs(out["terminal"] - init * np.exp(-.1 * .05 * discrete_lam))))
        assert error < 1e-10
        report[f"FD_expm_discrete_mode_d{dim}_linf"] = error
    nonlinear = {"dim": 1, "Fo": .1, "T": 1., "law": "exp_minus_theta_minus_one", "bounds": [[0, 1]]}
    u0 = np.where(x <= .5, 1., .8)
    nlfw = forward_1d(nonlinear, x, u0, rtol=1e-6, atol=1e-8, max_step=.05)
    assert nlfw["minimum_temperature"] >= .5 - 1e-7
    assert nlfw["maximum_temperature"] <= 1. + 1e-7
    assert np.allclose(nlfw["terminal"][[0, -1]], [.5, .5])
    report["nonlinear_positivity_and_boundary"] = True
    report["status"] = "passed; lightweight verification only"
    return report
