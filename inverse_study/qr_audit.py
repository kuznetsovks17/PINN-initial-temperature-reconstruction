"""One-dimensional P1 mixed quasi-reversibility closure audit.

This independent SciPy implementation transcribes the supplied brass notebook's
mixed blocks and the nonlinear FreeFEM 1D form. It is a 1D diagnostic, not a
validation of the archived 2D/3D FreeFEM calculations. Changing the auxiliary
boundary condition changes the regularised inverse problem.
"""
from __future__ import annotations

import warnings
from time import perf_counter
import numpy as np
from scipy.sparse import bmat, diags
from scipy.sparse.linalg import splu

from .numerics import boundary_values, diffusivity, forward_1d


def assemble_p1_matrices(x, nodal_coefficient=None):
    """Return consistent P1 mass M and stiffness K on an increasing 1D mesh.

    With nodal_coefficient, K_ij=integral a_h*phi'_i*phi'_j dx, integrating
    the P1 interpolant a_h exactly. This is the archived lagged-coefficient
    convention; it is not a nonlinear Newton solve at the new time level.
    """
    x = np.asarray(x, dtype=float).ravel()
    h = np.diff(x)
    if len(x) < 3 or np.any(h <= 0):
        raise ValueError("At least three increasing mesh nodes are required")
    a = np.ones(len(x)) if nodal_coefficient is None else np.asarray(nodal_coefficient, dtype=float).ravel()
    if len(a) != len(x) or np.any(a <= 0) or not np.all(np.isfinite(a)):
        raise ValueError("Positive finite diffusivity is required at every mesh node")
    md = np.r_[h[0] / 3, (h[:-1] + h[1:]) / 3, h[-1] / 3]
    M = diags([h / 6, md, h / 6], [-1, 0, 1], format="csc")
    f = .5 * (a[:-1] + a[1:]) / h
    kd = np.r_[f[0], f[:-1] + f[1:], f[-1]]
    K = diags([-f, kd, -f], [-1, 0, 1], format="csc")
    return M, K


def solve_qr_1d(problem, observations, gamma, nx=81, dt=.005,
                closure="archived", safety_bounds=(-5.0, 5.0),
                forward_options=None):
    """Run the archived-style or compatible/Navier 1D mixed P1 QR system.

    Reverse time s=T-t, q=sqrt(gamma)*w_xx:

      (M-ds*K_a) w_new - ds*sqrt(gamma)*K q_new = M w_old
      sqrt(gamma)*K w_new + M q_new = 0.

    K_a contains Fo*a(w_old); gamma is independent of Fo. Only w has known
    essential boundary values in ``closure='archived'``. The unrestricted
    auxiliary test space implies a natural normal-derivative closure for w.
    ``closure='navier'`` uses q=0 at both endpoints and restricts its test space.
    The second branch is a different regularisation, provided for comparison.

    ``nx`` counts NODES, unlike the source brass notebook's NX (elements).
    Sparse terminal observations are linearly interpolated with known endpoint
    values to initialise the entire QR mesh. This extra preprocessing is
    explicitly returned and must not be called identical point-data assimilation.

    Stable reconstruction is independently forward-propagated by forward_1d
    (Kirchhoff-flux/BDF). No true initial field enters this routine. An unstable
    candidate returns stable=False, residual_norm=None, failed_step and reason;
    its last iterate must not be treated as a reconstructed t=0 state.
    """
    start = perf_counter()
    if int(problem.get("dim", 1)) != 1:
        raise ValueError("This audit is 1D only; it does not validate 2D/3D archived QR")
    if closure not in ("archived", "navier"):
        raise ValueError("closure must be 'archived' or 'navier'")
    if gamma < 0 or not np.isfinite(gamma) or dt <= 0 or nx < 4:
        raise ValueError("Require gamma>=0, dt>0, nx>=4")
    T = float(problem["T"])
    if T <= 0 or float(problem["Fo"]) <= 0:
        raise ValueError("T and Fo must be positive")
    b = np.asarray(problem.get("bounds", [[0., 1.]]), dtype=float)[0]
    x = np.linspace(b[0], b[1], int(nx))
    X = np.asarray(observations["X_obs"], dtype=float).reshape(-1)
    y = np.asarray(observations["y_obs"], dtype=float).ravel()
    if not len(X) or len(X) != len(y) or np.any(X <= b[0]) or np.any(X >= b[1]) or not np.all(np.isfinite(y)):
        raise ValueError("Provide strictly interior point observations with finite values")
    order = np.argsort(X)
    if np.any(np.diff(X[order]) <= 0):
        raise ValueError("Duplicate observation coordinates require an explicit aggregation rule")
    terminal_boundaries = boundary_values(problem, T)
    w = np.interp(x, np.r_[b[0], X[order], b[1]], np.r_[terminal_boundaries[0], y[order], terminal_boundaries[1]])
    mesh_terminal = w.copy()
    interpolation_residual = float(np.linalg.norm(np.interp(X, x, w) - y))
    M, K = assemble_p1_matrices(x)
    interior = np.arange(1, int(nx) - 1)
    boundary = np.array([0, int(nx) - 1])
    aux_nodes = np.arange(int(nx)) if closure == "archived" else interior
    nsteps = int(np.ceil(T / dt))
    ds = T / nsteps
    sqrt_gamma = np.sqrt(float(gamma))
    B = -ds * sqrt_gamma * K
    C = sqrt_gamma * K
    B_iq = B[interior][:, aux_nodes]
    C_qi = C[aux_nodes][:, interior]
    C_qb = C[aux_nodes][:, boundary]
    M_qq = M[aux_nodes][:, aux_nodes]
    lower, upper = map(float, safety_bounds)
    if not lower < upper:
        raise ValueError("safety_bounds must be increasing")
    fixed_law = problem.get("law", "constant") == "constant"
    stable, failure_reason, failed_step = True, None, None
    q = np.zeros(int(nx))
    algebraic_residual = 0.0
    inverse_start = perf_counter()
    lu = None
    block = None
    A_ib = None
    completed_steps = 0
    for step in range(1, nsteps + 1):
        physical_time = T - step * ds
        g = boundary_values(problem, max(0.0, physical_time))
        try:
            if not fixed_law or lu is None:
                _, K_a = assemble_p1_matrices(x, float(problem["Fo"]) * diffusivity(problem, w))
                A = M - ds * K_a
                A_ii = A[interior][:, interior]
                A_ib = A[interior][:, boundary]
                block = bmat([[A_ii, B_iq], [C_qi, M_qq]], format="csc")
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    lu = splu(block)
            rhs1 = (M @ w)[interior] - A_ib @ g
            rhs2 = -(C_qb @ g)
            rhs = np.r_[rhs1, rhs2]
            sol = lu.solve(rhs)
            new_w = np.empty(int(nx))
            new_w[boundary] = g
            new_w[interior] = sol[:len(interior)]
            q.fill(0.0)
            q[aux_nodes] = sol[len(interior):]
            if not np.all(np.isfinite(new_w)) or not np.all(np.isfinite(q)):
                raise FloatingPointError("Non-finite QR iterate")
            if np.min(new_w) < lower or np.max(new_w) > upper:
                w = new_w
                raise FloatingPointError("QR iterate left the declared safety interval")
            diffusivity(problem, new_w)
            w = new_w
            completed_steps = step
            if step == nsteps:
                algebraic_residual = float(np.linalg.norm(block @ sol - rhs) / max(np.linalg.norm(rhs), 1e-30))
        except (ValueError, RuntimeError, FloatingPointError, Warning) as exc:
            stable, failed_step, failure_reason = False, int(step), str(exc)
            break
    inverse_seconds = float(perf_counter() - inverse_start)
    result = {"x": x, "theta0": w.copy() if stable else None, "last_iterate": w.copy(),
              "auxiliary_q": q.copy(), "mesh_terminal_input": mesh_terminal,
              "closure": closure, "gamma": float(gamma), "nx_nodes": int(nx),
              "nsteps": nsteps, "completed_steps": int(completed_steps),
              "dt_requested": float(dt), "dt_actual": float(ds),
              "stable": bool(stable), "failed_step": failed_step, "failure_reason": failure_reason,
              "inverse_seconds": inverse_seconds, "forward_seconds": 0.0,
              "algebraic_relative_residual": algebraic_residual if stable else None,
              "y_fit": None, "terminal_forward": None, "residual_norm": None,
              "preprocessing": "piecewise-linear interior sensor interpolation with exact known boundary endpoints onto full QR mesh",
              "sensor_to_mesh_backprojection_residual": interpolation_residual,
              "comparison_scope": "independent 1D P1 closure audit; does not validate archived 2D/3D FreeFEM",
              "selection_uses_initial_truth": False}
    if stable:
        forward_start = perf_counter()
        options = dict(rtol=1e-7, atol=1e-9)
        if forward_options:
            options.update(forward_options)
        try:
            fw = forward_1d(problem, x, w, times=[T], **options)
            y_fit = np.interp(X, x, fw["terminal"])
            result.update(y_fit=y_fit, terminal_forward=fw["terminal"],
                          residual_norm=float(np.linalg.norm(y_fit - y)),
                          forward_seconds=float(perf_counter() - forward_start),
                          initial_endpoint_slopes=np.array([(w[1] - w[0]) / (x[1] - x[0]),
                                                            (w[-1] - w[-2]) / (x[-1] - x[-2])]))
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            result["stable"] = False
            result["failure_reason"] = "Independent forward replay failed: " + str(exc)
            result["forward_seconds"] = float(perf_counter() - forward_start)
    result["wall_seconds"] = float(perf_counter() - start)
    return result


def select_qr_by_discrepancy(results, noise_norm, numerical_noise_floor, tau=1.1):
    """Choose largest tested gamma meeting the forward-replay discrepancy.

    Apply separately to each closure and mesh/time resolution. The input floor
    must include measured forward/interpolation discretisation uncertainty.
    It must not be tuned to the true initial field. Returned selection is
    provisional until the chosen gamma and closure pass mesh/time studies.
    """
    if numerical_noise_floor <= 0 or noise_norm < 0 or tau < 1 or not results:
        raise ValueError("Need results, positive numerical floor, nonnegative noise, tau>=1")
    if len({r["closure"] for r in results}) != 1:
        raise ValueError("Do not mix closures in a single parameter-selection rule")
    if len({(r["nx_nodes"], r["dt_actual"]) for r in results}) != 1:
        raise ValueError("Select gamma separately at each mesh/time resolution")
    target = float(tau) * max(float(noise_norm), float(numerical_noise_floor))
    feasible = [i for i, r in enumerate(results) if r["stable"] and r["residual_norm"] is not None and r["residual_norm"] <= target]
    stable = [i for i, r in enumerate(results) if r["stable"] and r["residual_norm"] is not None]
    if feasible:
        index = max(feasible, key=lambda i: results[i]["gamma"])
        status = "discrepancy_attained"
        if results[index]["gamma"] == max(r["gamma"] for r in results):
            status += "_at_strongest_grid_edge_expand_gamma"
    elif stable:
        index = min(stable, key=lambda i: results[i]["residual_norm"])
        status = "discrepancy_unattainable_among_stable_candidates"
    else:
        index, status = None, "no_stable_candidate"
    return {"selected_index": index, "selected_gamma": results[index]["gamma"] if index is not None else None,
            "selection_status": status, "target_residual": target,
            "selected_result": results[index] if index is not None else None,
            "numerical_noise_floor": float(numerical_noise_floor)}


def run_qr_self_checks():
    """P1 assembly and Navier discrete-mode checks, plus a closure difference."""
    x = np.linspace(0.0, 1.0, 41)
    M, K = assemble_p1_matrices(x)
    one = np.ones(len(x))
    assert abs(float(one @ M @ one) - 1.) < 1e-13
    assert np.linalg.norm(K @ one) < 1e-12
    p = {"dim": 1, "law": "constant", "Fo": .1, "T": .1, "bounds": [[0, 1]]}
    truth = np.sin(np.pi * x)
    y = truth[1:-1] * np.exp(-p["Fo"] * np.pi ** 2 * p["T"])
    obs = {"X_obs": x[1:-1, None], "y_obs": y, "noise_norm": 0.0}
    gamma, dt = 1e-4, .001
    navier = solve_qr_1d(p, obs, gamma, nx=len(x), dt=dt, closure="navier")
    assert navier["stable"]
    h = x[1] - x[0]
    fe_lambda = 6 * (1 - np.cos(np.pi * h)) / (h * h * (2 + np.cos(np.pi * h)))
    ds, nsteps = navier["dt_actual"], navier["nsteps"]
    expected_amplitude = np.exp(-p["Fo"] * np.pi ** 2 * p["T"]) * (1 - ds * p["Fo"] * fe_lambda + ds * gamma * fe_lambda ** 2) ** (-nsteps)
    error = float(np.max(np.abs(navier["theta0"] - expected_amplitude * truth)))
    assert error < 1e-10, error
    archived = solve_qr_1d(p, obs, gamma, nx=len(x), dt=dt, closure="archived")
    assert archived["stable"]
    difference = float(np.linalg.norm(archived["theta0"] - navier["theta0"]) / np.linalg.norm(navier["theta0"]))
    assert difference > 1e-3, "This check should expose the different auxiliary closure"
    return {"status": "passed; software/closure checks only", "P1_Navier_discrete_mode_linf": error,
            "archived_vs_navier_relative_difference": difference,
            "archived_forward_residual": archived["residual_norm"],
            "navier_forward_residual": navier["residual_norm"],
            "scope": "1D only, not archived 2D/3D validation"}
