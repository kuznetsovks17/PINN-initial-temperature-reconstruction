# CELL 1
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix, bmat
from scipy.sparse.linalg import spsolve

plt.rcParams.update({"font.size": 12})

# CELL 3
PROFILE_TAU = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)

# columns:
# xi, theta(tau=0), theta(tau=0.2), theta(tau=0.4),
# theta(tau=0.6), theta(tau=0.8), theta(tau=1)
_EXP = np.array([
    [0.000000000000, 0.167843653, 0.199499979, 0.235610437, 0.275561737, 0.326274347, 0.376139326],
    [0.020408163265, 0.164309063, 0.198577611, 0.238609484, 0.275561737, 0.325045547, 0.377290511],
    [0.040816326531, 0.166305342, 0.200113005, 0.239222163, 0.275561737, 0.329883163, 0.380363874],
    [0.061224489796, 0.167843653, 0.203340911, 0.242756668, 0.278560784, 0.331650668, 0.382669921],
    [0.081632653061, 0.169994732, 0.203340911, 0.242756668, 0.283324084, 0.332882211, 0.385127174],
    [0.102040816327, 0.169994732, 0.203953937, 0.245749716, 0.283324084, 0.334111005, 0.386820195],
    [0.122448979592, 0.169994732, 0.206950068, 0.246904200, 0.285165563, 0.335185174, 0.388662016],
    [0.142857142857, 0.173835658, 0.206027358, 0.247516884, 0.286858584, 0.338874305, 0.394041084],
    [0.163265306122, 0.173835658, 0.208410805, 0.248136079, 0.287474353, 0.338261621, 0.394656858],
    [0.183673469388, 0.174754942, 0.207565837, 0.252283611, 0.291086600, 0.343021837, 0.398191700],
    [0.204081632653, 0.176757479, 0.211100511, 0.255895511, 0.297613979, 0.348401247, 0.403567684],
    [0.224489795918, 0.177370337, 0.215018674, 0.258888389, 0.298845863, 0.352551516, 0.407718300],
    [0.244897959184, 0.182749574, 0.216479579, 0.262422889, 0.303606074, 0.357314816, 0.412481600],
    [0.265306122449, 0.188435242, 0.220088400, 0.263654774, 0.307217979, 0.360388179, 0.416167642],
    [0.285714285714, 0.186284079, 0.221855905, 0.266034795, 0.307759774, 0.365151821, 0.422623968],
    [0.306122448980, 0.184900653, 0.225467811, 0.270801179, 0.316744574, 0.369918205, 0.428003379],
    [0.326530612245, 0.191199089, 0.231153479, 0.273181374, 0.322123984, 0.376368358, 0.432766679],
    [0.346938775510, 0.191969747, 0.233227584, 0.279789579, 0.326890374, 0.383591989, 0.442983700],
    [0.367346938776, 0.196807358, 0.234994405, 0.286320216, 0.331727816, 0.390200200, 0.448285368],
    [0.387755102041, 0.196807358, 0.240683163, 0.291086600, 0.338797163, 0.397266121, 0.456738826],
    [0.408163265306, 0.202109200, 0.244830689, 0.297078695, 0.345408111, 0.405102779, 0.465035947],
    [0.428571428571, 0.206411358, 0.245443374, 0.302380363, 0.352012889, 0.411097958, 0.472256837],
    [0.448979591837, 0.214712416, 0.253821663, 0.307759774, 0.359082242, 0.420627989, 0.483015316],
    [0.469387755102, 0.215325184, 0.260042784, 0.316209121, 0.366918900, 0.430154584, 0.494844537],
    [0.489795918367, 0.219472626, 0.266037963, 0.321510958, 0.375832816, 0.440293868, 0.503309695],
    [0.510204081633, 0.223007132, 0.272262005, 0.331115132, 0.383592247, 0.448595100, 0.514052616],
    [0.530612244898, 0.225777321, 0.277944758, 0.338797163, 0.393731532, 0.459350489, 0.524204458],
    [0.551020408163, 0.229305653, 0.283940111, 0.346021137, 0.403335700, 0.471337426, 0.535408779],
    [0.571428571429, 0.233765695, 0.291699368, 0.353702832, 0.411023905, 0.479019458, 0.547386584],
    [0.591836734694, 0.242756926, 0.301532226, 0.363845537, 0.421776211, 0.491003647, 0.559395800],
    [0.612244897959, 0.243598811, 0.305989268, 0.372298311, 0.433147374, 0.503619379, 0.571218847],
    [0.632653061224, 0.252667353, 0.315825547, 0.382437595, 0.445060511, 0.515445511, 0.585053863],
    [0.653061224490, 0.258046589, 0.324816521, 0.394350726, 0.456354274, 0.527423405, 0.598114963],
    [0.673469387755, 0.268421242, 0.333188811, 0.405028721, 0.468954232, 0.542310684, 0.613002374],
    [0.693877551020, 0.278325500, 0.344253716, 0.417016000, 0.482015679, 0.556610184, 0.627363542],
    [0.714285714286, 0.286320132, 0.354625326, 0.430689784, 0.495856774, 0.570909337, 0.642900963],
    [0.734693877551, 0.294079474, 0.362694226, 0.442677058, 0.508914747, 0.585951458, 0.657818974],
    [0.755102040816, 0.301532395, 0.376445758, 0.456354274, 0.525040426, 0.601488668, 0.674470632],
    [0.775510204082, 0.314135611, 0.389584084, 0.469570347, 0.539958437, 0.617459242, 0.691245895],
    [0.795918367347, 0.325658574, 0.403261300, 0.485089037, 0.555465000, 0.633615653, 0.707990384],
    [0.816326530612, 0.335262574, 0.415245489, 0.499388195, 0.571002205, 0.649741068, 0.725199016],
    [0.836734693878, 0.347175537, 0.428926474, 0.514300332, 0.587746737, 0.667104458, 0.743181384],
    [0.857142857143, 0.358853258, 0.443599684, 0.529187611, 0.604955368, 0.683693889, 0.760451905],
    [0.877551020408, 0.370530932, 0.457273642, 0.544725163, 0.619935263, 0.700469153, 0.777134295],
    [0.897959183673, 0.384205016, 0.473717789, 0.562614532, 0.639063032, 0.720122905, 0.797407158],
    [0.918367346939, 0.399417284, 0.491619758, 0.579916121, 0.657509716, 0.739962395, 0.816534711],
    [0.938775510204, 0.415248663, 0.506841284, 0.599012474, 0.676018284, 0.758997295, 0.835662347],
    [0.959183673469, 0.429229132, 0.524142742, 0.616933263, 0.696321879, 0.778062747, 0.854727889],
    [0.979591836735, 0.446286132, 0.541413258, 0.635998800, 0.715449305, 0.797716500, 0.875031479],
    [1.000000000000, 0.463500937, 0.560540726, 0.655714353, 0.735722253, 0.818639289, 0.895304432]
], dtype=np.float64)

xi_exp = _EXP[:, 0]
theta_exp = _EXP[:, 1:]

g_left_samples = theta_exp[0, :].copy()
g_right_samples = theta_exp[-1, :].copy()

def g_left(tau):
    return np.interp(tau, PROFILE_TAU, g_left_samples)

def g_right(tau):
    return np.interp(tau, PROFILE_TAU, g_right_samples)

plt.figure(figsize=(8, 5))
for j, tau in enumerate(PROFILE_TAU):
    plt.plot(xi_exp, theta_exp[:, j], "o-", ms=3, label=fr"$\tau={tau:g}$")
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\theta$")
plt.title("Normalized experimental brass profiles")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

# CELL 5
NX = 1000
DS = 1.0e-5
N_STEPS = int(round(1.0 / DS))

# Single QR regularization parameter.
GAMMA = 5.0e-3
SQRT_GAMMA = np.sqrt(GAMMA)

# beta(theta) = B0 + B1*theta
B0 = 0.287265051903114
B1 = 0.011656401384083

# Same broad divergence guard idea as the FreeFEM code.
SAFETY_MIN = -5.0
SAFETY_MAX = 5.0

SAVE_RESULTS = True
OUTPUT_DIR = Path("./zhang2016_qr_single_gamma")

print("NX =", NX)
print("nodes =", NX + 1)
print("DS =", DS)
print("N_STEPS =", N_STEPS)
print("GAMMA =", GAMMA)
print("sqrt(GAMMA) =", SQRT_GAMMA)
print("B0 =", B0)
print("B1 =", B1)

# CELL 7
def assemble_mass_matrix(nx):
    n = nx + 1
    h = 1.0 / nx

    rows, cols, vals = [], [], []
    local = (h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])

    for e in range(nx):
        nodes = [e, e + 1]
        for a in range(2):
            for b in range(2):
                rows.append(nodes[a])
                cols.append(nodes[b])
                vals.append(local[a, b])

    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def assemble_unit_stiffness_matrix(nx):
    n = nx + 1
    h = 1.0 / nx

    rows, cols, vals = [], [], []
    local = (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])

    for e in range(nx):
        nodes = [e, e + 1]
        for a in range(2):
            for b in range(2):
                rows.append(nodes[a])
                cols.append(nodes[b])
                vals.append(local[a, b])

    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def assemble_variable_stiffness(nx, nodal_coefficient):
    """
    Assemble integral a(x) u_x v_x dx for P1 functions.

    The lagged coefficient is represented nodally. Since u_x and v_x are
    constant on each element, using the arithmetic mean of the two nodal
    coefficient values gives the exact element integral of the P1 coefficient.
    """
    n = nx + 1
    h = 1.0 / nx
    a = np.asarray(nodal_coefficient, dtype=float)

    if a.shape != (n,):
        raise ValueError(f"Expected coefficient shape {(n,)}, got {a.shape}")

    rows, cols, vals = [], [], []

    for e in range(nx):
        abar = 0.5 * (a[e] + a[e + 1])
        local = (abar / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        nodes = [e, e + 1]

        for i_local in range(2):
            for j_local in range(2):
                rows.append(nodes[i_local])
                cols.append(nodes[j_local])
                vals.append(local[i_local, j_local])

    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


M = assemble_mass_matrix(NX)
K = assemble_unit_stiffness_matrix(NX)

xi_qr = np.linspace(0.0, 1.0, NX + 1)
interior = np.arange(1, NX)
boundary = np.array([0, NX])

print("M:", M.shape, "nnz =", M.nnz)
print("K:", K.shape, "nnz =", K.nnz)

# CELL 9
def beta(theta):
    return B0 + B1 * np.asarray(theta)


def qr_step(w_old, s_new):
    """
    One lagged-coefficient mixed quasi-reversibility step.

    Unknown vector:
      [w_new at interior nodes,
       q_new at all nodes]

    Only w receives essential Dirichlet boundary values.
    """
    tau_phys = 1.0 - s_new

    w_boundary = np.array(
        [g_left(tau_phys), g_right(tau_phys)],
        dtype=float,
    )

    a_old = beta(w_old)
    K_a = assemble_variable_stiffness(NX, a_old)

    A = M - DS * K_a
    B = -DS * SQRT_GAMMA * K
    C = SQRT_GAMMA * K
    D = M

    A_ii = A[interior][:, interior]
    A_ib = A[interior][:, boundary]
    B_iq = B[interior, :]

    C_qi = C[:, interior]
    C_qb = C[:, boundary]

    rhs_w = M @ w_old
    rhs1 = rhs_w[interior] - A_ib @ w_boundary
    rhs2 = -(C_qb @ w_boundary)

    block = bmat(
        [
            [A_ii, B_iq],
            [C_qi, D],
        ],
        format="csr",
    )

    rhs = np.concatenate([rhs1, rhs2])
    solution = spsolve(block, rhs)

    n_w_int = len(interior)

    w_new = np.empty(NX + 1, dtype=float)
    w_new[boundary] = w_boundary
    w_new[interior] = solution[:n_w_int]

    q_new = solution[n_w_int:]

    return w_new, q_new

# CELL 11
theta_terminal_exp = theta_exp[:, -1]

w0 = np.interp(xi_qr, xi_exp, theta_terminal_exp)
w0[0] = g_left(1.0)
w0[-1] = g_right(1.0)

plt.figure(figsize=(8, 5))
plt.plot(xi_exp, theta_terminal_exp, "ko", ms=4, label="terminal experiment")
plt.plot(xi_qr, w0, "-", lw=1.5, label="QR-mesh interpolation")
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\theta$")
plt.title(r"$w(\xi,0)=\theta(\xi,1)$")
plt.legend()
plt.tight_layout()
plt.show()

# CELL 13
RUN_QR = True

if not RUN_QR:
    raise RuntimeError("Set RUN_QR=True.")

w = w0.copy()
q = np.zeros_like(w)

qr_snapshots = {0.0: w.copy()}
target_s = [0.2, 0.4, 0.6, 0.8, 1.0]
target_steps = {int(round(s / DS)): s for s in target_s}

stable = True
failed_step = None

tic = time.perf_counter()

for step in range(1, N_STEPS + 1):
    s_new = step * DS
    w, q = qr_step(w, s_new)

    w_min = float(w.min())
    w_max = float(w.max())

    if w_min < SAFETY_MIN or w_max > SAFETY_MAX:
        stable = False
        failed_step = step
        print(
            "QR divergence guard triggered:",
            f"step={step}, s={s_new:.6f}, min={w_min:.6f}, max={w_max:.6f}"
        )
        break

    if step in target_steps:
        qr_snapshots[target_steps[step]] = w.copy()

    if step == 1 or step % 200 == 0 or step == N_STEPS:
        print(
            f"step={step:4d}/{N_STEPS}, "
            f"s={s_new:.4f}, tau={1.0-s_new:.4f}, "
            f"min(w)={w_min:.5f}, max(w)={w_max:.5f}"
        )

qr_seconds = time.perf_counter() - tic

print("stable =", stable)
print("QR runtime [s] =", qr_seconds)

if not stable:
    raise RuntimeError(
        f"QR became unstable at step {failed_step}. "
        "Try a larger GAMMA and/or smaller DS."
    )

theta0_qr = w.copy()

# CELL 15
def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a-b)**2)))


def relative_l2(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.linalg.norm(a-b) / np.linalg.norm(b))


theta0_true = theta_exp[:, 0]
theta0_qr_on_exp = np.interp(xi_exp, xi_qr, theta0_qr)

initial_metrics = {
    "gamma": float(GAMMA),
    "initial_rmse_theta": rmse(theta0_qr_on_exp, theta0_true),
    "initial_relative_l2": relative_l2(theta0_qr_on_exp, theta0_true),
    "initial_max_abs_error_theta": float(
        np.max(np.abs(theta0_qr_on_exp-theta0_true))
    ),
}

print(json.dumps(initial_metrics, indent=2))

plt.figure(figsize=(8, 5))
plt.plot(xi_exp, theta0_true, "ko", ms=4, label="experimental hidden IC")
plt.plot(xi_qr, theta0_qr, "-", lw=2, label=fr"QR, $\gamma={GAMMA:.1e}$")
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\theta$")
plt.title("Quasi-reversibility initial-field reconstruction")
plt.legend()
plt.tight_layout()
plt.show()

# CELL 17
reverse_rows = []

plt.figure(figsize=(9, 6))

for s, w_snap in sorted(qr_snapshots.items()):
    tau = 1.0 - s
    j = int(np.argmin(np.abs(PROFILE_TAU - tau)))

    pred = np.interp(xi_exp, xi_qr, w_snap)
    true = theta_exp[:, j]

    reverse_rows.append({
        "s": float(s),
        "tau": float(tau),
        "rmse_theta": rmse(pred, true),
        "relative_l2": relative_l2(pred, true),
        "max_abs_error_theta": float(np.max(np.abs(pred-true))),
    })

    plt.plot(xi_exp, true, "o", ms=3)
    plt.plot(xi_qr, w_snap, "-", lw=1.5, label=fr"QR $\tau={tau:g}$")

plt.xlabel(r"$\xi$")
plt.ylabel(r"$\theta$")
plt.title("QR reverse trajectory: experiment markers, QR lines")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

reverse_validation_df = pd.DataFrame(reverse_rows)
display(reverse_validation_df)

# CELL 19
def forward_step(u_old, tau_new):
    a_old = beta(u_old)
    K_a = assemble_variable_stiffness(NX, a_old)

    A = M + DS * K_a

    u_boundary = np.array(
        [g_left(tau_new), g_right(tau_new)],
        dtype=float,
    )

    A_ii = A[interior][:, interior]
    A_ib = A[interior][:, boundary]

    rhs = M @ u_old
    rhs_i = rhs[interior] - A_ib @ u_boundary

    u_new = np.empty_like(u_old)
    u_new[boundary] = u_boundary
    u_new[interior] = spsolve(A_ii, rhs_i)

    return u_new


u = theta0_qr.copy()

forward_snapshots = {0.0: u.copy()}
target_tau = [0.2, 0.4, 0.6, 0.8, 1.0]
target_steps = {int(round(tau / DS)): tau for tau in target_tau}

tic = time.perf_counter()

for step in range(1, N_STEPS + 1):
    tau_new = step * DS
    u = forward_step(u, tau_new)

    if step in target_steps:
        forward_snapshots[target_steps[step]] = u.copy()

    if step % 400 == 0 or step == N_STEPS:
        print(
            f"forward step={step:4d}/{N_STEPS}, "
            f"tau={tau_new:.4f}, min={u.min():.5f}, max={u.max():.5f}"
        )

forward_seconds = time.perf_counter() - tic
print("forward runtime [s] =", forward_seconds)

# CELL 21
forward_rows = []

plt.figure(figsize=(9, 6))

for tau in PROFILE_TAU:
    u_snap = forward_snapshots[float(tau)]
    pred = np.interp(xi_exp, xi_qr, u_snap)

    j = int(np.argmin(np.abs(PROFILE_TAU - tau)))
    true = theta_exp[:, j]

    forward_rows.append({
        "tau": float(tau),
        "rmse_theta": rmse(pred, true),
        "relative_l2": relative_l2(pred, true),
        "max_abs_error_theta": float(np.max(np.abs(pred-true))),
    })

    plt.plot(xi_exp, true, "o", ms=3)
    plt.plot(xi_qr, u_snap, "-", lw=1.5, label=fr"forward $\tau={tau:g}$")

plt.xlabel(r"$\xi$")
plt.ylabel(r"$\theta$")
plt.title(
    "Forward replay from QR-recovered IC\n"
    "markers: experiment, lines: deterministic forward solve"
)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

forward_metrics_df = pd.DataFrame(forward_rows)
display(forward_metrics_df)

# CELL 23
terminal_pred = np.interp(
    xi_exp,
    xi_qr,
    forward_snapshots[1.0],
)
terminal_true = theta_exp[:, -1]

terminal_metrics = {
    "gamma": float(GAMMA),
    "terminal_rmse_theta": rmse(terminal_pred, terminal_true),
    "terminal_relative_l2": relative_l2(terminal_pred, terminal_true),
    "terminal_max_abs_error_theta": float(
        np.max(np.abs(terminal_pred-terminal_true))
    ),
}

print(json.dumps(terminal_metrics, indent=2))

plt.figure(figsize=(8, 5))
plt.plot(xi_exp, terminal_true, "ko", ms=4, label="experimental terminal")
plt.plot(xi_exp, terminal_pred, "-", lw=2, label="forward replay from QR IC")
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\theta$")
plt.title(
    "Terminal replay check\n"
    f"relative L2={terminal_metrics['terminal_relative_l2']:.3e}"
)
plt.legend()
plt.tight_layout()
plt.show()

# CELL 25
if SAVE_RESULTS:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gamma_tag = f"{GAMMA:.0e}".replace("-", "m")

    pd.DataFrame({
        "xi": xi_qr,
        "theta0_qr": theta0_qr,
    }).to_csv(
        OUTPUT_DIR / f"qr_gamma_{gamma_tag}_recovered_initial.csv",
        index=False,
    )

    reverse_validation_df.to_csv(
        OUTPUT_DIR / f"qr_gamma_{gamma_tag}_reverse_validation.csv",
        index=False,
    )

    forward_metrics_df.to_csv(
        OUTPUT_DIR / f"qr_gamma_{gamma_tag}_forward_metrics.csv",
        index=False,
    )

    forward_out = pd.DataFrame({"xi": xi_qr})
    for tau in PROFILE_TAU:
        forward_out[f"theta_forward_tau_{tau:g}"] = forward_snapshots[float(tau)]
    forward_out.to_csv(
        OUTPUT_DIR / f"qr_gamma_{gamma_tag}_forward_profiles.csv",
        index=False,
    )

    summary = {
        "gamma": float(GAMMA),
        "sqrt_gamma": float(SQRT_GAMMA),
        "nx": int(NX),
        "ds": float(DS),
        "n_steps": int(N_STEPS),
        "qr_runtime_seconds": float(qr_seconds),
        "forward_runtime_seconds": float(forward_seconds),
        **initial_metrics,
        **terminal_metrics,
    }

    with (OUTPUT_DIR / f"qr_gamma_{gamma_tag}_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("saved to:", OUTPUT_DIR.resolve())