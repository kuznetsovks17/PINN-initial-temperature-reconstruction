# CELL 1
# Uncomment only if your environment is missing the dependencies:
# %pip install tensorflow tf-fourier-features scikit-optimize pandas matplotlib

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from skopt.sampler import Sobol
from tf_fourier_features import FourierFeatureProjection

plt.rcParams.update({"font.size": 12})

print("TensorFlow:", tf.__version__)
print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

# CELL 3
PROFILE_TAU = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)

# columns: sensor_index, xi, theta(tau=0), ..., theta(tau=1)
_NORMALIZED_ROWS = np.array([
    [0, 0.000000000000, 0.167843653, 0.199499979, 0.235610437, 0.275561737, 0.326274347, 0.376139326],
    [1, 0.020408163265, 0.164309063, 0.198577611, 0.238609484, 0.275561737, 0.325045547, 0.377290511],
    [2, 0.040816326531, 0.166305342, 0.200113005, 0.239222163, 0.275561737, 0.329883163, 0.380363874],
    [3, 0.061224489796, 0.167843653, 0.203340911, 0.242756668, 0.278560784, 0.331650668, 0.382669921],
    [4, 0.081632653061, 0.169994732, 0.203340911, 0.242756668, 0.283324084, 0.332882211, 0.385127174],
    [5, 0.102040816327, 0.169994732, 0.203953937, 0.245749716, 0.283324084, 0.334111005, 0.386820195],
    [6, 0.122448979592, 0.169994732, 0.206950068, 0.246904200, 0.285165563, 0.335185174, 0.388662016],
    [7, 0.142857142857, 0.173835658, 0.206027358, 0.247516884, 0.286858584, 0.338874305, 0.394041084],
    [8, 0.163265306122, 0.173835658, 0.208410805, 0.248136079, 0.287474353, 0.338261621, 0.394656858],
    [9, 0.183673469388, 0.174754942, 0.207565837, 0.252283611, 0.291086600, 0.343021837, 0.398191700],
    [10, 0.204081632653, 0.176757479, 0.211100511, 0.255895511, 0.297613979, 0.348401247, 0.403567684],
    [11, 0.224489795918, 0.177370337, 0.215018674, 0.258888389, 0.298845863, 0.352551516, 0.407718300],
    [12, 0.244897959184, 0.182749574, 0.216479579, 0.262422889, 0.303606074, 0.357314816, 0.412481600],
    [13, 0.265306122449, 0.188435242, 0.220088400, 0.263654774, 0.307217979, 0.360388179, 0.416167642],
    [14, 0.285714285714, 0.186284079, 0.221855905, 0.266034795, 0.307759774, 0.365151821, 0.422623968],
    [15, 0.306122448980, 0.184900653, 0.225467811, 0.270801179, 0.316744574, 0.369918205, 0.428003379],
    [16, 0.326530612245, 0.191199089, 0.231153479, 0.273181374, 0.322123984, 0.376368358, 0.432766679],
    [17, 0.346938775510, 0.191969747, 0.233227584, 0.279789579, 0.326890374, 0.383591989, 0.442983700],
    [18, 0.367346938776, 0.196807358, 0.234994405, 0.286320216, 0.331727816, 0.390200200, 0.448285368],
    [19, 0.387755102041, 0.196807358, 0.240683163, 0.291086600, 0.338797163, 0.397266121, 0.456738826],
    [20, 0.408163265306, 0.202109200, 0.244830689, 0.297078695, 0.345408111, 0.405102779, 0.465035947],
    [21, 0.428571428571, 0.206411358, 0.245443374, 0.302380363, 0.352012889, 0.411097958, 0.472256837],
    [22, 0.448979591837, 0.214712416, 0.253821663, 0.307759774, 0.359082242, 0.420627989, 0.483015316],
    [23, 0.469387755102, 0.215325184, 0.260042784, 0.316209121, 0.366918900, 0.430154584, 0.494844537],
    [24, 0.489795918367, 0.219472626, 0.266037963, 0.321510958, 0.375832816, 0.440293868, 0.503309695],
    [25, 0.510204081633, 0.223007132, 0.272262005, 0.331115132, 0.383592247, 0.448595100, 0.514052616],
    [26, 0.530612244898, 0.225777321, 0.277944758, 0.338797163, 0.393731532, 0.459350489, 0.524204458],
    [27, 0.551020408163, 0.229305653, 0.283940111, 0.346021137, 0.403335700, 0.471337426, 0.535408779],
    [28, 0.571428571429, 0.233765695, 0.291699368, 0.353702832, 0.411023905, 0.479019458, 0.547386584],
    [29, 0.591836734694, 0.242756926, 0.301532226, 0.363845537, 0.421776211, 0.491003647, 0.559395800],
    [30, 0.612244897959, 0.243598811, 0.305989268, 0.372298311, 0.433147374, 0.503619379, 0.571218847],
    [31, 0.632653061224, 0.252667353, 0.315825547, 0.382437595, 0.445060511, 0.515445511, 0.585053863],
    [32, 0.653061224490, 0.258046589, 0.324816521, 0.394350726, 0.456354274, 0.527423405, 0.598114963],
    [33, 0.673469387755, 0.268421242, 0.333188811, 0.405028721, 0.468954232, 0.542310684, 0.613002374],
    [34, 0.693877551020, 0.278325500, 0.344253716, 0.417016000, 0.482015679, 0.556610184, 0.627363542],
    [35, 0.714285714286, 0.286320132, 0.354625326, 0.430689784, 0.495856774, 0.570909337, 0.642900963],
    [36, 0.734693877551, 0.294079474, 0.362694226, 0.442677058, 0.508914747, 0.585951458, 0.657818974],
    [37, 0.755102040816, 0.301532395, 0.376445758, 0.456354274, 0.525040426, 0.601488668, 0.674470632],
    [38, 0.775510204082, 0.314135611, 0.389584084, 0.469570347, 0.539958437, 0.617459242, 0.691245895],
    [39, 0.795918367347, 0.325658574, 0.403261300, 0.485089037, 0.555465000, 0.633615653, 0.707990384],
    [40, 0.816326530612, 0.335262574, 0.415245489, 0.499388195, 0.571002205, 0.649741068, 0.725199016],
    [41, 0.836734693878, 0.347175537, 0.428926474, 0.514300332, 0.587746737, 0.667104458, 0.743181384],
    [42, 0.857142857143, 0.358853258, 0.443599684, 0.529187611, 0.604955368, 0.683693889, 0.760451905],
    [43, 0.877551020408, 0.370530932, 0.457273642, 0.544725163, 0.619935263, 0.700469153, 0.777134295],
    [44, 0.897959183673, 0.384205016, 0.473717789, 0.562614532, 0.639063032, 0.720122905, 0.797407158],
    [45, 0.918367346939, 0.399417284, 0.491619758, 0.579916121, 0.657509716, 0.739962395, 0.816534711],
    [46, 0.938775510204, 0.415248663, 0.506841284, 0.599012474, 0.676018284, 0.758997295, 0.835662347],
    [47, 0.959183673469, 0.429229132, 0.524142742, 0.616933263, 0.696321879, 0.778062747, 0.854727889],
    [48, 0.979591836735, 0.446286132, 0.541413258, 0.635998800, 0.715449305, 0.797716500, 0.875031479],
    [49, 1.000000000000, 0.463500937, 0.560540726, 0.655714353, 0.735722253, 0.818639289, 0.895304432]
], dtype=np.float64)

sensor_index = _NORMALIZED_ROWS[:, 0].astype(int)
xi_exp = _NORMALIZED_ROWS[:, 1:2]
theta_exp = _NORMALIZED_ROWS[:, 2:]

assert theta_exp.shape == (50, 6)
assert np.allclose(xi_exp[[0, -1], 0], [0.0, 1.0])

normalized_frame = pd.DataFrame({
    "sensor": sensor_index,
    "xi": xi_exp[:, 0],
    **{f"theta_tau_{tt:g}": theta_exp[:, j] for j, tt in enumerate(PROFILE_TAU)},
})
display(normalized_frame.head())
display(normalized_frame.tail())

plt.figure(figsize=(8, 5))
for j, tt in enumerate(PROFILE_TAU):
    plt.plot(xi_exp[:, 0], theta_exp[:, j], "o-", ms=3, label=fr"$\tau={tt:g}$")
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\theta$")
plt.title("Normalized experimental brass profiles")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()


# CELL 5
@dataclass
class Config:
    seed: int = 2026

    # Fully normalized domain
    xi_min: float = 0.0
    xi_max: float = 1.0
    tau_min: float = 0.0
    tau_max: float = 1.0

    # Dimensionless PDE coefficients only
    B0: float = 0.287265051903114
    B1: float = 0.011656401384083

    # Full-batch fixed point sets
    N_r: int = 12000
    N_b: int = 200

    # EPINN architecture retained from the supplied notebook
    dtype: str = "float32"
    num_shared_layers: int = 2
    shared_dim: int = 60
    num_branch_layers: int = 1
    branch_dim: int = 60
    fourier_projection_dimension: int = 16
    encoding_mode: str = "anisotropic"
    adaptive_weights: bool = True

    learning_rate_model: float = 2e-4
    learning_rate_weights: float = 5e-4
    adam_beta_1: float = 0.99
    eta_min: float = 1e-2
    eta_max: float = 1e2

    epochs: int = 60000
    print_every: int = 500
    history_every: int = 10
    save_dir: str = "./zhang2016_brass_epinn_normalized"

    @property
    def fourier_scales(self):
        return (1.0, 1.0)

cfg = Config()
np.random.seed(cfg.seed)
tf.keras.utils.set_random_seed(cfg.seed)
print(json.dumps(asdict(cfg), indent=2))
print("Dimensionless AFFE scales [xi, tau]:", cfg.fourier_scales)


# CELL 7
class NormalizedExperimentalData:
    def __init__(self, cfg, xi_exp, theta_exp, profile_tau):
        self.cfg = cfg
        self.dtype = tf.as_dtype(cfg.dtype)
        self.xi_exp = np.asarray(xi_exp, dtype=np.float64)
        self.theta_exp = np.asarray(theta_exp, dtype=np.float64)
        self.profile_tau = np.asarray(profile_tau, dtype=np.float64)

        # Terminal field. Endpoints are handled by the boundary loss.
        self.X_terminal_np = np.column_stack([
            self.xi_exp[1:-1, 0],
            np.ones(len(self.xi_exp)-2),
        ])
        self.theta_terminal_np = self.theta_exp[1:-1, -1:].copy()

        # Hidden retrospective truth.
        self.X_initial_truth_np = np.column_stack([
            self.xi_exp[:, 0],
            np.zeros(len(self.xi_exp)),
        ])
        self.theta_initial_truth_np = self.theta_exp[:, 0:1].copy()

        # Intermediate profiles: validation only.
        Xv, Yv = [], []
        for j, tt in enumerate(self.profile_tau[1:-1], start=1):
            Xv.append(np.column_stack([
                self.xi_exp[:, 0],
                np.full(len(self.xi_exp), tt),
            ]))
            Yv.append(self.theta_exp[:, j:j+1])
        self.X_validation_np = np.vstack(Xv)
        self.theta_validation_np = np.vstack(Yv)

        # Normalized measured endpoint values.
        self.left_measured = self.theta_exp[0, :].copy()
        self.right_measured = self.theta_exp[-1, :].copy()

        n_left = cfg.N_b // 2
        n_right = cfg.N_b - n_left
        tau_left = np.linspace(0.0, 1.0, n_left)
        tau_right = np.linspace(0.0, 1.0, n_right)

        theta_left = np.interp(tau_left, self.profile_tau, self.left_measured)
        theta_right = np.interp(tau_right, self.profile_tau, self.right_measured)

        X_left = np.column_stack([np.zeros(n_left), tau_left])
        X_right = np.column_stack([np.ones(n_right), tau_right])
        self.X_boundary_np = np.vstack([X_left, X_right])
        self.theta_boundary_np = np.concatenate([theta_left, theta_right]).reshape(-1, 1)

        # Deterministic Sobol collocation in the normalized open unit square.
        sampler = Sobol(skip=0, randomize=False)
        unit = np.asarray(
            sampler.generate([(0.0, 1.0), (0.0, 1.0)], cfg.N_r + 2)[2:],
            dtype=np.float64,
        )
        inside = (
            (unit[:, 0] > 0.0) & (unit[:, 0] < 1.0)
            & (unit[:, 1] > 0.0) & (unit[:, 1] < 1.0)
        )
        self.X_residual_np = unit[inside][:cfg.N_r]
        if len(self.X_residual_np) < cfg.N_r:
            raise RuntimeError("Not enough Sobol points after filtering.")

        # Fixed full-batch tensors.
        self.X_r = tf.constant(self.X_residual_np, dtype=self.dtype)
        self.X_b = tf.constant(self.X_boundary_np, dtype=self.dtype)
        self.theta_b = tf.constant(self.theta_boundary_np, dtype=self.dtype)
        self.X_T = tf.constant(self.X_terminal_np, dtype=self.dtype)
        self.theta_T = tf.constant(self.theta_terminal_np, dtype=self.dtype)

        print("PDE points:", len(self.X_residual_np))
        print("Boundary points:", len(self.X_boundary_np))
        print("Terminal interior points:", len(self.X_terminal_np))
        print("Hidden initial points:", len(self.X_initial_truth_np))
        print("Withheld validation points:", len(self.X_validation_np))


data = NormalizedExperimentalData(cfg, xi_exp, theta_exp, PROFILE_TAU)

boundary_table = pd.DataFrame({
    "tau": PROFILE_TAU,
    "g_left": data.left_measured,
    "g_right": data.right_measured,
})
display(boundary_table)

plt.figure(figsize=(8, 4))
plt.plot(PROFILE_TAU, data.left_measured, "o-", label=r"$g_L(\tau)$")
plt.plot(PROFILE_TAU, data.right_measured, "o-", label=r"$g_R(\tau)$")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$\theta$")
plt.title("Normalized Dirichlet boundary histories")
plt.legend()
plt.tight_layout()
plt.show()


# CELL 9
class ConductiveTreePINN(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dtype = tf.as_dtype(cfg.dtype)

        self.coordinate_projections = []
        if cfg.encoding_mode == "none":
            self.fourier_scales = ()
        elif cfg.encoding_mode == "anisotropic":
            self.fourier_scales = (1.0, 1.0)
            self.coordinate_projections = [
                FourierFeatureProjection(
                    gaussian_projection=cfg.fourier_projection_dimension,
                    gaussian_scale=1.0,
                )
                for _ in range(2)
            ]
        else:
            raise ValueError("Use encoding_mode='none' or 'anisotropic'.")

        self.shared_layers = [
            tf.keras.layers.Dense(cfg.shared_dim, activation="tanh")
            for _ in range(cfg.num_shared_layers)
        ]
        self.temperature_branch = [
            tf.keras.layers.Dense(cfg.branch_dim, activation="tanh")
            for _ in range(cfg.num_branch_layers)
        ]
        self.temperature_output = tf.keras.layers.Dense(
            1,
            activation="linear",
            bias_initializer="zeros",
            kernel_initializer="zeros",
            name="theta",
        )
        self.output_shift = tf.Variable(
            0.0, trainable=False, dtype=dtype, name="theta_shift"
        )

    def set_output_shift(self, value):
        self.output_shift.assign(tf.cast(value, self.output_shift.dtype))

    def project_inputs(self, X):
        if self.cfg.encoding_mode == "none":
            return X
        return tf.concat(
            [proj(X[:, j:j+1]) for j, proj in enumerate(self.coordinate_projections)],
            axis=1,
        )

    def call(self, X, training=False):
        z = self.project_inputs(X)
        for layer in self.shared_layers:
            z = layer(z, training=training)
        for layer in self.temperature_branch:
            z = layer(z, training=training)
        return self.temperature_output(z, training=training) + self.output_shift


model = ConductiveTreePINN(cfg)
_ = model(tf.zeros((1, 2), dtype=tf.as_dtype(cfg.dtype)), training=False)

# Shift uses only normalized quantities available in training.
known_theta_mean = np.mean(np.concatenate([
    data.theta_boundary_np.reshape(-1),
    data.theta_terminal_np.reshape(-1),
]))
model.set_output_shift(known_theta_mean)

print("Normalized output shift:", float(model.output_shift.numpy()))
print("Fourier scales:", model.fourier_scales)
model.summary()


# CELL 11
# class AdaptiveEta(tf.Module):
#     def __init__(self, names, dtype):
#         super().__init__()
#         self.names = list(names)
#         for name in self.names:
#             setattr(self, name, tf.Variable(1.0, trainable=True, dtype=dtype, name=f"eta_{name}"))

#     def variables_list(self):
#         return [getattr(self, name) for name in self.names]

#     def as_dict(self):
#         return {name: getattr(self, name) for name in self.names}
class AdaptiveEta(tf.Module):
    def __init__(self, sizes, dtype):
        super().__init__()

        # Example:
        # sizes = {
        #     "pde": N_r,
        #     "boundary": N_b,
        #     "terminal": N_T,
        # }

        self.names = list(sizes.keys())

        for name, n_points in sizes.items():
            setattr(
                self,
                name,
                tf.Variable(
                    tf.ones((n_points, 1), dtype=dtype),
                    trainable=True,
                    name=f"eta_{name}",
                )
            )

    def variables_list(self):
        return [getattr(self, name) for name in self.names]

    def as_dict(self):
        return {name: getattr(self, name) for name in self.names}

class Trainer:
    def __init__(self, cfg, data, model):
        self.cfg = cfg
        self.data = data
        self.model = model
        # self.eta = AdaptiveEta(["pde", "boundary", "terminal"], tf.as_dtype(cfg.dtype))
        eta_sizes = {
            "pde": int(data.X_r.shape[0]),
            "boundary": int(data.X_b.shape[0]),
            "terminal": int(data.X_T.shape[0]),
        }
        
        self.eta = AdaptiveEta(
            eta_sizes,
            tf.as_dtype(cfg.dtype)
        )
        self.model_optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.learning_rate_model,
            beta_1=cfg.adam_beta_1,
        )
        self.eta_optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.learning_rate_weights,
            beta_1=cfg.adam_beta_1,
        )

        self.history = {
            "epoch": [], "total": [], "pde": [], "boundary": [], "terminal": [],
            "eta_pde": [], "eta_boundary": [], "eta_terminal": [],
        }

    @staticmethod
    def mse(r):
        return tf.reduce_mean(tf.square(r))

    def beta(self, theta):
        return tf.cast(self.cfg.B0, theta.dtype) + tf.cast(self.cfg.B1, theta.dtype) * theta

    def pde_residual(self, X, training=True):
        # r = theta_tau - (B0+B1*theta) theta_xixi - B1 theta_xi^2
        with tf.GradientTape(persistent=True) as outer:
            outer.watch(X)
            with tf.GradientTape() as inner:
                inner.watch(X)
                theta = self.model(X, training=training)
            grad = inner.gradient(theta, X)
            theta_xi = grad[:, 0:1]
            theta_tau = grad[:, 1:2]

        second = outer.gradient(
            theta_xi, X,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        del outer
        theta_xixi = second[:, 0:1]

        return (
            theta_tau
            - self.beta(theta) * theta_xixi
            - tf.cast(self.cfg.B1, theta.dtype) * tf.square(theta_xi)
        )

    # def raw_losses(self):
    #     r = self.pde_residual(self.data.X_r, training=True)
    #     theta_b_pred = self.model(self.data.X_b, training=True)
    #     theta_T_pred = self.model(self.data.X_T, training=True)
    #     return {
    #         "pde": self.mse(r),
    #         "boundary": self.mse(theta_b_pred - self.data.theta_b),
    #         "terminal": self.mse(theta_T_pred - self.data.theta_T),
    #     }
    def residual_terms(self):
        r_pde = self.pde_residual(
            self.data.X_r,
            training=True
        )
    
        r_boundary = (
            self.model(self.data.X_b, training=True)
            - self.data.theta_b
        )
    
        r_terminal = (
            self.model(self.data.X_T, training=True)
            - self.data.theta_T
        )
    
        return {
            "pde": r_pde,
            "boundary": r_boundary,
            "terminal": r_terminal,
        }
    def objective(self, residuals):
        eta = self.eta.as_dict()
    
        loss_pde = tf.reduce_mean(
            tf.square(eta["pde"]) *
            tf.square(residuals["pde"])
        )
    
        loss_boundary = tf.reduce_mean(
            tf.square(eta["boundary"]) *
            tf.square(residuals["boundary"])
        )
    
        loss_terminal = tf.reduce_mean(
            tf.square(eta["terminal"]) *
            tf.square(residuals["terminal"])
        )
    
        total = (
            loss_pde
            + loss_boundary
            + loss_terminal
        )
    
        return total        
    # def objective(self, losses):
    #     if not self.cfg.adaptive_weights:
    #         return losses["pde"] + losses["boundary"] + losses["terminal"]
    #     eta = self.eta.as_dict()
    #     return (
    #         tf.square(eta["pde"]) * losses["pde"]
    #         + tf.square(eta["boundary"]) * losses["boundary"]
    #         + tf.square(eta["terminal"]) * losses["terminal"]
    #     )

    @tf.function(reduce_retracing=True)
    def train_step(self):
        # Full-batch training: every fixed point is used every epoch.
        with tf.GradientTape(persistent=True) as tape:
            losses = self.raw_losses()
            total = self.objective(losses)

        model_grads = tape.gradient(total, self.model.trainable_variables)
        self.model_optimizer.apply_gradients([
            (g, v) for g, v in zip(model_grads, self.model.trainable_variables) if g is not None
        ])

        if self.cfg.adaptive_weights:
            eta_vars = self.eta.variables_list()
            eta_grads = tape.gradient(total, eta_vars)
            eta_pairs = [(-g, v) for g, v in zip(eta_grads, eta_vars) if g is not None]
            if eta_pairs:
                self.eta_optimizer.apply_gradients(eta_pairs)
            for v in eta_vars:
                v.assign(tf.clip_by_value(
                    v,
                    tf.cast(self.cfg.eta_min, v.dtype),
                    tf.cast(self.cfg.eta_max, v.dtype),
                ))

        del tape
        return total, losses

    def _record(self, epoch, total, losses):
        self.history["epoch"].append(int(epoch))
        self.history["total"].append(float(total.numpy()))
        for key in ("pde", "boundary", "terminal"):
            self.history[key].append(float(losses[key].numpy()))
        eta = self.eta.as_dict()
        self.history["eta_pde"].append(float(eta["pde"].numpy()))
        self.history["eta_boundary"].append(float(eta["boundary"].numpy()))
        self.history["eta_terminal"].append(float(eta["terminal"].numpy()))

    def fit(self):
        wall_start = time.perf_counter()
        interval_start = wall_start
        for epoch in range(self.cfg.epochs):
            total, losses = self.train_step()
            if epoch % self.cfg.history_every == 0:
                self._record(epoch, total, losses)
            if epoch % self.cfg.print_every == 0 or epoch == self.cfg.epochs - 1:
                eta = self.eta.as_dict()
                elapsed = time.perf_counter() - interval_start
                print(
                    f"epoch={epoch:7d} total={float(total.numpy()):.3e} "
                    f"pde={float(losses['pde'].numpy()):.3e} "
                    f"bc={float(losses['boundary'].numpy()):.3e} "
                    f"terminal={float(losses['terminal'].numpy()):.3e} "
                    f"eta_r={float(eta['pde'].numpy()):.3e} "
                    f"eta_b={float(eta['boundary'].numpy()):.3e} "
                    f"eta_T={float(eta['terminal'].numpy()):.3e} dt={elapsed:.1f}s"
                )
                interval_start = time.perf_counter()
        self.wall_time_seconds = time.perf_counter() - wall_start
        return self.history


trainer = Trainer(cfg, data, model)


# CELL 13
RUN_TRAINING = True

if RUN_TRAINING:
    history = trainer.fit()
else:
    history = trainer.history
    print("Training skipped.")

# CELL 15
def predict_numpy(X):
    X_tf = tf.convert_to_tensor(X, dtype=tf.as_dtype(cfg.dtype))
    return model(X_tf, training=False).numpy().astype(np.float64)


def rmse(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true)**2)))


def relative_l2(pred, true):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


theta0_pred = predict_numpy(data.X_initial_truth_np)
theta0_true = data.theta_initial_truth_np

metrics = {
    "initial_rmse_theta": rmse(theta0_pred, theta0_true),
    "initial_relative_l2": relative_l2(theta0_pred, theta0_true),
    "initial_max_abs_error_theta": float(np.max(np.abs(theta0_pred - theta0_true))),
}

metrics["terminal_rmse_theta"] = rmse(
    predict_numpy(data.X_terminal_np), data.theta_terminal_np
)
metrics["boundary_rmse_theta"] = rmse(
    predict_numpy(data.X_boundary_np), data.theta_boundary_np
)

validation_rows = []
for j, tt in enumerate(PROFILE_TAU[1:-1], start=1):
    Xj = np.column_stack([xi_exp[:, 0], np.full(len(xi_exp), tt)])
    pred = predict_numpy(Xj)
    true = theta_exp[:, j:j+1]
    validation_rows.append({
        "tau": float(tt),
        "rmse_theta": rmse(pred, true),
        "relative_l2": relative_l2(pred, true),
        "max_abs_error_theta": float(np.max(np.abs(pred - true))),
    })

validation_metrics = pd.DataFrame(validation_rows)
print(json.dumps(metrics, indent=2))
display(validation_metrics)


# CELL 16
plt.figure(figsize=(8, 5))
plt.plot(xi_exp[:, 0], theta0_true[:, 0], "ko", ms=4, label=r"withheld $\theta(\xi,0)$")
plt.plot(xi_exp[:, 0], theta0_pred[:, 0], "-", lw=2, label=r"reconstructed $\widehat\theta(\xi,0)$")
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\theta$")
plt.title(
    "Normalized retrospective initial-field reconstruction\n"
    f"RMSE={metrics['initial_rmse_theta']:.3e}, "
    f"max |e|={metrics['initial_max_abs_error_theta']:.3e}"
)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))
for j, tt in enumerate(PROFILE_TAU):
    Xj = np.column_stack([xi_exp[:, 0], np.full(len(xi_exp), tt)])
    pred = predict_numpy(Xj)
    plt.plot(xi_exp[:, 0], theta_exp[:, j], "o", ms=3)
    plt.plot(xi_exp[:, 0], pred[:, 0], "-", lw=1.5, label=fr"$\tau={tt:g}$")
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\theta$")
plt.title("Normalized experimental profiles (markers) and EPINN solution (lines)")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

if len(history["epoch"]) > 0:
    h = pd.DataFrame(history)
    plt.figure(figsize=(8, 5))
    plt.semilogy(h["epoch"], h["pde"], label="PDE")
    plt.semilogy(h["epoch"], h["boundary"], label="boundary")
    plt.semilogy(h["epoch"], h["terminal"], label="terminal")
    plt.xlabel("epoch")
    plt.ylabel("dimensionless MSE")
    plt.title("Normalized training losses")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.semilogy(h["epoch"], h["eta_pde"], label=r"$\eta_r$")
    plt.semilogy(h["epoch"], h["eta_boundary"], label=r"$\eta_b$")
    plt.semilogy(h["epoch"], h["eta_terminal"], label=r"$\eta_T$")
    plt.xlabel("epoch")
    plt.ylabel("adaptive coefficient")
    plt.title("Adaptive loss coefficients")
    plt.legend()
    plt.tight_layout()
    plt.show()


# CELL 18
SAVE_RESULTS = True

if SAVE_RESULTS and RUN_TRAINING:
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.save_weights(save_dir / "model.weights.h5")
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)
    validation_metrics.to_csv(save_dir / "withheld_intermediate_metrics_normalized.csv", index=False)

    reconstruction = pd.DataFrame({
        "xi": xi_exp[:, 0],
        "theta_initial_experimental": theta0_true[:, 0],
        "theta_initial_reconstructed": theta0_pred[:, 0],
        "error_theta": theta0_pred[:, 0] - theta0_true[:, 0],
    })
    reconstruction.to_csv(
        save_dir / "retrospective_initial_reconstruction_normalized.csv", index=False
    )

    with (save_dir / "metrics_normalized.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (save_dir / "config_dimensionless.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("Saved normalized results to:", save_dir.resolve())
