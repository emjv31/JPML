"""
Execution pipeline.

1. Run Example B of the paper of Li: linear Gaussian model (dimension=64).
2. Run Example C of the paper of Li: skew-t Poisson model for selected dimensions (d=144, d=400).
3. Run EDH/LEDH diagnostics.
4. Run beta schedule comparison for straight/geometric/optimal (Dai-based) beta.
5. Run Hu matrix/scalar PFF for different m1 values (sparse and non-sparse scenarios) and dimensions.
6. Run OT tuning and gradient comparison for resampling methods.
7. Save result dictionaries, summary CSVs, and figures.

Edit the CONFIG block below for final/quick runs.
"""

import os
import pickle
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Imports from project files
# ------------------------------------------------------------
from simulator_model_comps import *
from replicate_Li_filters import *
from Hu_filters_utils import *
from metric_utils import *
from differentiablePF_resampling import *

# Dai: import only what is needed for the construction of the optimal beta (beta*)
from replicate_Dai import compute_H, compute_J, optimize_beta, compute_derivative_beta


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "submission_results"

# Set to False for submission runs.
QUICK_RUN = True  # set False only for final long run

SEED = 123

# Example B: Gaussian
EXAMPLE_B_D = 64
EXAMPLE_B_T = 10
EXAMPLE_B_NP = 20 if QUICK_RUN else 200
EXAMPLE_B_N_MC = 3 if QUICK_RUN else 100

# Example C: Poisson/skew-t
EXAMPLE_C_DIMS = [20] if QUICK_RUN else [144, 400]
EXAMPLE_C_T = 10
EXAMPLE_C_NP = 20 if QUICK_RUN else 200
EXAMPLE_C_N_MC = 3 if QUICK_RUN else 100

# EDH/LEDH diagnostics pipeline
DIAGNOSTIC_D = 10
DIAGNOSTIC_NP = 10 if QUICK_RUN else 200
DIAGNOSTIC_N_MC_ACCURACY = 3 if QUICK_RUN else 100
DIAGNOSTIC_N_MC_DIAGNOSTICS = 2 if QUICK_RUN else 100

# Hu comparison for sparsity through m1
HU_D = 10 if QUICK_RUN else 144
HU_NP = 10 if QUICK_RUN else 200
HU_N_MC = 2 if QUICK_RUN else 100
HU_N_STEPS = 5 if QUICK_RUN else 10
HU_EPS = 0.5
M1_VALUES = [1.0, 0.1, 0.05]   # 0.05 = sparse case from your notebook
HU_DIMS = [10] if QUICK_RUN else [10, 144, 400]
M2_VALUE = 1.0 / 3.0

# Dai beta comparison
BETA_COMPARISON_DIMS = [10] if QUICK_RUN else [10, 144, 400]
BETA_COMPARISON_M1_VALUES = [0.05, 1.0]

# OT tuning and gradient comparison
RUN_OT_TUNING = True
RUN_GRADIENT_COMPARISON = True
GRADIENT_N_RUNS = 5 if QUICK_RUN else 50

# Common parameters
ALPHA = 0.9
SIGMA_Z = 1.0
GAMMA = 0.3
NU = 5
BETA_Q = 1.2
BETA_N_LAMBDA = 29

# Heavy baselines from notebook such as PF with Np=10000/100000 are off by default.
INCLUDE_HEAVY_BASELINES = False


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def tensor_to_numpy_safe(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return x


def save_summary_csv(results_dict, filename):
    try:
        df = sims_to_df(results_dict)
        df.to_csv(filename)
        return df
    except Exception as exc:
        print(f"[WARN] Could not save summary CSV {filename}: {exc}")
        return None


def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)


def make_gaussian_wrappers_local(alpha, F_tf, H_tf, L_tf, sigma_z):
    """Local version wrappers."""
    @tf.function
    def prop_fn(x):
        return prop_linear_gaussian(x, F_tf, L_tf)

    @tf.function
    def h_func(x):
        return tf.matmul(x, tf.cast(H_tf, x.dtype), transpose_b=True)

    @tf.function
    def llk_fn(particles, y):
        return loglik_gaussian(particles, y, sigma_z)

    @tf.function
    def H_jac_fn(x, t):
        del t
        return tf.eye(tf.shape(x)[0], dtype=x.dtype)

    @tf.function
    def F_jac_fn(x, t):
        del t
        return tf.eye(tf.shape(x)[0], dtype=x.dtype) * tf.cast(alpha, x.dtype)

    return {
        "prop_fn": prop_fn,
        "h_func": h_func,
        "llk_fn": llk_fn,
        "H_jac_fn": H_jac_fn,
        "F_jac_fn": F_jac_fn,
    }


def make_poisson_model_wrappers(
    alpha,
    Sigma_tf,
    gamma_vec,
    nu,
    m1=1.0,
    m2=1.0 / 3.0,
):
    """Same wrapper logic, kept local and explicit."""
    @tf.function
    def prop_fn_single(x):
        return sample_skewed_t_v1(
            x,
            alpha=alpha,
            Sigma=Sigma_tf,
            gamma=gamma_vec,
            nu=nu,
        )

    @tf.function
    def prop_fn_batch(x):
        return tf.vectorized_map(prop_fn_single, x)

    @tf.function
    def h_func(x):
        return m1 * tf.exp(m2 * x)

    @tf.function
    def llk_fn_poisson(particles, y):
        return log_likelihood_poisson(
            particles,
            y,
            m1=m1,
            m2=m2,
        )

    @tf.function
    def H_jac_fn(x, t):
        return H_jac_t(
            x,
            t,
            m1=m1,
            m2=m2,
        )

    @tf.function
    def F_jac_fn(x, t):
        del t
        return tf.eye(tf.shape(x)[0], dtype=x.dtype) * tf.cast(alpha, x.dtype)

    return {
        "prop_fn_single": prop_fn_single,
        "prop_fn": prop_fn_batch,
        "h_func": h_func,
        "llk_fn": llk_fn_poisson,
        "H_jac_fn": H_jac_fn,
        "F_jac_fn": F_jac_fn,
    }


def build_dai_beta_star(dtype=tf.float64, N=50, mu_value=0.2, num_iters=1200):
    """
    Reproduces your Dai beta* block:
    - compute H from sensors/R
    - loss_J = compute_J(...)
    - optimize_beta(...)
    - derivative beta
    Does NOT call compute_stiffness_ratio.
    """
    mu = tf.constant(mu_value, dtype)
    h = tf.constant(1.0 / N, dtype)

    sensors = tf.constant(
        [[3.5, 0.0],
         [-3.5, 0.0]],
        dtype=dtype,
    )

    x_true = tf.constant([4.0, 4.0], dtype)
    x_prior = tf.constant([3.0, 5.0], dtype)

    R = tf.constant(
        [[0.04, 0.0],
         [0.0, 0.04]],
        dtype=dtype,
    )

    Q = tf.constant(
        [[4.0, 0.0],
         [0.0, 0.4]],
        dtype=dtype,
    )

    P = tf.constant(
        [[1000.0, 0.0],
         [0.0, 2.0]],
        dtype=dtype,
    )

    beta_straight = tf.linspace(
        tf.constant(0.0, dtype),
        tf.constant(1.0, dtype),
        N + 1,
    )

    H = compute_H(x=x_true, sensors=sensors, R=R)
    loss_J = lambda beta: compute_J(beta, H=H, Q=Q, mu=mu, h=h, dtype=dtype)

    J_straight = compute_J(beta_straight, H=H, Q=Q, mu=mu, h=h, dtype=dtype)
    beta_star, J_star = optimize_beta(
        loss_fct=loss_J,
        N=N,
        dtype=dtype,
        num_iters=num_iters,
    )

    beta_dot = compute_derivative_beta(beta_star)

    return {
        "beta_star": tf.cast(beta_star, tf.float32),
        "beta_straight": tf.cast(beta_straight, tf.float32),
        "beta_dot": tf.cast(beta_dot, tf.float32),
        "J_straight": J_straight,
        "J_star": J_star,
        "H": H,
        "Q": Q,
        "R": R,
        "P": P,
        "x_true": x_true,
        "x_prior": x_prior,
    }


def plot_beta_star(beta_info, filename):
    beta_star = beta_info["beta_star"].numpy()
    beta_straight = beta_info["beta_straight"].numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(beta_straight)), beta_straight, marker="o", label="straight")
    ax.plot(np.arange(len(beta_star)), beta_star, marker="o", label="Dai beta*")
    ax.set_xlabel("Homotopy index")
    ax.set_ylabel("beta")
    ax.set_title("Dai optimized homotopy schedule")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot helpers from pipeline, with save-and-close behavior
# ============================================================

def plot_gradient_conditioning_pip(
    diagnostic_results,
    hu_matrix=None,
    hu_scalar=None,
    title="Monte Carlo Estimate of Gradient Conditioning",
    filename=None,
):
    T = diagnostic_results["EDH"]["diagnostics"]["cond_J"].shape[1]
    fig, ax = plt.subplots(figsize=(8, 4))

    for name, color in zip(["EDH", "wEDH", "LEDH", "wLEDH"], ["C0", "C1", "C2", "C3"]):
        if name not in diagnostic_results:
            continue
        arr = diagnostic_results[name]["diagnostics"]["cond_J"]
        mean_arr = tf.reduce_mean(arr, axis=0)
        if len(mean_arr.shape) == 2:
            mean_arr = tf.reduce_mean(mean_arr, axis=1)
        ax.plot(range(T), mean_arr, marker="o", label=name, color=color)

    if hu_matrix is not None:
        hu = tf.reduce_mean(hu_matrix["diagnostics"]["grad_cond"], axis=0)
        ax.plot(range(T), hu, marker="o", label="matrix-PFF", color="C4")

    if hu_scalar is not None:
        hu = tf.reduce_mean(hu_scalar["diagnostics"]["grad_cond"], axis=0)
        ax.plot(range(T), hu, marker="o", label="scalar-PFF", color="C5")

    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Gradient conditioning value")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_flow_norm_boxplot_pip(
    diagnostic_results,
    hu_matrix=None,
    hu_scalar=None,
    title="Monte Carlo Estimates of Flow Norm Distributions",
    filename=None,
    log_scale=True,
):
    filters_diagn = {
        name: diagnostic_results[name]["diagnostics"]["flow_norm"]
        for name in ["EDH", "wEDH", "LEDH", "wLEDH"]
        if name in diagnostic_results
    }

    if hu_matrix is not None:
        filters_diagn["matrix-PFF"] = hu_matrix["diagnostics"]["flow_norm"]

    if hu_scalar is not None:
        filters_diagn["scalar-PFF"] = hu_scalar["diagnostics"]["flow_norm"]

    df_all_flow = []

    for algo_name, flow_data in filters_diagn.items():
        flow_mc_mean = tf.reduce_mean(flow_data, axis=0).numpy()
        if flow_mc_mean.ndim == 1:
            flow_mc_mean = flow_mc_mean[:, None]

        T, Np = flow_mc_mean.shape
        flow_flat = flow_mc_mean.reshape(-1)
        time_idx = np.repeat(np.arange(T), Np)

        df_all_flow.append(pd.DataFrame({
            "time_step": time_idx,
            "flow_norm": flow_flat,
            "algorithm": algo_name,
        }))

    df_plot = pd.concat(df_all_flow, axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df_plot,
        x="time_step",
        y="flow_norm",
        hue="algorithm",
        palette="Set2",
        fliersize=1,
        linewidth=1,
        ax=ax,
    )

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("MC Average Flow Norm (log scale)")
    else:
        ax.set_ylabel("MC Average Flow Norm")

    ax.set_xlabel("Time step")
    ax.set_title(title)
    ax.legend(title="Filter", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_norms_pip(
    diagnostic_results,
    hu_matrix=None,
    hu_scalar=None,
    title="Monte Carlo Estimate of Spectral Norms",
    filename=None,
):
    specJ_dict = {
        name: diagnostic_results[name]["diagnostics"]["spec_J"]
        for name in ["EDH", "wEDH", "LEDH", "wLEDH"]
        if name in diagnostic_results
    }

    if hu_matrix is not None:
        specJ_dict["matrix-PFF"] = hu_matrix["diagnostics"]["spec_J"]

    if hu_scalar is not None:
        specJ_dict["scalar-PFF"] = hu_scalar["diagnostics"]["spec_J"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for name, specJ in specJ_dict.items():
        arr = tf.convert_to_tensor(specJ).numpy()

        if arr.ndim >= 3:
            arr = np.mean(arr, axis=tuple(range(2, arr.ndim)))
        elif arr.ndim != 2:
            raise ValueError(f"{name}: unsupported shape {arr.shape}")

        arr = np.clip(arr, 1e-10, None)
        mean_arr = np.mean(arr, axis=0)

        t = np.arange(arr.shape[1])
        ax.plot(t, mean_arr, marker="o", linewidth=2, label=name)

    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Spectral norm")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def covariance_diag_over_time(particles, weights=None, return_std=True):
    arr = np.asarray(particles)
    if arr.ndim == 3:
        arr = arr[None, ...]
    N_MC, T, Np, d = arr.shape

    if weights is None:
        w = np.ones((N_MC, T, Np), dtype=float) / Np
    else:
        w = np.asarray(weights)
        if w.ndim == 2:
            w = w[None, ...]
        w = w / np.sum(w, axis=-1, keepdims=True)

    diag_vals = np.zeros((N_MC, T, d), dtype=float)

    for m in range(N_MC):
        for t in range(T):
            X = arr[m, t]
            wt = w[m, t]
            mu = np.sum(X * wt[:, None], axis=0)
            XC = X - mu
            var_diag = np.sum(wt[:, None] * (XC ** 2), axis=0)
            diag_vals[m, t] = np.sqrt(var_diag) if return_std else var_diag

    diag_mean = diag_vals.mean(axis=0)
    q_low = np.quantile(diag_vals, 0.025, axis=0)
    q_high = np.quantile(diag_vals, 0.975, axis=0)
    return diag_mean, diag_vals, q_low, q_high


def plot_posterior_spread_comparison(
    diagnostic_results,
    hu_matrix=None,
    methods=("EDH", "LEDH"),
    use_std=True,
    cmap="plasma",
    title=None,
    filename=None,
):
    panels = []

    if hu_matrix is not None:
        panels.append(("Matrix PFF (Hu)", hu_matrix["particles"]))

    for name in methods:
        if name in diagnostic_results:
            panels.append((f"PFPF-{name}", diagnostic_results[name]["particles"]))

    spread_maps = []
    for _, particles in panels:
        spread, _, _, _ = covariance_diag_over_time(
            particles,
            weights=None,
            return_std=use_std,
        )
        spread_maps.append(spread.T)

    vmin = min(H.min() for H in spread_maps)
    vmax = max(H.max() for H in spread_maps)

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6 * n_panels, 5),
        sharey=True,
    )

    if n_panels == 1:
        axes = [axes]

    fig.subplots_adjust(right=0.88)
    im = None

    for ax, (panel_title, _), H in zip(axes, panels, spread_maps):
        d, T = H.shape
        im = ax.imshow(
            H,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[1, T, 1, d],
        )
        ax.set_title(panel_title)
        ax.set_xlabel("Time step")

    axes[0].set_ylabel("State dimension")
    for ax in axes[1:]:
        ax.set_yticks([])

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Posterior std" if use_std else "Posterior variance")

    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig, axes, spread_maps


# ============================================================
# Example B: Gaussian
# ============================================================

def run_example_b_gaussian(beta, output_dir):
    print("\n========== EXAMPLE B: Gaussian ==========")

    d = EXAMPLE_B_D
    T = EXAMPLE_B_T
    Np = EXAMPLE_B_NP
    alpha = ALPHA
    sigma_z = tf.constant(SIGMA_Z, dtype=tf.float32)

    Sigma_tf = compute_Sigma_tf(d=d)
    true_states, measurements = Sim_HD_LGSSM(
        d=d,
        T=T,
        alpha=alpha,
        sigma_z=float(SIGMA_Z),
        Sigma_tf=Sigma_tf,
    )

    Y64 = tf.cast(tf.transpose(measurements), tf.float32)

    Q_mat_hd = Sigma_tf
    R_mat_hd = tf.eye(d, dtype=tf.float32) * sigma_z**2

    F_tf = tf.cast(tf.eye(d) * alpha, dtype=tf.float32)
    H_tf = tf.eye(d, dtype=tf.float32)
    Q_tf = Sigma_tf
    R_tf = tf.eye(d, dtype=tf.float32) * sigma_z**2
    L_tf = tf.linalg.cholesky(Q_tf)

    m0 = tf.zeros(d, dtype=tf.float32)
    P0 = tf.eye(d, dtype=tf.float32)

    F_tf_j = tf.eye(d, dtype=tf.float64) * alpha
    H_tf_j = tf.eye(d, dtype=tf.float64)

    wrappers_b = make_gaussian_wrappers_local(
        alpha=alpha,
        F_tf=F_tf,
        H_tf=H_tf,
        L_tf=L_tf,
        sigma_z=sigma_z,
    )

    prop_fn_b = wrappers_b["prop_fn"]
    h_func_b = wrappers_b["h_func"]
    llk_fn_b = wrappers_b["llk_fn"]
    H_jac_tf = wrappers_b["H_jac_fn"]

    transition_fn_gauss = make_transition_logpdf_gaussian(
        alpha=alpha,
        gamma=0.0,
    )

    out_ekf = run_ekf_wrap(
        Y=Y64,
        m0=m0,
        P0=P0,
        Q=Q_mat_hd,
        R=R_mat_hd,
        F=lambda x, t: alpha * x,
        H=lambda x, t: x,
        F_jac=lambda x, t: tf.eye(tf.shape(x)[0], dtype=x.dtype) * tf.cast(alpha, x.dtype),
        H_jac=lambda x, t: tf.eye(tf.shape(x)[0], dtype=x.dtype),
        measurement_type="gaussian",
    )

    pfpf_wedh_tf_b = run_pfpf_fn(flow_type="EDH", use_weights=True, beta=beta)
    pfpf_wledh_tf_b = run_pfpf_fn(flow_type="LEDH", use_weights=True, beta=beta)
    pfpf_edh_tf_b = run_pfpf_fn(flow_type="EDH", use_weights=False, beta=beta)
    pfpf_ledh_tf_b = run_pfpf_fn(flow_type="LEDH", use_weights=False, beta=beta)

    filters_config = {
        "PFPF-EDH": lambda: pfpf_wedh_tf_b(
            Y=measurements,
            Np=Np,
            P_pred=out_ekf["P_pred"],
            R_mat=R_tf,
            prop_fn=prop_fn_b,
            log_likelihood_fn=llk_fn_b,
            h_func=h_func_b,
            jacobian_func=H_jac_tf,
        )[:2],
        "PFPF-LEDH": lambda: pfpf_wledh_tf_b(
            Y=measurements,
            Np=Np,
            P_pred=out_ekf["P_pred"],
            R_mat=R_tf,
            prop_fn=prop_fn_b,
            log_likelihood_fn=llk_fn_b,
            h_func=h_func_b,
            jacobian_func=H_jac_tf,
        )[:2],
        "PF-EDH": lambda: pfpf_edh_tf_b(
            Y=measurements,
            Np=Np,
            P_pred=out_ekf["P_pred"],
            R_mat=R_tf,
            prop_fn=prop_fn_b,
            log_likelihood_fn=llk_fn_b,
            h_func=h_func_b,
            jacobian_func=H_jac_tf,
        )[:2],
        "PF-LEDH": lambda: pfpf_ledh_tf_b(
            Y=measurements,
            Np=Np,
            P_pred=out_ekf["P_pred"],
            R_mat=R_tf,
            prop_fn=prop_fn_b,
            log_likelihood_fn=llk_fn_b,
            h_func=h_func_b,
            jacobian_func=H_jac_tf,
        )[:2],
        "KF": lambda: run_kf_wrap(
            Y=Y64,
            m0=m0,
            P0=P0,
            Q=Q_mat_hd,
            R=R_mat_hd,
            F_mat=F_tf_j,
            H_mat=H_tf_j,
            measurement_type="gaussian",
        ),
        "UKF": lambda: run_ukf_wrap(
            Y=Y64,
            m0=m0,
            P0=P0,
            Q=Q_mat_hd,
            R=R_mat_hd,
            F=lambda x, t: alpha * x,
            H=lambda x, t: x,
        ),
        "BPF": lambda: run_bpf(
            Y=measurements,
            Np=Np,
            prop_fn=prop_fn_b,
            log_likelihood_fn=llk_fn_b,
        ),
        "UPF": lambda: run_upf(
            Y=measurements,
            Np=Np,
            Sigma=Sigma_tf,
            alpha=alpha,
            nu=200,
            gamma=0.0,
            transition_logpdf_fn=transition_fn_gauss,
            log_likelihood_fn=llk_fn_b,
            transition_mean_fn=gh_dynamics_mean,
        ),
        "ESRF": lambda: run_esrf(
            Y=tf.cast(measurements, tf.float32),
            Np=Np,
            F_func=lambda x: tf.matmul(x, tf.cast(F_tf, tf.float32), transpose_b=True),
            Q=Q_tf,
            H_func=lambda x: tf.matmul(x, tf.cast(H_tf, tf.float32), transpose_b=True),
            R=R_tf,
        ),
        "GSMC": lambda: run_gsmc(
            Y=measurements,
            Np=Np,
            log_likelihood_fn=llk_fn_b,
            prop_fn=prop_fn_b,
        ),
    }

    if INCLUDE_HEAVY_BASELINES:
        filters_config["PF-EDH-v1"] = lambda: pfpf_wedh_tf_b(
            Y=measurements,
            Np=10000,
            P_pred=out_ekf["P_pred"],
            R_mat=R_tf,
            prop_fn=prop_fn_b,
            log_likelihood_fn=llk_fn_b,
            h_func=h_func_b,
            jacobian_func=H_jac_tf,
        )[:2]
        filters_config["BPF-v1"] = lambda: run_bpf(
            Y=measurements,
            Np=100000,
            prop_fn=prop_fn_b,
            log_likelihood_fn=llk_fn_b,
        )

    results = run_monte_carlo_sim(
        filters_config=filters_config,
        true_data=true_states,
        N_MC=EXAMPLE_B_N_MC,
        monte_carlo_fn=monte_carlo_light_lost,
    )

    save_pickle(
        {
            "results": results,
            "true_states": true_states,
            "measurements": measurements,
            "out_ekf": out_ekf,
            "config": {
                "d": d,
                "T": T,
                "Np": Np,
                "N_MC": EXAMPLE_B_N_MC,
                "alpha": alpha,
                "sigma_z": float(SIGMA_Z),
                "beta": beta,
            },
        },
        os.path.join(output_dir, "example_B_gaussian_results.pkl"),
    )

    save_summary_csv(results, os.path.join(output_dir, "example_B_gaussian_summary.csv"))
    return {
        "results": results,
        "true_states": true_states,
        "measurements": measurements,
        "Sigma": Sigma_tf,
        "out_ekf": out_ekf,
        "prop_fn": prop_fn_b,
        "llk_fn": llk_fn_b,
        "h_func": h_func_b,
        "H_jac_fn": H_jac_tf,
        "Np": Np,
    }


# ============================================================
# Example C: Poisson/skew-t
# ============================================================

def run_example_c_poisson_for_dim(d_c, beta, output_dir):
    print(f"\n========== EXAMPLE C: Poisson/skew-t, d={d_c} ==========")

    T = EXAMPLE_C_T
    Np = EXAMPLE_C_NP
    alpha = ALPHA
    sigma_z = SIGMA_Z
    nu = NU
    gamma = GAMMA
    m1 = 1.0
    m2 = 1.0 / 3.0

    Sigma_tf_c = compute_Sigma_tf(d=d_c)
    gamma_c = tf.ones(d_c, dtype=tf.float32) * gamma

    data_c = generate_skt_poi_data(
        T=T,
        d=d_c,
        alpha=alpha,
        Sigma_proc=Sigma_tf_c,
        gamma=gamma_c,
        nu=nu,
        m1=m1,
        m2=m2,
        seed=None,
    )

    m0_t = tf.zeros(d_c, dtype=tf.float32)
    Q_mat_hd_t = Sigma_tf_c
    P0_t = tf.eye(d_c, dtype=tf.float32)
    R_mat_t = tf.eye(d_c, dtype=tf.float32) * sigma_z**2

    out_ekf_t = run_ekf_wrap(
        Y=tf.transpose(data_c[1]),
        m0=m0_t,
        P0=P0_t,
        Q=Q_mat_hd_t,
        R=R_mat_t,
        F=lambda x, t: alpha * x,
        H=lambda x, t: m1 * tf.exp(m2 * x),
        F_jac=lambda x, t: tf.eye(tf.shape(x)[0], dtype=x.dtype) * tf.cast(alpha, x.dtype),
        H_jac=lambda x, t: H_jac_t(x, t, m1=m1, m2=m2),
        measurement_type="poisson",
    )

    wrappers_poisson = make_poisson_model_wrappers(
        alpha=alpha,
        Sigma_tf=Sigma_tf_c,
        gamma_vec=gamma_c,
        nu=nu,
        m1=m1,
        m2=m2,
    )

    prop_fn_fixed = wrappers_poisson["prop_fn"]
    h_func_fixed = wrappers_poisson["h_func"]
    llk_fn_poisson = wrappers_poisson["llk_fn"]
    H_jac_t_tf = wrappers_poisson["H_jac_fn"]

    transition_fn_skewt = make_transition_logpdf_skewt(
        alpha=alpha,
        gamma=gamma_c,
        nu=nu,
    )

    pfpf_wedh_tf = run_pfpf_fn(flow_type="EDH", use_weights=True, measurement_type="poisson", beta=beta)
    pfpf_wledh_tf = run_pfpf_fn(flow_type="LEDH", use_weights=True, measurement_type="poisson", beta=beta)
    pfpf_edh_tf = run_pfpf_fn(flow_type="EDH", use_weights=False, measurement_type="poisson", beta=beta)
    pfpf_ledh_tf = run_pfpf_fn(flow_type="LEDH", use_weights=False, measurement_type="poisson", beta=beta)

    filters_config = {
        "PFPF-EDH": lambda: pfpf_wedh_tf(
            Y=data_c[1],
            Np=Np,
            P_pred=tf.cast(out_ekf_t["P_pred"], tf.float32),
            R_mat=R_mat_t,
            prop_fn=prop_fn_fixed,
            log_likelihood_fn=llk_fn_poisson,
            h_func=h_func_fixed,
            jacobian_func=H_jac_t_tf,
        )[:2],
        "PFPF-LEDH": lambda: pfpf_wledh_tf(
            Y=data_c[1],
            Np=Np,
            P_pred=tf.cast(out_ekf_t["P_pred"], tf.float32),
            R_mat=R_mat_t,
            prop_fn=prop_fn_fixed,
            log_likelihood_fn=llk_fn_poisson,
            h_func=h_func_fixed,
            jacobian_func=H_jac_t_tf,
        )[:2],
        "PF-EDH": lambda: pfpf_edh_tf(
            Y=data_c[1],
            Np=Np,
            P_pred=tf.cast(out_ekf_t["P_pred"], tf.float32),
            R_mat=R_mat_t,
            prop_fn=prop_fn_fixed,
            log_likelihood_fn=llk_fn_poisson,
            h_func=h_func_fixed,
            jacobian_func=H_jac_t_tf,
        )[:2],
        "PF-LEDH": lambda: pfpf_ledh_tf(
            Y=data_c[1],
            Np=Np,
            P_pred=tf.cast(out_ekf_t["P_pred"], tf.float32),
            R_mat=R_mat_t,
            prop_fn=prop_fn_fixed,
            log_likelihood_fn=llk_fn_poisson,
            h_func=h_func_fixed,
            jacobian_func=H_jac_t_tf,
        )[:2],
        "EKF": lambda: tf.transpose(out_ekf_t["mu_filt"]),
        "UKF": lambda: run_ukf_wrap(
            Y=tf.transpose(data_c[1]),
            m0=m0_t,
            P0=P0_t,
            Q=Q_mat_hd_t,
            R=R_mat_t,
            F=lambda x, t: alpha * x,
            H=lambda x, t: m1 * tf.exp(m2 * x),
            measurement_type="poisson",
        ),
        "ESRF": lambda: run_esrf(
            Y=data_c[1],
            Np=Np,
            F_func=lambda particles: tf.cast(alpha, particles.dtype) * particles,
            Q=Q_mat_hd_t,
            H_func=lambda particles: m1 * tf.exp(m2 * particles),
            R=R_mat_t,
            measurement_type="poisson",
        ),
        "UPF": lambda: run_upf(
            Y=data_c[1],
            Np=Np,
            Sigma=Sigma_tf_c,
            alpha=alpha,
            nu=nu,
            gamma=gamma_c,
            transition_logpdf_fn=transition_fn_skewt,
            log_likelihood_fn=llk_fn_poisson,
            transition_mean_fn=gh_dynamics_mean,
        ),
        "GSMC": lambda: run_gsmc(
            Y=data_c[1],
            Np=Np,
            log_likelihood_fn=llk_fn_poisson,
            prop_fn=prop_fn_fixed,
        ),
        "SMHMC": lambda: smhmc_helper(
            Y=data_c[1],
            Np=Np,
            prop_fn=wrappers_poisson["prop_fn_single"],
            log_likelihood_fn=llk_fn_poisson,
            leapfrog_steps=5,
            epsilon=0.03,
            resample_threshold=True,
            dtype=tf.float32,
        ),
    }

    if INCLUDE_HEAVY_BASELINES:
        filters_config["PF-EDH-v1"] = lambda: run_pfpf_fn(
            flow_type="EDH",
            use_weights=False,
            measurement_type="poisson",
        )(
            Y=data_c[1],
            Np=10000,
            P_pred=out_ekf_t["P_pred"],
            R_mat=R_mat_t,
            prop_fn=prop_fn_fixed,
            log_likelihood_fn=llk_fn_poisson,
            h_func=h_func_fixed,
            jacobian_func=H_jac_t_tf,
        )[:2]

        filters_config["Block-BPF"] = lambda: bpf_block(
            Y=data_c[1],
            Np=10000,
            prop_fn=wrappers_poisson["prop_fn_single"],
            log_likelihood_fn=llk_fn_poisson,
            resample_threshold=False,
        )

    results = run_monte_carlo_sim(
        filters_config=filters_config,
        true_data=data_c[0],
        N_MC=EXAMPLE_C_N_MC,
        monte_carlo_fn=monte_carlo_light_lost,
    )

    save_pickle(
        {
            "results": results,
            "data": data_c,
            "out_ekf": out_ekf_t,
            "config": {
                "d": d_c,
                "T": T,
                "Np": Np,
                "N_MC": EXAMPLE_C_N_MC,
                "alpha": alpha,
                "sigma_z": sigma_z,
                "gamma": gamma,
                "nu": nu,
                "m1": m1,
                "m2": m2,
                "beta": beta,
            },
        },
        os.path.join(output_dir, f"example_C_poisson_d{d_c}_results.pkl"),
    )

    save_summary_csv(results, os.path.join(output_dir, f"example_C_poisson_d{d_c}_summary.csv"))
    return results


# ============================================================
# EDH/LEDH pipeline and diagnostics
# ============================================================

def run_edh_ledh_pipeline(
    d,
    T=10,
    Np=200,
    N_MC_accuracy=100,
    N_MC_diagnostics=2,
    alpha=0.9,
    sigma_z=1.0,
    gamma=0.3,
    nu=5,
    m1=1.0,
    m2=1.0 / 3.0,
    beta=None,
    seed=None,
    methods=None,
    run_accuracy=True,
    run_diagnostics=True,
    monte_carlo_fn=monte_carlo_light_lost,
):
    if methods is None:
        methods = [
            ("EDH", False),
            ("EDH", True),
            ("LEDH", False),
            ("LEDH", True),
        ]

    Sigma_tf = compute_Sigma_tf(d=d)
    gamma_vec = tf.ones(d, dtype=tf.float32) * gamma

    data = generate_skt_poi_data(
        T=T,
        d=d,
        alpha=alpha,
        Sigma_proc=Sigma_tf,
        gamma=gamma_vec,
        nu=nu,
        m1=m1,
        m2=m2,
        seed=seed,
    )

    Q = Sigma_tf
    R = tf.eye(d, dtype=tf.float32) * sigma_z**2

    out_ekf = run_ekf_wrap(
        Y=tf.transpose(data[1]),
        m0=tf.zeros(d, dtype=tf.float32),
        P0=tf.eye(d, dtype=tf.float32),
        Q=Q,
        R=R,
        F=lambda x, t: alpha * x,
        H=lambda x, t: m1 * tf.exp(m2 * x),
        F_jac=lambda x, t: tf.eye(tf.shape(x)[0], dtype=x.dtype) * tf.cast(alpha, x.dtype),
        H_jac=lambda x, t: H_jac_t(x, t, m1=m1, m2=m2),
        measurement_type="poisson",
    )

    wrappers = make_poisson_model_wrappers(
        alpha=alpha,
        Sigma_tf=Sigma_tf,
        gamma_vec=gamma_vec,
        nu=nu,
        m1=m1,
        m2=m2,
    )

    prop_fn = wrappers["prop_fn"]
    h_func = wrappers["h_func"]
    llk_fn = wrappers["llk_fn"]
    H_jac_fn = wrappers["H_jac_fn"]

    filter_kwargs = dict(
        Y=data[1],
        Np=Np,
        P_pred=tf.cast(out_ekf["P_pred"], tf.float32),
        R_mat=R,
        prop_fn=prop_fn,
        log_likelihood_fn=llk_fn,
        h_func=h_func,
        jacobian_func=H_jac_fn,
    )

    filter_fns = {}
    filters_config = {}

    for flow_type, use_weights in methods:
        name = f"{'w' if use_weights else ''}{flow_type}"

        pfpf_fn = run_pfpf_fn(
            flow_type=flow_type,
            use_weights=use_weights,
            measurement_type="poisson",
            beta=beta,
        )

        filter_fns[name] = pfpf_fn
        filters_config[name] = lambda pfpf_fn=pfpf_fn: pfpf_fn(**filter_kwargs)[:2]

    cfg = {
        "d": d,
        "T": T,
        "Np": Np,
        "data": data,
        "Sigma": Sigma_tf,
        "gamma_vec": gamma_vec,
        "Q": Q,
        "R": R,
        "out_ekf": out_ekf,
        "methods": methods,
        "filter_fns": filter_fns,
        "filters_config": filters_config,
        **wrappers,
        "params": {
            "alpha": alpha,
            "sigma_z": sigma_z,
            "gamma": gamma,
            "nu": nu,
            "m1": m1,
            "m2": m2,
            "beta": beta,
        },
    }

    output = {
        "config": cfg,
        "accuracy_mc": None,
        "diagnostics_mc": None,
    }

    if run_accuracy:
        output["accuracy_mc"] = run_monte_carlo_sim(
            filters_config=filters_config,
            true_data=data[0],
            N_MC=N_MC_accuracy,
            monte_carlo_fn=monte_carlo_fn,
        )

    if run_diagnostics:
        diagnostics_kwargs = dict(
            N_MC=N_MC_diagnostics,
            output_names=["ests", "ESS", "particles", "diagnostics"],
            **filter_kwargs,
        )

        output["diagnostics_mc"] = {
            name: monte_carlo_final(
                filter_fn=pfpf_fn,
                **diagnostics_kwargs,
            )
            for name, pfpf_fn in filter_fns.items()
        }

    return output


def run_diagnostic_pipeline(beta, output_dir, m1=0.05):
    print(f"\n========== EDH/LEDH diagnostics, m1={m1} ==========")

    out = run_edh_ledh_pipeline(
        d=DIAGNOSTIC_D,
        T=EXAMPLE_C_T,
        Np=DIAGNOSTIC_NP,
        N_MC_accuracy=DIAGNOSTIC_N_MC_ACCURACY,
        N_MC_diagnostics=DIAGNOSTIC_N_MC_DIAGNOSTICS,
        alpha=ALPHA,
        sigma_z=SIGMA_Z,
        gamma=GAMMA,
        nu=NU,
        m1=m1,
        m2=M2_VALUE,
        beta=beta,
        seed=None,
        run_accuracy=True,
        run_diagnostics=True,
    )

    save_pickle(out, os.path.join(output_dir, f"edh_ledh_diagnostics_m1_{m1}.pkl"))

    if out["accuracy_mc"] is not None:
        save_summary_csv(
            out["accuracy_mc"],
            os.path.join(output_dir, f"edh_ledh_accuracy_m1_{m1}.csv"),
        )

    return out


# ============================================================
# Hu pipeline for m1 sparsity values
# ============================================================

def run_hu_for_m1(m1, d, beta, output_dir):
    print(f"\n========== Hu PFF comparison, m1={m1}, d={d} ==========")

    d = int(d)
    T = EXAMPLE_C_T
    Np = HU_NP
    alpha = ALPHA
    gamma = GAMMA
    nu = NU
    m2 = M2_VALUE

    Sigma_tf = compute_Sigma_tf(d=d)
    gamma_vec = tf.ones(d, dtype=tf.float32) * gamma

    data = generate_skt_poi_data(
        T=T,
        d=d,
        alpha=alpha,
        Sigma_proc=Sigma_tf,
        gamma=gamma_vec,
        nu=nu,
        m1=m1,
        m2=m2,
        seed=None,
    )

    wrappers = make_poisson_model_wrappers(
        alpha=alpha,
        Sigma_tf=Sigma_tf,
        gamma_vec=gamma_vec,
        nu=nu,
        m1=m1,
        m2=m2,
    )

    loglik_grad_fn = make_poisson_grad_wrapper(
        m1=m1,
        m2=m2,
    )

    hu_runners = make_hu_pff_runners(
        Sigma_tf=Sigma_tf,
        prop_fn=wrappers["prop_fn_single"],
        loglik_grad_fn=loglik_grad_fn,
        n_steps=HU_N_STEPS,
        eps=HU_EPS,
    )

    out_hu_mc_diagn_matrix = monte_carlo_final(
        filter_fn=hu_runners["matrix"],
        Y=data[1],
        N_MC=HU_N_MC,
        output_names=["ests", "particles", "diagnostics"],
        Np=Np,
    )

    out_hu_mc_diagn_scalar = monte_carlo_final(
        filter_fn=hu_runners["scalar"],
        Y=data[1],
        N_MC=HU_N_MC,
        output_names=["ests", "particles", "diagnostics"],
        Np=Np,
    )

    filter_config_hu = {
        "matrix-PFF": lambda: hu_runners["matrix"](Y=data[1], Np=Np)[0],
        "scalar-PFF": lambda: hu_runners["scalar"](Y=data[1], Np=Np)[0],
    }

    results_mc_hu = run_monte_carlo_sim(
        filters_config=filter_config_hu,
        true_data=data[0],
        N_MC=HU_N_MC,
        monte_carlo_fn=monte_carlo_light_lost,
    )

    # For side-by-side plots, run EDH/LEDH diagnostics at same m1.
    out_edh_ledh = run_edh_ledh_pipeline(
        d=d,
        T=T,
        Np=Np,
        N_MC_accuracy=HU_N_MC,
        N_MC_diagnostics=HU_N_MC,
        alpha=alpha,
        sigma_z=SIGMA_Z,
        gamma=gamma,
        nu=nu,
        m1=m1,
        m2=m2,
        beta=beta,
        seed=None,
        run_accuracy=True,
        run_diagnostics=True,
    )

    diagnostic_results = out_edh_ledh["diagnostics_mc"]

    tag = str(m1).replace(".", "p")
    plot_gradient_conditioning_pip(
        diagnostic_results,
        hu_matrix=out_hu_mc_diagn_matrix,
        hu_scalar=out_hu_mc_diagn_scalar,
        title=f"Gradient Conditioning, m1={m1}, d={d}",
        filename=os.path.join(output_dir, f"gradient_conditioning_m1_{tag}_d{d}.png"),
    )

    plot_flow_norm_boxplot_pip(
        diagnostic_results,
        hu_matrix=out_hu_mc_diagn_matrix,
        hu_scalar=out_hu_mc_diagn_scalar,
        title=f"Flow Norm Distributions, m1={m1}, d={d}",
        filename=os.path.join(output_dir, f"flow_norm_boxplot_m1_{tag}_d{d}.png"),
    )

    plot_spectral_norms_pip(
        diagnostic_results,
        hu_matrix=out_hu_mc_diagn_matrix,
        hu_scalar=out_hu_mc_diagn_scalar,
        title=f"Spectral Norms, m1={m1}, d={d}",
        filename=os.path.join(output_dir, f"spectral_norms_m1_{tag}_d{d}.png"),
    )

    try:
        plot_posterior_spread_comparison(
            diagnostic_results,
            hu_matrix=out_hu_mc_diagn_matrix,
            methods=("EDH", "LEDH"),
            title=f"Posterior Spread, m1={m1}, d={d}",
            filename=os.path.join(output_dir, f"posterior_spread_m1_{tag}_d{d}.png"),
        )
    except Exception as exc:
        print(f"[WARN] posterior spread plot skipped for m1={m1}: {exc}")

    out = {
        "m1": m1,
        "m2": m2,
        "data": data,
        "Sigma": Sigma_tf,
        "hu_matrix": out_hu_mc_diagn_matrix,
        "hu_scalar": out_hu_mc_diagn_scalar,
        "hu_accuracy": results_mc_hu,
        "edh_ledh": out_edh_ledh,
        "config": {
            "d": d,
            "T": T,
            "Np": Np,
            "N_MC": HU_N_MC,
            "n_steps": HU_N_STEPS,
            "eps": HU_EPS,
            "alpha": alpha,
            "gamma": gamma,
            "nu": nu,
            "beta": beta,
        },
    }

    save_pickle(out, os.path.join(output_dir, f"hu_sparsity_m1_{tag}_d{d}.pkl"))
    save_summary_csv(results_mc_hu, os.path.join(output_dir, f"hu_accuracy_m1_{tag}_d{d}.csv"))

    return out




# ============================================================
# Dai beta schedule comparison: straight / geometric / beta*
# ============================================================

def run_beta_schedule_comparison(beta_schedules, output_dir):
    """
    Compare EDH/LEDH under multiple homotopy schedules.

    This is the part from the notebook where beta_star from Dai is compared
    against other beta schedules. It runs the same EDH/LEDH diagnostic pipeline
    for each beta and for the requested dimensions and sparsity levels.
    """
    print("\n========== Dai beta schedule comparison ==========")
    rows = []
    all_outputs = {}

    for d in BETA_COMPARISON_DIMS:
        for m1 in BETA_COMPARISON_M1_VALUES:
            for beta_name, beta_value in beta_schedules.items():
                print(f"\n--- beta={beta_name}, d={d}, m1={m1} ---")
                out = run_edh_ledh_pipeline(
                    d=d,
                    T=EXAMPLE_C_T,
                    Np=DIAGNOSTIC_NP,
                    N_MC_accuracy=DIAGNOSTIC_N_MC_ACCURACY,
                    N_MC_diagnostics=DIAGNOSTIC_N_MC_DIAGNOSTICS,
                    alpha=ALPHA,
                    sigma_z=SIGMA_Z,
                    gamma=GAMMA,
                    nu=NU,
                    m1=m1,
                    m2=M2_VALUE,
                    beta=beta_value,
                    seed=None,
                    run_accuracy=True,
                    run_diagnostics=True,
                )
                key = f"d{d}_m1{str(m1).replace('.', 'p')}_{beta_name}"
                all_outputs[key] = out
                save_pickle(out, os.path.join(output_dir, f"beta_compare_{key}.pkl"))

                if out["accuracy_mc"] is not None:
                    df = sims_to_df(out["accuracy_mc"])
                    df["beta"] = beta_name
                    df["d"] = d
                    df["m1"] = m1
                    rows.append(df.reset_index().rename(columns={"index": "method"}))

                # Diagnostic plots for each beta schedule.
                try:
                    diagnostic_results = out["diagnostics_mc"]
                    plot_gradient_conditioning_pip(
                        diagnostic_results,
                        title=f"Gradient Conditioning: beta={beta_name}, d={d}, m1={m1}",
                        filename=os.path.join(output_dir, f"beta_compare_gradient_{key}.png"),
                    )
                    plot_spectral_norms_pip(
                        diagnostic_results,
                        title=f"Spectral Norms: beta={beta_name}, d={d}, m1={m1}",
                        filename=os.path.join(output_dir, f"beta_compare_spectral_{key}.png"),
                    )
                except Exception as exc:
                    print(f"[WARN] beta comparison plots skipped for {key}: {exc}")

    if rows:
        table = pd.concat(rows, axis=0)
        table.to_csv(os.path.join(output_dir, "beta_schedule_comparison_summary.csv"), index=False)
    else:
        table = pd.DataFrame()

    save_pickle(all_outputs, os.path.join(output_dir, "beta_schedule_comparison_all.pkl"))
    return all_outputs, table


# ============================================================
# Gradient comparison for resampling methods
# ============================================================

def gradient_comparison_experiment_local(
    Y,
    Np,
    Sigma_chol,
    Sigma,
    sigma_z,
    methods,
    n_runs=50,
    dtype=tf.float32,
    base_seed=12345,
):
    """
    Lightweight local implementation of the gradient comparison block.

    It compares gradients through the resampling step using the same method
    dictionary used in the notebook: each entry has {"fn": ..., "carry": ...}.
    If an external gradient_comparison_experiment exists, run_gradient_comparison
    will use it instead.
    """
    Y = tf.cast(Y, dtype)
    Sigma_chol = tf.cast(Sigma_chol, dtype)
    sigma_z = tf.cast(sigma_z, dtype)
    T = int(Y.shape[0])
    d = int(Y.shape[1])

    def one_grad(method_fn, seed):
        tf.random.set_seed(seed)
        x0 = tf.random.normal((Np, d), dtype=dtype)
        with tf.GradientTape() as tape:
            tape.watch(x0)
            particles = x0
            weights = tf.ones((Np,), dtype=dtype) / tf.cast(Np, dtype)
            loss = tf.constant(0.0, dtype=dtype)
            for t in range(T):
                eps = tf.random.normal((Np, d), dtype=dtype)
                particles = particles + tf.matmul(eps, Sigma_chol, transpose_b=True)
                llk = loglik_gaussian(particles, Y[t], sigma_z)
                log_w = llk - tf.reduce_logsumexp(llk)
                weights = tf.exp(log_w)
                est = tf.reduce_sum(particles * weights[:, None], axis=0)
                loss = loss + tf.reduce_mean((est - Y[t]) ** 2)
                particles, weights = method_fn(particles, weights)
            loss = loss / tf.cast(T, dtype)
        grad = tape.gradient(loss, x0)
        if grad is None:
            grad = tf.zeros_like(x0)
        grad = tf.where(tf.math.is_finite(grad), grad, tf.zeros_like(grad))
        return grad, loss

    rows = []
    for run in range(n_runs):
        seed = base_seed + run
        ref_grad, ref_loss = one_grad(methods["No-Resampling"]["fn"], seed)
        ref_norm = tf.norm(ref_grad).numpy()
        for name, spec in methods.items():
            grad, loss = one_grad(spec["fn"], seed)
            grad_rmse = tf.sqrt(tf.reduce_mean((grad - ref_grad) ** 2)).numpy()
            rows.append({
                "method": name,
                "run": run,
                "loss": float(loss.numpy()),
                "grad_norm": float(tf.norm(grad).numpy()),
                "reference_grad_norm": float(ref_norm),
                "grad_rmse": float(grad_rmse),
                "carry": bool(spec.get("carry", False)),
            })

    return pd.DataFrame(rows)


def run_ot_tuning_and_gradient_comparison(example_b_result, output_dir):
    """Run the OT tuning block and the gradient comparison block from the notebook."""
    print("\n========== OT tuning + gradient comparison ==========")
    true_states = example_b_result["true_states"]
    measurements = example_b_result["measurements"]
    Sigma_tf = example_b_result["Sigma"]
    prop_fn_b = example_b_result["prop_fn"]
    llk_fn_b = example_b_result["llk_fn"]
    Np = example_b_result["Np"]

    best_params_ot = {"epsilon": 1e-2, "sinkhorn_iters": 8}
    best_results_ot = None
    results_ot = None

    best_params_robust_ot = {
        "epsilon": 1e-2,
        "sinkhorn_iters": 8,
        "robust_cost": True,
        "lambda_robust": 5.0,
        "robust_mode": "smooth_clip",
    }
    best_results_robust_ot = None
    results_robust_ot = None

    if RUN_OT_TUNING:
        try:
            best_params_ot, best_results_ot, results_ot = tune_ot_entropy_regularized(
                filter_fn=lambda resampling_fn: run_bpf_ot(
                    resampling_fn=resampling_fn,
                    Y=measurements,
                    Np_value=Np,
                    prop_fn=prop_fn_b,
                    log_likelihood_fn=llk_fn_b,
                    dtype=tf.float32,
                ),
                true_state=true_states,
                Np=Np,
                niter_grid=(4, 6, 8, 10, 12),
                eps_grid=(1e-3, 1e-2, 1e-1, 1.0),
                lambda_ess=0.96,
                lambda_speed=0.04,
                n_repeats=2 if QUICK_RUN else 3,
            )
            print("best_params_ot =", best_params_ot)
            save_pickle(
                {"best_params": best_params_ot, "best_results": best_results_ot, "results_table": results_ot},
                os.path.join(output_dir, "ot_tuning_results.pkl"),
            )
            try:
                pd.DataFrame(results_ot).to_csv(os.path.join(output_dir, "ot_tuning_results.csv"), index=False)
            except Exception:
                pass
        except Exception as exc:
            print(f"[WARN] OT tuning failed; using default params {best_params_ot}. Error: {exc}")
        
        try:
            best_params_robust_ot, best_results_robust_ot, results_robust_ot = tune_ot_entropy_regularized(
                filter_fn=lambda resampling_fn: run_bpf_ot(
                    resampling_fn=resampling_fn,
                    Y=measurements,
                    Np_value=Np,
                    prop_fn=prop_fn_b,
                    log_likelihood_fn=llk_fn_b,
                    dtype=tf.float32,
                ),
                true_state=true_states,
                Np=Np,
                niter_grid=(4, 6, 8, 10, 12),
                eps_grid=(1e-3, 1e-2, 1e-1, 1.0),
                lambda_ess=0.96,
                lambda_speed=0.04,
                n_repeats=2 if QUICK_RUN else 3,
                robust_cost=True,
                lambda_robust=5.0,
                robust_mode="smooth_clip",
            )

            print("best_params_robust_ot =", best_params_robust_ot)

            save_pickle(
                {
                    "best_params": best_params_robust_ot,
                    "best_results": best_results_robust_ot,
                    "results_table": results_robust_ot,
                },
                os.path.join(output_dir, "robust_ot_tuning_results.pkl"),
            )

            try:
                pd.DataFrame(results_robust_ot).to_csv(
                    os.path.join(output_dir, "robust_ot_tuning_results.csv"),
                    index=False,
                )
            except Exception:
                pass

        except Exception as exc:
            print(
                f"[WARN] Robust OT tuning failed; using default params "
                f"{best_params_robust_ot}. Error: {exc}"
            )

    if not RUN_GRADIENT_COMPARISON:
        return best_params_ot, best_results_ot, results_ot, None

    dtype_grad = tf.float32
    grad_resampling_methods = {
        "No-Resampling": {
            "fn": no_resampling,
            "carry": True,
        },
        "Multinomial": {
            "fn": multinomial_resampling,
            "carry": False,
        },
        "Mixture-Uniform Multinomial": {
            "fn": mixture_unif_multinomial_resampling,
            "carry": False,
        },
        "PFNet-Soft": {
            "fn": lambda p, w: soft_resampling_pfnet(p, w, alpha=0.35),
            "carry": True,
        },
        "Sinkhorn-OT": {
            "fn": lambda p, w: soft_resample_ot(
                p,
                w,
                epsilon=tf.cast(best_params_ot["epsilon"], dtype_grad),
                sinkhorn_iters=best_params_ot["sinkhorn_iters"],
                normalize_cost=True,
            ),
            "carry": False,
        },
        "Robust-Sinkhorn-OT": {
            "fn": lambda p, w: soft_resample_ot(
                p,
                w,
                epsilon=tf.cast(best_params_robust_ot["epsilon"], dtype_grad),
                sinkhorn_iters=best_params_robust_ot["sinkhorn_iters"],
                normalize_cost=True,
                robust_cost=True,
                lambda_robust=tf.cast(best_params_robust_ot.get("lambda_robust", 5.0), p.dtype),
                robust_mode=best_params_robust_ot.get("robust_mode", "smooth_clip"),
            ),
            "carry": False,
        },
    }

    Sigma_chol_grad = tf.linalg.cholesky(tf.cast(Sigma_tf, dtype_grad))
    sigma_z_grad = tf.cast(tf.convert_to_tensor(SIGMA_Z), dtype_grad)

    grad_fn = globals().get("gradient_comparison_experiment", gradient_comparison_experiment_local)

    results_df_grad = grad_fn(
        Y=tf.cast(measurements, dtype_grad),
        Np=Np,
        Sigma_chol=Sigma_chol_grad,
        Sigma=tf.cast(Sigma_tf, dtype_grad),
        sigma_z=sigma_z_grad,
        methods=grad_resampling_methods,
        n_runs=GRADIENT_N_RUNS,
        dtype=dtype_grad,
        base_seed=12345,
    )

    print(results_df_grad.sort_values("grad_rmse").head())
    results_df_grad.to_csv(os.path.join(output_dir, "gradient_comparison_results.csv"), index=False)
    return best_params_ot, best_results_ot, results_ot, results_df_grad


# ============================================================
# Main
# ============================================================

def main():
    start = time.time()
    ensure_dir(OUTPUT_DIR)
    set_seed(SEED)

    print("Running submission pipeline")
    print(f"QUICK_RUN = {QUICK_RUN}")
    print(f"Output directory: {OUTPUT_DIR}")

    # 1. Geometric beta used in the first notebook blocks.
    beta_geom, steps = make_beta_schedule(
        N_lambda=BETA_N_LAMBDA,
        q=BETA_Q,
        dtype=tf.float32,
    )
    print("Geometric beta schedule:")
    print("  first step =", steps[0].numpy())
    print("  last step  =", steps[-1].numpy())
    print("  sum steps  =", tf.reduce_sum(steps).numpy())

    # 2. Dai beta*.
    beta_iters = 200 if QUICK_RUN else 1200
    beta_info = build_dai_beta_star(
        dtype=tf.float64,
        N=50,
        mu_value=0.2,
        num_iters=beta_iters,
    )
    beta_star = beta_info["beta_star"]

    print("Dai beta*:")
    print("  J straight =", beta_info["J_straight"].numpy())
    print("  J star     =", beta_info["J_star"].numpy())

    save_pickle(beta_info, os.path.join(OUTPUT_DIR, "dai_beta_star.pkl"))
    plot_beta_star(beta_info, os.path.join(OUTPUT_DIR, "dai_beta_star.png"))

    # 3. Example B with the original geometric beta from the notebook.
    example_b_results = run_example_b_gaussian(
        beta=beta_geom,
        output_dir=OUTPUT_DIR,
    )

    beta_schedules = {
        "straight": tf.cast(beta_info["beta_straight"], tf.float32),
        "geometric": beta_geom,
        "dai_star": beta_star,
    }

    # 4. Dai beta comparison: straight/geometric/beta* across d and m1.
    beta_comparison_outputs, beta_comparison_table = run_beta_schedule_comparison(
        beta_schedules=beta_schedules,
        output_dir=OUTPUT_DIR,
    )

    # 5. Example C for requested dimensions.
    example_c_results = {}
    for d_c in EXAMPLE_C_DIMS:
        example_c_results[d_c] = run_example_c_poisson_for_dim(
            d_c=d_c,
            beta=beta_geom,
            output_dir=OUTPUT_DIR,
        )

    # 6. Diagnostics with sparse m1 and geometric beta.
    sparse_m1 = 0.05
    diagnostic_out = run_diagnostic_pipeline(
        beta=beta_geom,
        output_dir=OUTPUT_DIR,
        m1=sparse_m1,
    )

    # 7. Diagnostics with Dai beta* for comparison.
    diagnostic_out_beta_star = run_diagnostic_pipeline(
        beta=beta_star,
        output_dir=OUTPUT_DIR,
        m1=sparse_m1,
    )
    save_pickle(
        diagnostic_out_beta_star,
        os.path.join(OUTPUT_DIR, f"edh_ledh_diagnostics_beta_star_m1_{sparse_m1}.pkl"),
    )

    # 8. Hu for different m1 sparsity values and dimensions.
    hu_results = {}
    for d_hu in HU_DIMS:
        for m1 in M1_VALUES:
            hu_results[(d_hu, m1)] = run_hu_for_m1(
                m1=m1,
                d=d_hu,
                beta=beta_geom,
                output_dir=OUTPUT_DIR,
            )

    # 9. OT tuning and gradient comparison from the final block.
    ot_grad_results = run_ot_tuning_and_gradient_comparison(
        example_b_result=example_b_results,
        output_dir=OUTPUT_DIR,
    )

    final_index = {
        "quick_run": QUICK_RUN,
        "output_dir": OUTPUT_DIR,
        "example_B": "example_B_gaussian_results.pkl",
        "beta_schedule_comparison": "beta_schedule_comparison_summary.csv",
        "example_C_dims": EXAMPLE_C_DIMS,
        "diagnostics_sparse": f"edh_ledh_diagnostics_m1_{sparse_m1}.pkl",
        "diagnostics_sparse_beta_star": f"edh_ledh_diagnostics_beta_star_m1_{sparse_m1}.pkl",
        "hu_m1_values": M1_VALUES,
        "hu_dims": HU_DIMS,
        "gradient_comparison": "gradient_comparison_results.csv",
        "ot_tuning": "ot_tuning_results.pkl",
        "runtime_seconds": time.time() - start,
    }

    save_pickle(final_index, os.path.join(OUTPUT_DIR, "RUN_INDEX.pkl"))

    print("\nDONE")
    print(f"Total runtime: {final_index['runtime_seconds']:.2f} seconds")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

# END OF FILE
