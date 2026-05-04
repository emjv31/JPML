import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.stattools import acf, pacf



def compute_sigma_points_tf(mu, P, alpha=0.5, beta=2.0, kappa=0.0):
    """
    Computes sigma points and corresponding weights for the Unscented Transform
    Operates on a single scalar state; for multivariate states, apply independently per dimension.

    Input:
    ------
    mu : float or tf.Tensor, mean of the state
    P : float or tf.Tensor, variance of the state 
    alpha : float, spread scaling parameter (optional, set to 0.5)
    beta : float, distribution prior info parameter (optional, set to 2.0)
    kappa : float, secondary scaling parameter, (optional, set to 0.0)

    Output:
    ------
    dict: 
        "X" : tf.Tensor, sigma points [mu, mu+spread, mu-spread]  (shape [3])
        "Wm" :  tf.Tensor, weights for mean  (shape [3])
        "Wc" tf.Tensor, weights for covariance (shape [3])
    """
    
    # Conversion 
    mu = tf.convert_to_tensor(mu, dtype=tf.float64)
    P = tf.convert_to_tensor(P, dtype=tf.float64)
    alpha = tf.convert_to_tensor(alpha, dtype=tf.float64)
    beta = tf.convert_to_tensor(beta, dtype=tf.float64)
    kappa = tf.convert_to_tensor(kappa, dtype=tf.float64)

    # Check input 
    if not tf.math.is_finite(mu):
        raise ValueError("Sigma points: mu not finite")
    if not tf.math.is_finite(P) or P <= 0:
        raise ValueError("Sigma points: P must be positive")

    n = 1
    lambda_ = alpha**2 * (n + kappa) - n
    denom = n + lambda_

    if denom <= 0 or not tf.math.is_finite(denom):
        raise ValueError("Invalid scaling: n + lambda <= 0")

    gamma = tf.sqrt(denom)
    spread = gamma * tf.sqrt(P)

    # Robustness checks
    if not tf.math.is_finite(spread) or spread <= 0:
        raise ValueError("Sigma points collapsed: spread <= 0")
    # 
    if spread < 1e-14 * (tf.abs(mu) + 1):
        raise ValueError("Sigma points numerically degenerate (spread too small)")

    # Compute sigma points: in the univariate case three points are needed 
    X = tf.stack([mu, mu + spread, mu - spread])

    # Check degeneracy 
    tol = 1e-12
    if tf.reduce_max(tf.abs(X - X[0])) < tol:
        raise ValueError("Sigma points degenerate: all points coincide")

    # Compute weights
    Wm = tf.stack([lambda_ / denom, 1.0/(2*denom), 1.0/(2*denom)])
    Wc = tf.identity(Wm)
    Wc = tf.tensor_scatter_nd_add(Wc, [[0]], [1 - alpha**2 + beta])

    # Check weights finite
    if not tf.reduce_all(tf.math.is_finite(Wm)) or not tf.reduce_all(tf.math.is_finite(Wc)):
        raise ValueError("Non-finite UKF weights")

    return {"X": X, "Wm": Wm, "Wc": Wc}


def compute_jacobian_tf(fun, x, tol=1e-6):
    """
    Compute numerical derivative of fun at x using central difference.

    Input:
    ------
    fun : callable, function to differentiate
    x : float or tf.Tensor, point where derivative is evaluated 
    tol : float, small perturbation for finite difference (optional, set to 1e-6)

    Output:
    ------
        J : float or tf.Tensor, numerical derivative of fun at x
    """
    
    if isinstance(x, tf.Tensor):
        x = tf.cast(x, dtype=tf.float64)
    else:
        x = tf.convert_to_tensor(x, dtype=tf.float64)

    if not tf.math.is_finite(x):
        raise ValueError("Jacobian: x not finite")

    if tol <= 0:
        raise ValueError("Jacobian: tol must be positive")

    f1 = tf.convert_to_tensor(fun(x + tol), dtype=tf.float64)
    f2 = tf.convert_to_tensor(fun(x - tol), dtype=tf.float64)

    if not tf.math.is_finite(f1) or not tf.math.is_finite(f2):
        raise ValueError("Jacobian: function returned non-finite values")

    J = (f1 - f2) / (2.0 * tol)

    if not tf.math.is_finite(J):
        raise ValueError("Jacobian: non-finite derivative")

    if tf.abs(J) < 1e-8:
        raise ValueError(f"Jacobian numerically zero at x = {x.numpy()}")

    return J



def EKF_UKF_univariate(
    y,
    f_fun,
    h_fun,
    sigma_eta,
    sigma_e,
    m0=0.0,
    P0=1.0,
    method="EKF",
    return_diagnostics=False,
    stop_on_failure=True,
    x_tol=0.05,              # chosen "bad geometry" threshold
    crosscov_tol=1e-10,      # UKF sigma-point degeneracy threshold
    jf_fun=None,
    jh_fun=None
):
    """
    Univariate EKF / UKF with diagnostics.

    Preferred bad-geometry diagnostic:
        abs(mu_pred_t) < x_tol

    For UKF, we also track sigma-point degeneracy:
        abs(C_xy) < crosscov_tol

    Parameters
    ----------
    y : 1D array-like
        Observations
    f_fun : callable
        State transition function
    h_fun : callable
        Observation function
    sigma_eta : float
        State noise std
    sigma_e : float
        Observation noise std
    m0 : float
        Initial mean
    P0 : float
        Initial variance
    method : {"EKF","UKF"}
    return_diagnostics : bool
        If True, return filtering diagnostics
    stop_on_failure : bool
        If True, stop when a method-specific degeneracy is detected
    x_tol : float
        Threshold defining the bad-geometry region around zero
    crosscov_tol : float
        Threshold for UKF cross-covariance degeneracy
    jf_fun : callable or None
        Explicit derivative of f_fun, if available
    jh_fun : callable or None
        Explicit derivative of h_fun, if available
    """

    # ---------- Method check ----------
    method = method.upper()
    if method not in ("EKF", "UKF"):
        raise ValueError("method must be 'EKF' or 'UKF'")

    # ---------- Convert inputs ----------
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    m0 = tf.convert_to_tensor(m0, dtype=tf.float64)
    P0 = tf.convert_to_tensor(P0, dtype=tf.float64)
    sigma_eta = tf.convert_to_tensor(sigma_eta, dtype=tf.float64)
    sigma_e = tf.convert_to_tensor(sigma_e, dtype=tf.float64)
    x_tol = tf.convert_to_tensor(x_tol, dtype=tf.float64)
    crosscov_tol = tf.convert_to_tensor(crosscov_tol, dtype=tf.float64)

    # ---------- Checks ----------
    if y.shape.ndims != 1 or tf.size(y) < 2:
        raise ValueError("y must be a 1D vector of length >= 2")
    if not tf.reduce_all(tf.math.is_finite(y)):
        raise ValueError("the values of y must be finite")

    for val, name in [(m0, "m0"), (P0, "P0"), (sigma_eta, "sigma_eta"), (sigma_e, "sigma_e")]:
        if not tf.reduce_all(tf.math.is_finite(val)):
            raise ValueError(f"{name} must be finite")
        if val.shape.ndims != 0:
            raise ValueError(f"{name} must be a scalar")

    if P0 <= 0 or sigma_eta <= 0 or sigma_e <= 0:
        raise ValueError("P0, sigma_eta, and sigma_e must be positive")

    if not callable(f_fun):
        raise ValueError("f_fun must be callable")
    if not callable(h_fun):
        raise ValueError("h_fun must be callable")

    n = tf.shape(y)[0]
    pi_tf = tf.constant(math.pi, dtype=tf.float64)

    # ---------- Allocation ----------
    mu_pred = tf.Variable(tf.zeros(n, dtype=tf.float64))
    mu_filt = tf.Variable(tf.zeros(n, dtype=tf.float64))
    P_pred  = tf.Variable(tf.zeros(n, dtype=tf.float64))
    P_filt  = tf.Variable(tf.zeros(n, dtype=tf.float64))
    v       = tf.Variable(tf.zeros(n, dtype=tf.float64))
    llk     = tf.Variable(0.0, dtype=tf.float64)

    # diagnostics
    Jh_vec = tf.Variable(tf.fill([n], tf.constant(np.nan, dtype=tf.float64)))   # EKF
    K_vec  = tf.Variable(tf.fill([n], tf.constant(np.nan, dtype=tf.float64)))
    S_vec  = tf.Variable(tf.fill([n], tf.constant(np.nan, dtype=tf.float64)))
    Cxy_vec = tf.Variable(tf.fill([n], tf.constant(np.nan, dtype=tf.float64)))  # UKF
    mu_y_vec = tf.Variable(tf.fill([n], tf.constant(np.nan, dtype=tf.float64))) # UKF

    # flags
    bad_geom_flag = tf.Variable(tf.zeros(n, dtype=tf.int32))   # |mu_pred| < x_tol
    bad_sigma_flag = tf.Variable(tf.zeros(n, dtype=tf.int32))  # UKF: |C_xy| small
    bad_flag = tf.Variable(tf.zeros(n, dtype=tf.int32))        # method-specific summary

    # ---------- Initialization ----------
    mu_filt[0].assign(m0)
    P_filt[0].assign(P0)
    mu_pred[0].assign(m0)
    P_pred[0].assign(P0)
    v[0].assign(tf.constant(0.0, dtype=tf.float64))

    t_indices = tf.range(1, n, dtype=tf.int32)

    # ---------- Recursion ----------
    for t in t_indices:
        # ----- Prediction -----
        if method == "EKF":
            if jf_fun is not None:
                Jf = jf_fun(mu_filt[t - 1])
            else:
#                Jf = compute_jacobian_tf(f_fun, tf.identity(mu_filt[t - 1]), tol=0.0)
                Jf = compute_jacobian_tf(f_fun, tf.identity(mu_filt[t - 1]), tol=1e-6)

            mu_pred_t = f_fun(mu_filt[t - 1])
            P_pred_t = Jf**2 * P_filt[t - 1] + sigma_eta**2

        else:  # UKF
            sp = compute_sigma_points_tf(mu_filt[t - 1], P_filt[t - 1])
            X, Wm, Wc = sp["X"], sp["Wm"], sp["Wc"]

            X_pred = tf.stack([f_fun(xi) for xi in X])

            if not tf.reduce_all(tf.math.is_finite(X_pred)):
                raise ValueError(f"Non-finite prediction at t={t}")

            mu_pred_t = tf.reduce_sum(Wm * X_pred)
            P_pred_t = sigma_eta**2 + tf.reduce_sum(Wc * (X_pred - mu_pred_t)**2)

        if not tf.math.is_finite(P_pred_t) or P_pred_t <= 0:
            raise ValueError(f"Invalid P_pred at t={t}")

        mu_pred[t].assign(mu_pred_t)
        P_pred[t].assign(P_pred_t)

        # ----- Common bad geometry diagnostic -----
        geom_t = tf.cast(tf.abs(mu_pred_t) < x_tol, tf.int32)
        bad_geom_flag[t].assign(geom_t)

        # ----- Update -----
        if method == "EKF":
            if jh_fun is not None:
                Jh = jh_fun(mu_pred_t)
            else:
#                Jh = compute_jacobian_tf(h_fun, tf.identity(mu_pred_t), tol=0.0)
                Jh = compute_jacobian_tf(h_fun, tf.identity(mu_pred_t), tol=1e-6)

            Jh_vec[t].assign(Jh)

            # For EKF, bad flag is the geometry flag
            bad_flag[t].assign(geom_t)

            if stop_on_failure and geom_t == 1:
                raise ValueError(f"EKF bad geometry detected at t={t}: |mu_pred| < x_tol")

            v_t = y[t] - h_fun(mu_pred_t)
            S_t = Jh**2 * P_pred_t + sigma_e**2

            if not tf.math.is_finite(S_t) or S_t <= 0:
                raise ValueError(f"Invalid S_t at t={t}")

            K = P_pred_t * Jh / S_t
            mu_t_filt = mu_pred_t + K * v_t
            P_t_filt = (1.0 - K * Jh) * P_pred_t

        else:  # UKF
            sp = compute_sigma_points_tf(mu_pred_t, P_pred_t)
            X, Wm, Wc = sp["X"], sp["Wm"], sp["Wc"]

            Y = tf.stack([h_fun(xi) for xi in X])
            if not tf.reduce_all(tf.math.is_finite(Y)):
                raise ValueError(f"Non-finite observation mapping at t={t}")

            mu_y = tf.reduce_sum(Wm * Y)
            mu_y_vec[t].assign(mu_y)

            S_t = sigma_e**2 + tf.reduce_sum(Wc * (Y - mu_y)**2)
            if not tf.math.is_finite(S_t) or S_t <= 0:
                raise ValueError(f"Invalid S_t at t={t}")

            C_xy = tf.reduce_sum(Wc * (X - mu_pred_t) * (Y - mu_y))
            Cxy_vec[t].assign(C_xy)

            sigma_t = tf.cast(tf.abs(C_xy) < crosscov_tol, tf.int32)
            bad_sigma_flag[t].assign(sigma_t)

            # For UKF, define bad flag as union of bad geometry and sigma-point degeneracy
            bad_t = tf.cast(tf.logical_or(tf.cast(geom_t, tf.bool), tf.cast(sigma_t, tf.bool)), tf.int32)
            bad_flag[t].assign(bad_t)

            if stop_on_failure and sigma_t == 1:
                raise ValueError(f"UKF sigma-point degeneracy detected at t={t}: |C_xy| < crosscov_tol")

            K = C_xy / S_t
            v_t = y[t] - mu_y
            mu_t_filt = mu_pred_t + K * v_t
            P_t_filt = P_pred_t - K**2 * S_t

        # ----- State checks -----
        if not tf.math.is_finite(mu_t_filt) or not tf.math.is_finite(P_t_filt):
            raise ValueError(f"Invalid filtered state at t={t}")

        P_t_filt = tf.maximum(P_t_filt, tf.constant(1e-12, dtype=tf.float64))

        mu_filt[t].assign(mu_t_filt)
        P_filt[t].assign(P_t_filt)
        v[t].assign(v_t)
        K_vec[t].assign(K)
        S_vec[t].assign(S_t)

        # ----- Likelihood -----
        llk_inc = -0.5 * (tf.math.log(2.0 * pi_tf * S_t) + v_t**2 / S_t)
        if not tf.math.is_finite(llk_inc):
            raise ValueError(f"Invalid log-likelihood increment at t={t}")

        llk.assign_add(llk_inc)

    out = {
        "mu_filt": mu_filt,
        "P_filt": P_filt,
        "mu_pred": mu_pred,
        "P_pred": P_pred,
        "llk": llk
    }

    if return_diagnostics:
        out["diagnostics"] = {
            "Jh": Jh_vec,
            "K": K_vec,
            "S": S_vec,
            "v": v,
            "C_xy": Cxy_vec,
            "mu_y": mu_y_vec,
            "bad_geom_flag": bad_geom_flag,
            "bad_sigma_flag": bad_sigma_flag,
            "bad_flag": bad_flag,
            "frac_bad_geom": tf.reduce_mean(tf.cast(bad_geom_flag, tf.float64)),
            "frac_bad_sigma": tf.reduce_mean(tf.cast(bad_sigma_flag, tf.float64)),
            "frac_bad": tf.reduce_mean(tf.cast(bad_flag, tf.float64)),
            "n_bad_geom": tf.reduce_sum(bad_geom_flag),
            "n_bad_sigma": tf.reduce_sum(bad_sigma_flag),
            "n_bad": tf.reduce_sum(bad_flag)
        }

    return out



#### SIMULATE MODEL 
def simulate_quadratic_model(T, phi, sigma_eta, sigma_e, dtype=tf.float64, seed=42):
    tf.random.set_seed(seed)
    # float64 is numerically safer for EKF/UKF diagnostics

    x0 = tf.random.normal([], mean=0.0, stddev=1.0, dtype=dtype)
    x = tf.TensorArray(tf.float64, size=T)
    x = x.write(0, x0)

    for t in range(1, T):
        eta_t = tf.random.normal([], mean=0.0, stddev=sigma_eta, dtype=dtype)
        x_t = phi * x.read(t - 1) + eta_t
        x = x.write(t, x_t)

    x = x.stack()
    y = x**2 / 5.0 + tf.random.normal([T], mean=0.0, stddev=sigma_e, dtype=dtype)

    return {"x": x, "y": y}


def make_quadratic_model_components(phi):
    def f_fun_tf(x):
        return phi * x

    def h_fun_tf(x):
        return x**2 / 5.0

    def jf_fun_tf(x):
        return tf.constant(phi, dtype=tf.float64)

    def jh_fun_tf(x):
        return 2.0 * x / 5.0

    return {
        "f_fun": f_fun_tf,
        "h_fun": h_fun_tf,
        "jf_fun": jf_fun_tf,
        "jh_fun": jh_fun_tf
    }


#### FUNCTIONS FOR THE EXPERIMENT

def run_heatmap_analysis(
    simulator_fun,
    phi_grid,
    sigma_e_grid,
    sigma_eta=0.2,
    T=100,
    m0=0.0,
    P0=1.0,
    x_tol=0.05,
    crosscov_tol=1e-10,
    seed=42
):
    rows = []
    diagnostics_store = {}

    for phi in phi_grid:
        components = make_quadratic_model_components(phi)

        for sigma_e in sigma_e_grid:
            sim_out = simulator_fun(T=T, phi=phi, sigma_eta=sigma_eta, sigma_e=sigma_e, seed=seed)

            x = sim_out["x"]
            y = sim_out["y"]

            res_ekf = EKF_UKF_univariate(
                y=y,
                f_fun=components["f_fun"],
                h_fun=components["h_fun"],
                sigma_eta=sigma_eta,
                sigma_e=sigma_e,
                m0=tf.constant(m0, dtype=tf.float64),
                P0=tf.constant(P0, dtype=tf.float64),
                method="EKF",
                return_diagnostics=True,
                stop_on_failure=False,
                x_tol=x_tol,
                jf_fun=components["jf_fun"],
                jh_fun=components["jh_fun"]
            )

            res_ukf = EKF_UKF_univariate(
                y=y,
                f_fun=components["f_fun"],
                h_fun=components["h_fun"],
                sigma_eta=sigma_eta,
                sigma_e=sigma_e,
                m0=tf.constant(m0, dtype=tf.float64),
                P0=tf.constant(P0, dtype=tf.float64),
                method="UKF",
                return_diagnostics=True,
                stop_on_failure=False,
                x_tol=x_tol,
                crosscov_tol=crosscov_tol
            )

            rmse_ekf = tf.sqrt(tf.reduce_mean((res_ekf["mu_filt"] - x)**2)).numpy()
            rmse_ukf = tf.sqrt(tf.reduce_mean((res_ukf["mu_filt"] - x)**2)).numpy()

            rows.append({
                "phi": phi,
                "sigma_e": sigma_e,
                "ekf_frac_bad_geom": float(res_ekf["diagnostics"]["frac_bad_geom"].numpy()),
                "ukf_frac_bad_geom": float(res_ukf["diagnostics"]["frac_bad_geom"].numpy()),
                "ukf_frac_bad_sigma": float(res_ukf["diagnostics"]["frac_bad_sigma"].numpy()),
                "ukf_frac_bad_total": float(res_ukf["diagnostics"]["frac_bad"].numpy()),
                "ekf_rmse": float(rmse_ekf),
                "ukf_rmse": float(rmse_ukf),
                "ekf_loglik": float(res_ekf["llk"].numpy()),
                "ukf_loglik": float(res_ukf["llk"].numpy())
            })

            diagnostics_store[(phi, sigma_e)] = {
                "x": x,
                "y": y,
                "ekf": res_ekf,
                "ukf": res_ukf
            }

    df = pd.DataFrame(rows)
    return df, diagnostics_store

def make_color_cmap():
    return LinearSegmentedColormap.from_list(
        "green_orange_red",
        ["#2E8B57", "#F4A261", "#D62828"]
    )


def plot_heatmaps_quadratic(df):
    cmap = make_color_cmap()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Precompute pivots once
    piv = lambda col: df.pivot(index="phi", columns="sigma_e", values=col)

    heatmaps = [
        ("ekf_frac_bad_geom", "EKF: fraction in bad geometry", ".2f", cmap, (0, 1)),
        ("ukf_frac_bad_sigma", "UKF: sigma-point degeneracy", ".2f", cmap, (0, 1)),
        ("ukf_frac_bad_total", "UKF: total bad fraction", ".2f", cmap, (0, 1)),
        ("ekf_rmse", "EKF RMSE", ".3f", cmap, None),
        ("ukf_rmse", "UKF RMSE", ".3f", cmap, None),
    ]

    # Plot first 5 heatmaps
    for ax, (col, title, fmt, cm, lims) in zip(axes.flat[:5], heatmaps):
        data = piv(col)
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap=cm,
            vmin=lims[0] if lims else None,
            vmax=lims[1] if lims else None,
            ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel("sigma_eps")
        ax.set_ylabel("phi")

    # RMSE difference
    heat_diff = piv("ukf_rmse") - piv("ekf_rmse")
    sns.heatmap(heat_diff, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=axes.flat[5])

    axes.flat[5].set_title("UKF RMSE - EKF RMSE")
    axes.flat[5].set_xlabel("sigma_eps")
    axes.flat[5].set_ylabel("phi")

    plt.tight_layout()
    plt.show()


def print_summary_stats(df):
    cols = [
        "ekf_frac_bad_geom",
        "ukf_frac_bad_geom",
        "ukf_frac_bad_sigma",
        "ukf_frac_bad_total",
        "ekf_rmse",
        "ukf_rmse",
        "ekf_loglik",
        "ukf_loglik"
    ]
    print(df[cols].describe().round(4))


def plot_fraction_lollipops(df):
    metrics = [
        ("ekf_frac_bad_geom", "EKF bad geometry", "#2E8B57"),
        ("ukf_frac_bad_sigma", "UKF sigma-point degeneracy", "#F4A261"),
        ("ukf_frac_bad_total", "UKF total bad fraction", "#D62828"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharex=False)

    for ax, (col, title, color) in zip(axes, metrics):
        df_plot = df.copy()
        df_plot["config"] = df_plot.apply(
            lambda r: f"(phi={r['phi']}, sigma_eps={r['sigma_e']})", axis=1
        )
        df_plot = df_plot.sort_values(col).reset_index(drop=True)

        y = np.arange(len(df_plot))
        x = df_plot[col].to_numpy()

        ax.hlines(y=y, xmin=0, xmax=x, color=color, alpha=0.75, linewidth=2)
        ax.scatter(x, y, color=color, s=50, zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels(df_plot["config"])
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Fraction")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)

        ax.axvline(np.mean(x), color="black", linestyle="--", linewidth=1.2, label="mean")
        ax.axvline(np.median(x), color="black", linestyle=":", linewidth=1.2, label="median")
        ax.legend()

    plt.tight_layout()
    plt.show()

def find_bad_pair(diagnostics_store, phi_target, sigma_e_target, tol=1e-10):
    for key in diagnostics_store.keys():
        phi_k, sigma_e_k = key
        if abs(float(phi_k) - float(phi_target)) < tol and abs(float(sigma_e_k) - float(sigma_e_target)) < tol:
            return key
    raise KeyError(
        f"No bad pair found for (phi={phi_target}, sigma_e={sigma_e_target}). "
        f"Available pairs are: {list(diagnostics_store.keys())}"
    )



def plot_bad_acf(diagnostics_store, selected_configs, nlags=30):
    valid_keys = list(diagnostics_store.keys())

    print("\nAvailable parameter combinations:")
    for k in valid_keys:
        print(k)

    matched_keys = []
    for cfg in selected_configs:
        if cfg in diagnostics_store:
            matched_keys.append(cfg)
        else:
            print(f"\nSkipping {cfg}: not available in diagnostics_store")

    if len(matched_keys) == 0:
        raise ValueError("None of the selected configurations are available.")

    print("\nConfigurations used in the ACF plot:")
    for k in matched_keys:
        print(k)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_ekf = axes[0]
    ax_ukf = axes[1]

    conf_ekf_ref = None
    conf_ukf_ref = None

    for key in matched_keys:
        phi, sigma_e = key
        out = diagnostics_store[key]

        ekf_bad = out["ekf"]["diagnostics"]["bad_geom_flag"].numpy().astype(float)
        ukf_bad = out["ukf"]["diagnostics"]["bad_sigma_flag"].numpy().astype(float)

        ekf_n_bad = int(np.sum(ekf_bad))
        ukf_n_bad = int(np.sum(ukf_bad))

        print(f"\nConfiguration phi={phi}, sigma_e={sigma_e}")
        print(f"EKF: number of bad-geometry time points = {ekf_n_bad}")
        print(f"EKF: fraction in bad geometry = {np.mean(ekf_bad):.3f}")
        print(f"UKF: number of sigma-point degeneracy time points = {ukf_n_bad}")
        print(f"UKF: fraction in sigma-point degeneracy = {np.mean(ukf_bad):.3f}")

        ekf_acf = acf(ekf_bad, nlags=nlags, fft=True)
        ukf_acf = acf(ukf_bad, nlags=nlags, fft=True)

        lags = np.arange(len(ekf_acf))

        conf_ekf_ref = 1.96 / np.sqrt(len(ekf_bad))
        conf_ukf_ref = 1.96 / np.sqrt(len(ukf_bad))

        ax_ekf.vlines(lags, 0, ekf_acf, linewidth=2, alpha=0.9, label=f"phi={phi}, σe={sigma_e}")
        ax_ekf.plot(lags, ekf_acf, "o", markersize=3)

        ax_ukf.vlines(lags, 0, ukf_acf, linewidth=2, alpha=0.9, label=f"phi={phi}, σe={sigma_e}")
        ax_ukf.plot(lags, ukf_acf, "o", markersize=3)

    ax_ekf.axhline(0, color="black", linewidth=1)
    ax_ukf.axhline(0, color="black", linewidth=1)

    if conf_ekf_ref is not None:
        ax_ekf.axhline(conf_ekf_ref, color="black", linestyle="--", linewidth=1)
        ax_ekf.axhline(-conf_ekf_ref, color="black", linestyle="--", linewidth=1)

    if conf_ukf_ref is not None:
        ax_ukf.axhline(conf_ukf_ref, color="black", linestyle="--", linewidth=1)
        ax_ukf.axhline(-conf_ukf_ref, color="black", linestyle="--", linewidth=1)

    ax_ekf.set_title("ACF of EKF bad-geometry indicator")
    ax_ukf.set_title("ACF of UKF sigma-point degeneracy indicator")

    for ax in axes:
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Selected configurations: " +
        ", ".join([f"(phi={phi}, sigma_e={sigma_e})" for phi, sigma_e in matched_keys]),
        fontsize=10
    )

    plt.tight_layout()
    plt.show()



def plot_bad_pacf(
    diagnostics_store,
    selected_configs,
    nlags=30,
    save=False,
    save_dir="figures",
    filename="bad_pacf.png",
    dpi=300,
    close_after_save=False,
):
    valid_keys = list(diagnostics_store.keys())

    print("\nAvailable parameter combinations:")
    for k in valid_keys:
        print(k)

    matched_keys = []
    for cfg in selected_configs:
        if cfg in diagnostics_store:
            matched_keys.append(cfg)
        else:
            print(f"\nSkipping {cfg}: not available in diagnostics_store")

    if len(matched_keys) == 0:
        raise ValueError("None of the selected configurations are available.")

    print("\nConfigurations used in the PACF plot:")
    for k in matched_keys:
        print(k)

    # bright colors for dots
    colors = sns.color_palette("bright", len(matched_keys))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_ekf, ax_ukf = axes

    # neutral bar color
    bar_color = "#B0BEC5"   # light blue-grey

    conf = None  # will set from sample size below

    for idx, key in enumerate(matched_keys):
        phi, sigma_e = key
        color = colors[idx]

        out = diagnostics_store[key]

        ekf_bad = out["ekf"]["diagnostics"]["bad_geom_flag"].numpy().astype(float)
        ukf_bad = out["ukf"]["diagnostics"]["bad_sigma_flag"].numpy().astype(float)

        ekf_n_bad = int(np.sum(ekf_bad))
        ukf_n_bad = int(np.sum(ukf_bad))

        print(f"\nConfiguration phi={phi}, sigma_e={sigma_e}")
        print(f"EKF: number of bad-geometry time points = {ekf_n_bad}")
        print(f"EKF: fraction in bad geometry = {np.mean(ekf_bad):.3f}")
        print(f"UKF: number of sigma-point degeneracy time points = {ukf_n_bad}")
        print(f"UKF: fraction in sigma-point degeneracy = {np.mean(ukf_bad):.3f}")

        # PACF
        # method="ywm" is usually a stable default
        ekf_pacf = pacf(ekf_bad, nlags=nlags, method="ywm")
        ukf_pacf = pacf(ukf_bad, nlags=nlags, method="ywm")

        lags = np.arange(len(ekf_pacf))

        conf = 1.96 / np.sqrt(len(ekf_bad))

        # --- EKF ---
        ax_ekf.vlines(
            lags, 0, ekf_pacf,
            color=bar_color,
            linewidth=2,
            alpha=0.6,
            zorder=1
        )

        ax_ekf.scatter(
            lags, ekf_pacf,
            s=30,
            color=color,
            label=f"(φ={phi}, σe={sigma_e})",
            zorder=3
        )

        # --- UKF ---
        ax_ukf.vlines(
            lags, 0, ukf_pacf,
            color=bar_color,
            linewidth=2,
            alpha=0.6,
            zorder=1
        )

        ax_ukf.scatter(
            lags, ukf_pacf,
            s=30,
            color=color,
            label=f"(φ={phi}, σe={sigma_e})",
            zorder=3
        )

    # zero line
    ax_ekf.axhline(0, color="black", linewidth=1)
    ax_ukf.axhline(0, color="black", linewidth=1)

    # CI
    if conf is not None:
        ax_ekf.axhline(conf, color="black", linestyle="--", linewidth=1.2)
        ax_ekf.axhline(-conf, color="black", linestyle="--", linewidth=1.2)

        ax_ukf.axhline(conf, color="black", linestyle="--", linewidth=1.2)
        ax_ukf.axhline(-conf, color="black", linestyle="--", linewidth=1.2)

    # titles
    ax_ekf.set_title("PACF of the EKF bad-geometry indicator")
    ax_ukf.set_title("PACF of UKF sigma-point degeneracy indicator")

    for ax in axes:
        ax.set_xlabel("Lag")
        ax.set_ylabel("PACF Value")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle("PACF across parameter configurations", fontsize=11)

    plt.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"\nSaved figure to: {save_path}")

    plt.show()

    if close_after_save:
        plt.close(fig)

    return fig, axes
