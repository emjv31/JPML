import time
import threading
import psutil

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from multiv_sv_bpf_core import (
    SV_model_sim_tf_h,
    make_prop_sv,
    make_loglik_sv,
    multinomial_resampling,
    bpf_generic_resampling,
    _as_transition_matrix_tf,
    _as_noise_chol_tf,
    _as_vector_param_tf,
)
from multiv_ekf_ukf_core import (
    _filter_core,
    make_ekf_kernels,
    make_ukf_kernels,
)



def compute_bias_rmse(h_true, est_mc):
    """
    h_true:  (T,) or (T,d)
    est_mc:  (R,T) or (R,T,d)
    """
    h_true = tf.cast(tf.convert_to_tensor(h_true), tf.float64)
    est_mc = tf.cast(tf.convert_to_tensor(est_mc), tf.float64)

    if h_true.shape.rank == 1:
        h_true = tf.expand_dims(h_true, axis=-1)
    if est_mc.shape.rank == 2:
        est_mc = tf.expand_dims(est_mc, axis=-1)

    bias = tf.reduce_mean(est_mc, axis=0) - h_true
    rmse = tf.sqrt(tf.reduce_mean((est_mc - h_true) ** 2, axis=0))

    return bias, rmse



def benchmark_cpu(func, *args, sample_interval=0.01, **kwargs):
    import time
    import threading
    import psutil

    process = psutil.Process()

    rss_before = process.memory_info().rss
    peak_rss = {"value": rss_before}
    stop_event = threading.Event()

    def monitor():
        peak = rss_before
        while not stop_event.is_set():
            rss = process.memory_info().rss
            if rss > peak:
                peak = rss
            time.sleep(sample_interval)
        peak_rss["value"] = peak

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    t0 = time.perf_counter()
    out = func(*args, **kwargs)
    t1 = time.perf_counter()

    stop_event.set()
    thread.join()

    rss_after = process.memory_info().rss

    stats = {
        "runtime_seconds": t1 - t0,
        "rss_before_mb": rss_before / (1024**2),
        "rss_after_mb": rss_after / (1024**2),
        "rss_peak_mb": peak_rss["value"] / (1024**2),
        "rss_delta_mb": (rss_after - rss_before) / (1024**2),
        "rss_peak_increase_mb": (peak_rss["value"] - rss_before) / (1024**2),
    }

    return out, stats


def run_kf_experiments_multivariate(
    y_tf,
    h_tf,
    phi,
    sigma_eps,
    sigma_eta,
    xi,
    dtype=tf.float64,
    m0=None,
    P0=None,
    alpha=1e-3,
    beta=2.0,
    kappa=0.0,
    eps_floor=1e-12,
    methods=None,
):
    """
    Multivariate EKF/UKF experiments for the SV model.

    Parameters
    ----------
    y_tf : tensor, shape (T,d)
    h_tf : tensor, shape (T,d)
    phi, sigma_eps, sigma_eta, xi : model parameters
    methods : list or None
        Subset of methods to run. Allowed values:
        - "EKF_misspec"
        - "UKF_misspec"
        - "EKF_correct"
        - "UKF_correct"

        If None, all four are run.

    Returns
    -------
    results_methods : dict
        Dictionary keyed by method name.
    """
    results_methods = {}

    if methods is None:
        methods = ["EKF_misspec", "UKF_misspec", "EKF_correct", "UKF_correct"]

    y_tf = tf.cast(y_tf, dtype)
    h_tf = tf.cast(h_tf, dtype)

    if len(y_tf.shape) != 2:
        raise ValueError("y_tf must be 2D, shape (T,d)")
    if len(h_tf.shape) != 2:
        raise ValueError("h_tf must be 2D, shape (T,d)")

    d = int(y_tf.shape[1])

    Y = tf.transpose(y_tf)   # (d,T)
    H_true = h_tf            # (T,d)

    Phi = _as_transition_matrix_tf(phi, d, dtype) # _as_transition_matrix

    L_eta = _as_noise_chol_tf(sigma_eta, d, dtype, "sigma_eta")
    Q = L_eta @ tf.transpose(L_eta)

    sigma_eps_vec = _as_vector_param_tf(sigma_eps, d, dtype, "sigma_eps") # _as_vector_param
    xi_vec = _as_vector_param_tf(xi, d, dtype, "xi") # _as_vector_param

    if m0 is None:
        m0 = tf.zeros([d], dtype=dtype)
    else:
        m0 = tf.cast(m0, dtype)

    if P0 is None:
        P0 = tf.eye(d, dtype=dtype)
    else:
        P0 = tf.cast(P0, dtype)

    # ============================================================
    # MISSPECIFIED CASE
    # ============================================================
    def F_func_mis(x, t):
        return tf.linalg.matvec(Phi, x)

    def F_jac_mis(x, t):
        return Phi

    def H_func_mis(x, t):
        return xi_vec * tf.exp(x / 2.0)

    def H_jac_mis(x, t):
        return tf.linalg.diag(0.5 * xi_vec * tf.exp(x / 2.0))

    R_mis = tf.linalg.diag(sigma_eps_vec**2)

    for method in ["EKF", "UKF"]:
        method_key = f"{method}_misspec"
        if method_key not in methods:
            continue

        print(f"  {method} (misspecified, multivariate)")

        if method == "EKF":
            predict_fn, update_fn = make_ekf_kernels(
                F_func=F_func_mis,
                H_func=H_func_mis,
                F_jac=F_jac_mis,
                H_jac=H_jac_mis,
                Q=Q
            )
        else:
            predict_fn, update_fn = make_ukf_kernels(
                F_func=F_func_mis,
                H_func=H_func_mis,
                Q=Q,
                alpha=alpha,
                beta=beta,
                kappa=kappa
            )

        out = _filter_core(
            Y=Y,
            predict_fn=predict_fn,
            update_fn=update_fn,
            R_mat=R_mis,
            m0=m0,
            P0=P0,
            measurement_type="gaussian",
            dtype=dtype
        )

        mu_filt = tf.transpose(out["mu_filt"])       # (T,d)
        mu_filt_mc = tf.expand_dims(mu_filt, axis=0) # (1,T,d)

        bias, rmse = compute_bias_rmse(H_true, mu_filt_mc)

        results_methods[method_key] = {
            "bias": bias,
            "rmse": rmse,
            "mu_filt": mu_filt,
            "P_filt": out["P_filt"],
            "P_pred": out["P_pred"],
            "loglik": out["loglik"]
        }

    # ============================================================
    # CORRECT CASE
    # ============================================================
    z_tf = tf.math.log(tf.maximum(y_tf**2, tf.constant(eps_floor, dtype=dtype)))
    Z = tf.transpose(z_tf)

    c_vec = tf.math.log((xi_vec**2) * (sigma_eps_vec**2)) - tf.constant(1.2704, dtype=dtype)

    def F_func_corr(x, t):
        return tf.linalg.matvec(Phi, x)

    def F_jac_corr(x, t):
        return Phi

    def H_func_corr(x, t):
        return x + c_vec

    def H_jac_corr(x, t):
        return tf.eye(d, dtype=dtype)

    R_corr = tf.eye(d, dtype=dtype) * tf.constant(np.pi**2 / 2.0, dtype=dtype)

    for method in ["EKF", "UKF"]:
        method_key = f"{method}_correct"
        if method_key not in methods:
            continue

        print(f"  {method} (correct, multivariate)")

        if method == "EKF":
            predict_fn, update_fn = make_ekf_kernels(
                F_func=F_func_corr,
                H_func=H_func_corr,
                F_jac=F_jac_corr,
                H_jac=H_jac_corr,
                Q=Q
            )
        else:
            predict_fn, update_fn = make_ukf_kernels(
                F_func=F_func_corr,
                H_func=H_func_corr,
                Q=Q,
                alpha=alpha,
                beta=beta,
                kappa=kappa
            )

        out = _filter_core(
            Y=Z,
            predict_fn=predict_fn,
            update_fn=update_fn,
            R_mat=R_corr,
            m0=m0,
            P0=P0,
            measurement_type="gaussian",
            dtype=dtype
        )

        mu_filt = tf.transpose(out["mu_filt"])       # (T,d)
        mu_filt_mc = tf.expand_dims(mu_filt, axis=0) # (1,T,d)

        bias, rmse = compute_bias_rmse(H_true, mu_filt_mc)

        results_methods[method_key] = {
            "bias": bias,
            "rmse": rmse,
            "mu_filt": mu_filt,
            "P_filt": out["P_filt"],
            "P_pred": out["P_pred"],
            "loglik": out["loglik"]
        }

    return results_methods


### BPF FOR EXECUTION
@tf.function(reduce_retracing=True)
def _run_bpf_mc_for_fixed_N(
    y_tf,
    h_tf,
    N,
    R,
    prop_fn,
    log_likelihood_fn,
    degeneracy_threshold,
    resampling_fn,
    resample_threshold,
    carry_resampled_weights,
    dtype
):
    """
    Run R Monte Carlo replications of the BPF for one fixed particle count N.

    Returns a dict of tensors.
    """
    y_tf = tf.cast(y_tf, dtype)
    h_tf = tf.cast(h_tf, tf.float64)

    if h_tf.shape.rank == 1:
        h_tf = tf.expand_dims(h_tf, axis=-1)

    T = tf.shape(y_tf)[0]
    d = tf.shape(y_tf)[1]

    est_ta = tf.TensorArray(tf.float64, size=R, infer_shape=False)
    ess_ta = tf.TensorArray(tf.float64, size=R, infer_shape=False)
    loglik_ta = tf.TensorArray(tf.float64, size=R)

    for r in tf.range(R):
        ests, ESSs, total_loglik = bpf_generic_resampling(
            Y=y_tf,
            Np=N,
            prop_fn=prop_fn,
            log_likelihood_fn=log_likelihood_fn,
            resampling_fn=resampling_fn,
            resample_threshold=resample_threshold,
            dtype=dtype,
            carry_resampled_weights=carry_resampled_weights,
        )

        est_ta = est_ta.write(r, tf.cast(ests, tf.float64))     # (T,d)
        ess_ta = ess_ta.write(r, tf.cast(ESSs, tf.float64))     # (T,)
        loglik_ta = loglik_ta.write(r, tf.cast(total_loglik, tf.float64))

    part_est_mc = est_ta.stack()      # (R,T,d)
    ESS_mc = ess_ta.stack()           # (R,T)
    loglik_mc = loglik_ta.stack()     # (R,)

    ESS_norm_mc = ESS_mc / tf.cast(N, tf.float64)

    ESS_mean_t = tf.reduce_mean(ESS_mc, axis=0)
    ESS_min_t = tf.reduce_min(ESS_mc, axis=0)

    ESS_norm_mean_t = tf.reduce_mean(ESS_norm_mc, axis=0)
    ESS_norm_min_t = tf.reduce_min(ESS_norm_mc, axis=0)

    bias, rmse = compute_bias_rmse(h_tf, part_est_mc)

    ess_mean = tf.reduce_mean(ESS_norm_mc)
    ess_min = tf.reduce_min(ESS_norm_mc)
    ess_frac_below = tf.reduce_mean(
        tf.cast(ESS_norm_mc < tf.cast(degeneracy_threshold, tf.float64), tf.float64)
    )
    ess_n_below = tf.reduce_sum(
        tf.cast(ESS_norm_mc < tf.cast(degeneracy_threshold, tf.float64), tf.int32)
    )
    loglik_mean = tf.reduce_mean(loglik_mc)

    return {
        "ESS_mean_t": ESS_mean_t,
        "ESS_min_t": ESS_min_t,
        "ESS_norm_mean_t": ESS_norm_mean_t,
        "ESS_norm_min_t": ESS_norm_min_t,
        "ESS_mc": ESS_mc,
        "ESS_norm_mc": ESS_norm_mc,
        "part_est_mc": part_est_mc,
        "loglikelihood_mc": loglik_mc,
        "bias": bias,
        "rmse": rmse,
        "ess_mean": ess_mean,
        "ess_min": ess_min,
        "ess_frac_below": ess_frac_below,
        "ess_n_below": ess_n_below,
        "loglik_mean": loglik_mean,
    }


def run_bpf_sv_experiments(
    phi,
    sigma_eta,
    sigma_eps,
    xi,
    Np,
    R,
    T=100,
    d=1,
    seed=123,
    degeneracy_threshold=0.2,
    resampling_fn=multinomial_resampling,
    resample_threshold=False,
    carry_resampled_weights=False,
    dtype=tf.float32
):
    if R <= 0:
        raise ValueError("R must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")
    if d <= 0:
        raise ValueError("d must be > 0")

    # --------------------------------------------------
    # STEP 1: simulate data
    # --------------------------------------------------
    sim_out = SV_model_sim_tf_h(
        iT=T,
        phi=phi,
        sigma_eta=sigma_eta,
        sigma_eps=sigma_eps,
        xi=xi,
        seed=seed,
        d=d,
        dtype=tf.float64
    )

    y_tf = tf.cast(sim_out["vY"], dtype)
    h_tf = tf.cast(sim_out["h"], tf.float64)

    if y_tf.shape.rank == 1:
        y_tf = tf.expand_dims(y_tf, axis=-1)
    if h_tf.shape.rank == 1:
        h_tf = tf.expand_dims(h_tf, axis=-1)

    # --------------------------------------------------
    # STEP 2: build filter components
    # --------------------------------------------------
    prop_fn = make_prop_sv(phi=phi, sigma_eta=sigma_eta, dtype=dtype)
    loglik_fn = make_loglik_sv(sigma_eps=sigma_eps, xi=xi, dtype=dtype)

    # --------------------------------------------------
    # STEP 3: run experiments for each N
    # --------------------------------------------------
    results_BPF = {}
    metrics = {}

    for N in Np:
        print(f"  N = {N}")

        outN = _run_bpf_mc_for_fixed_N(
            y_tf=y_tf,
            h_tf=h_tf,
            N=int(N),
            R=int(R),
            prop_fn=prop_fn,
            log_likelihood_fn=loglik_fn,
            degeneracy_threshold=tf.cast(degeneracy_threshold, tf.float64),
            resampling_fn=resampling_fn,
            resample_threshold=resample_threshold,
            carry_resampled_weights=carry_resampled_weights,
            dtype=dtype
        )

        results_BPF[N] = {
            "ESS_mean_t": outN["ESS_mean_t"],
            "ESS_min_t": outN["ESS_min_t"],
            "ESS_norm_mean_t": outN["ESS_norm_mean_t"],
            "ESS_norm_min_t": outN["ESS_norm_min_t"],
            "ESS_mc": outN["ESS_mc"],
            "ESS_norm_mc": outN["ESS_norm_mc"],
            "part_est_mc": outN["part_est_mc"],
            "loglikelihood_mc": outN["loglikelihood_mc"],
        }

        metrics[N] = {
            "bias": outN["bias"],
            "rmse": outN["rmse"],
            "ess_mean": outN["ess_mean"],
            "ess_min": outN["ess_min"],
            "ess_frac_below": outN["ess_frac_below"],
            "ess_n_below": outN["ess_n_below"],
            "loglik_mean": outN["loglik_mean"],
        }

    return {
        "sim": {
            "y_tf": y_tf,
            "h_tf": h_tf
        },
        "ESS": results_BPF,
        "metrics": metrics
    }



def compare_methods_one_config(
    d,
    phi,
    sigma_eps,
    sigma_eta=1.0,
    xi=1.0,
    Np=(5, 10, 20),
    T=10,
    N_MC=3,
    seed=123,
    dtype_bpf=tf.float32,
    dtype_kf=tf.float64,
    sample_interval=0.01,
):
    # ---------- simulate once ----------
    sim_out = SV_model_sim_tf_h(
        iT=T,
        phi=phi,
        sigma_eta=sigma_eta,
        sigma_eps=sigma_eps,
        xi=xi,
        seed=seed,
        d=d,
        dtype=tf.float64
    )

    y_tf = tf.cast(sim_out["vY"], tf.float64)
    h_tf = tf.cast(sim_out["h"], tf.float64)

    if y_tf.shape.rank == 1:
        y_tf = tf.expand_dims(y_tf, axis=-1)
    if h_tf.shape.rank == 1:
        h_tf = tf.expand_dims(h_tf, axis=-1)

    # ---------- BPF ----------
    prop_fn = make_prop_sv(phi=phi, sigma_eta=sigma_eta, dtype=dtype_bpf)
    loglik_fn = make_loglik_sv(sigma_eps=sigma_eps, xi=xi, dtype=dtype_bpf)

    ess_out = {}
    metrics_out = {}
    benchmark_out = {"BPF": {}, "KF": {}}

    for N in Np:
        outN, statsN = benchmark_cpu(
            _run_bpf_mc_for_fixed_N,
            y_tf=tf.cast(y_tf, dtype_bpf),
            h_tf=tf.cast(h_tf, tf.float64),
            N=int(N),
            R=int(N_MC),
            prop_fn=prop_fn,
            log_likelihood_fn=loglik_fn,
            degeneracy_threshold=tf.constant(0.2, tf.float64),
            resampling_fn=multinomial_resampling,
            resample_threshold=False,
            carry_resampled_weights=False,
            dtype=dtype_bpf,
            sample_interval=sample_interval,
        )

        ess_out[N] = {
            "ESS_mean_t": outN["ESS_mean_t"],
            "ESS_min_t": outN["ESS_min_t"],
            "ESS_norm_mean_t": outN["ESS_norm_mean_t"],
            "ESS_norm_min_t": outN["ESS_norm_min_t"],
#            "ESS_mc": outN["ESS_mc"],
#            "ESS_norm_mc": outN["ESS_norm_mc"],
#            "part_est_mc": outN["part_est_mc"],
#            "loglikelihood_mc": outN["loglikelihood_mc"],
        }

        metrics_out[N] = {
            "bias": outN["bias"],
            "rmse": outN["rmse"],
            "ess_mean": outN["ess_mean"],
            "ess_min": outN["ess_min"],
            "ess_frac_below": outN["ess_frac_below"],
            "ess_n_below": outN["ess_n_below"],
            "loglik_mean": outN["loglik_mean"],
        }

        benchmark_out["BPF"][N] = statsN

        print(
            f"BPF N={N} | runtime={statsN['runtime_seconds']:.4f}s | "
            f"peak+={statsN['rss_peak_increase_mb']:.2f} MB"
        )

    # ---------- EKF / UKF ----------
    methods_list = ["EKF_misspec", "UKF_misspec", "EKF_correct", "UKF_correct"]
    kf_out = {}

    for method_name in methods_list:
        out_method, stats_method = benchmark_cpu(
            run_kf_experiments_multivariate,
            y_tf=tf.cast(y_tf, dtype_kf),
            h_tf=tf.cast(h_tf, dtype_kf),
            phi=phi,
            sigma_eps=sigma_eps,
            sigma_eta=sigma_eta,
            xi=xi,
            dtype=dtype_kf,
            methods=[method_name],
            sample_interval=sample_interval,
        )

        kf_out.update(out_method)
        benchmark_out["KF"][method_name] = stats_method

        print(
            f"{method_name} | runtime={stats_method['runtime_seconds']:.4f}s | "
            f"peak+={stats_method['rss_peak_increase_mb']:.2f} MB"
        )

    return {
        "sim": {"y_tf": y_tf, "h_tf": h_tf},
        "ESS": ess_out,
        "metrics": metrics_out,
        "KF": kf_out,
        "benchmark": benchmark_out,
    }


def plot_benchmark_grouped(results_all):
    import matplotlib.pyplot as plt
    import numpy as np

    d_values = sorted({cfg[0] for cfg in results_all.keys()})

    first_cfg = next(iter(results_all))
    method_names = []

    for N in sorted(results_all[first_cfg]["benchmark"]["BPF"].keys()):
        method_names.append(f"BPF N={N}")

    for name in sorted(results_all[first_cfg]["benchmark"]["KF"].keys()):
        method_names.append(name)

    cpu_by_method = {m: [] for m in method_names}
    time_by_method = {m: [] for m in method_names}

    for d in d_values:
        cfgs_d = [cfg for cfg in results_all if cfg[0] == d]

        for m in method_names:
            cpu_vals = []
            time_vals = []

            for cfg in cfgs_d:
                bench = results_all[cfg]["benchmark"]

                if m.startswith("BPF N="):
                    N = int(m.replace("BPF N=", ""))
                    s = bench["BPF"][N]
                else:
                    s = bench["KF"][m]

                cpu_vals.append(s["rss_peak_mb"])
                time_vals.append(s["runtime_seconds"])

            cpu_by_method[m].append(np.mean(cpu_vals))
            time_by_method[m].append(np.mean(time_vals))

    x = np.arange(len(d_values))
    n_methods = len(method_names)
    width = 0.8 / n_methods

    # Plot style: 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(10, 10)) # 8, 10

    for j, m in enumerate(method_names):
        offset = (j - (n_methods - 1) / 2) * width

        axes[0].bar(x + offset, cpu_by_method[m], width=width, label=m)
        axes[1].bar(x + offset, time_by_method[m], width=width, label=m)

    # --- CPU ---
    axes[0].set_title("CPU peak memory")
    axes[0].set_ylabel("MB")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(d_values)
    axes[0].grid(axis="y", alpha=0.3)

    # --- Runtime ---
    axes[1].set_title("Runtime")
    axes[1].set_ylabel("seconds")
    axes[1].set_xlabel("State dimension d")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(d_values)
    axes[1].grid(axis="y", alpha=0.3)

    # legend on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=min(len(labels), 4), frameon=False)

    fig.subplots_adjust(top=0.88, hspace=0.3)

    plt.show()


def plot_ESS_over_time_bpf(
    results_all,
    phi_fixed,
    sigma_eps_fixed,
    d_list=None,
    N_bpf_list=None,
    use_min=False,
    figsize_per_row=5,
):
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import numpy as np

    configs = sorted(
        [
            (d, phi, sigma_eps)
            for (d, phi, sigma_eps) in results_all.keys()
            if phi == phi_fixed and sigma_eps == sigma_eps_fixed
        ],
        key=lambda x: x[0]
    )

    if not configs:
        raise ValueError(f"No configurations found for phi={phi_fixed}, sigma_eps={sigma_eps_fixed}")

    if d_list is None:
        d_list = [d for (d, _, _) in configs]

    d_list = sorted(d_list)

    if N_bpf_list is None:
        first_cfg = (d_list[0], phi_fixed, sigma_eps_fixed)
        N_bpf_list = sorted(results_all[first_cfg]["ESS"].keys())

    ess_key = "ESS_min_t" if use_min else "ESS_mean_t"

    palette = ["#00cc96", "#ffa600", "#ab63fa", "#19d3f3"]
    bpf_colors = {
        N: palette[i % len(palette)] for i, N in enumerate(sorted(N_bpf_list))
    }

    fig, axes = plt.subplots(
        len(d_list), 1,
        figsize=(8, figsize_per_row * len(d_list)),
        squeeze=False
    )

    for i, d in enumerate(d_list):
        ax = axes[i, 0]
        res = results_all[(d, phi_fixed, sigma_eps_fixed)]

        for N in N_bpf_list:
            y = tf.cast(res["ESS"][N][ess_key], tf.float64).numpy()
            x = np.arange(len(y))

            ax.plot(
                x, y,
                linewidth=2.5,
                color=bpf_colors[N],
                label=f"BPF N={N}"
            )

        ax.set_title(f"d = {d}", fontsize=12)
        ax.set_xlabel("Time")
        ax.set_ylabel("ESS")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9, ncol=2)

    title_metric = "minimum" if use_min else "mean"
    fig.suptitle(
        f"{title_metric} ESS over time | φ={phi_fixed}, σe={sigma_eps_fixed}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


def plot_metric_over_time_algorithms(
    results_all,
    phi_fixed,
    sigma_eps_fixed,
    d_list=None,
    metric="RMSE",
    N_bpf_list=None,
    reduce_mode="mean",
    figsize_per_row=5,
):
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import numpy as np

    metric = metric.upper()

    configs = sorted(
        [
            (d, phi, sigma_eps)
            for (d, phi, sigma_eps) in results_all.keys()
            if phi == phi_fixed and sigma_eps == sigma_eps_fixed
        ],
        key=lambda x: x[0]
    )

    if d_list is None:
        d_list = [d for (d, _, _) in configs]

    d_list = sorted(d_list)

    # Colors
    kf_colors = {
        "EKF_misspec": "#ff0054",  
        "EKF_correct": "#ff0054",
        "UKF_misspec": "#0096ff",   
        "UKF_correct": "#0096ff",
    }

    # BPF 
    if N_bpf_list is not None:
        palette = ["#00cc96", "#ffa600", "#ab63fa", "#19d3f3"]  
        bpf_colors = {
            N: palette[i % len(palette)] for i, N in enumerate(sorted(N_bpf_list))
        }
    else:
        bpf_colors = {}

    fig, axes = plt.subplots(
        len(d_list), 1,
        figsize=(8, figsize_per_row * len(d_list)),
        squeeze=False
    )

    def reduce_series(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr
        if reduce_mode == "mean":
            return np.mean(arr, axis=-1)
        if reduce_mode == "median":
            return np.median(arr, axis=-1)
        if reduce_mode == "max":
            return np.max(arr, axis=-1)
        raise ValueError("invalid reduce_mode")

    for i, d in enumerate(d_list):
        ax = axes[i, 0]
        res = results_all[(d, phi_fixed, sigma_eps_fixed)]

        # ---------- BPF ----------
        if N_bpf_list is not None:
            for N in N_bpf_list:
                if metric == "RMSE":
                    arr = tf.cast(res["metrics"][N]["rmse"], tf.float64).numpy()
                else:
                    arr = tf.cast(res["metrics"][N]["nees"], tf.float64).numpy()

                y = reduce_series(arr)
                x = np.arange(len(y))

                ax.plot(
                    x, y,
                    linewidth=2.5,
                    color=bpf_colors[N],
                    label=f"BPF N={N}"
                )

        # ---------- KF ----------
        for method_name, out in res["KF"].items():

            linestyle = "-" if "correct" in method_name else "--"
            color = kf_colors.get(method_name, "black")

            if metric == "RMSE":
                arr = tf.cast(out["rmse"], tf.float64).numpy()
            else:
                arr = tf.cast(out["nees"], tf.float64).numpy()

            y = reduce_series(arr)
            x = np.arange(len(y))

            ax.plot(
                x, y,
                linewidth=2.5,
                linestyle=linestyle,
                color=color,
                label=method_name.replace("_", " ")
            )

        ax.set_title(f"d = {d}", fontsize=12)
        ax.set_xlabel("Time")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)

        # legend
        ax.legend(frameon=False, fontsize=9, ncol=2)

    fig.suptitle(
        f"{metric} over time | φ={phi_fixed}, σe={sigma_eps_fixed}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


def plot_rmse_over_dimension(
    results_all,
    phi_fixed,
    sigma_eps_fixed,
    N_list,
    reduce_time="mean",
    reduce_coord="mean",
):
    """
    Plot RMSE as a function of state dimension d.

    Parameters
    ----------
    results_all : dict
        Dictionary keyed by (d, phi, sigma_eps).
    phi_fixed : float
        Fixed phi value.
    sigma_eps_fixed : float
        Fixed sigma_eps value.
    N_list : list
        BPF particle counts to include.
    reduce_time : str
        How to reduce over time:
        - "mean"
        - "median"
        - "max"
        - "final"
    reduce_coord : str
        How to reduce over coordinates if RMSE has shape (T,d):
        - "mean"
        - "median"
        - "max"
    """
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import numpy as np

    configs = sorted(
        [
            (d, phi, sigma_eps)
            for (d, phi, sigma_eps) in results_all.keys()
            if phi == phi_fixed and sigma_eps == sigma_eps_fixed
        ],
        key=lambda x: x[0]
    )

    if not configs:
        raise ValueError(
            f"No configurations found for phi={phi_fixed}, sigma_eps={sigma_eps_fixed}."
        )

    d_values = [d for (d, _, _) in configs]

    def reduce_rmse(arr):
        arr = np.asarray(arr)

        if arr.ndim == 2:
            if reduce_coord == "mean":
                arr = np.mean(arr, axis=-1)
            elif reduce_coord == "median":
                arr = np.median(arr, axis=-1)
            elif reduce_coord == "max":
                arr = np.max(arr, axis=-1)
            else:
                raise ValueError("reduce_coord must be 'mean', 'median', or 'max'.")

        if reduce_time == "mean":
            return np.mean(arr)
        elif reduce_time == "median":
            return np.median(arr)
        elif reduce_time == "max":
            return np.max(arr)
        elif reduce_time == "final":
            return arr[-1]
        else:
            raise ValueError("reduce_time must be 'mean', 'median', 'max', or 'final'.")

    plt.figure(figsize=(8, 5))

    # ---------------- BPF lines ----------------
    for N in N_list:
        y_vals = []

        for cfg in configs:
            rmse = tf.cast(results_all[cfg]["metrics"][N]["rmse"], tf.float64).numpy()
            y_vals.append(reduce_rmse(rmse))

        plt.plot(d_values, y_vals, marker="o", linewidth=2, label=f"BPF N={N}")

    # ---------------- KF lines ----------------
    kf_methods = ["EKF_misspec", "UKF_misspec", "EKF_correct", "UKF_correct"]

    for method in kf_methods:
        y_vals = []

        for cfg in configs:
            rmse = tf.cast(results_all[cfg]["KF"][method]["rmse"], tf.float64).numpy()
            y_vals.append(reduce_rmse(rmse))

        plt.plot(d_values, y_vals, marker="o", linewidth=2, linestyle="--", label=method)

    plt.xlabel("State dimension d")
    plt.ylabel("RMSE")
    plt.title(
        f"RMSE over dimension for fixed φ={phi_fixed}, σe={sigma_eps_fixed}"
    )
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    plt.show()


def plot_ESS_heatmap_over_d(
    results_all,
    N_list,
    phi,
    sigma_eps,
    use_min=False,
):
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import numpy as np

    # -------- FILTER CONFIGS --------
    configs = sorted(
        [cfg for cfg in results_all if cfg[1] == phi and cfg[2] == sigma_eps],
        key=lambda x: x[0]
    )

    if not configs:
        raise ValueError(f"No configurations found for phi={phi}, sigma_eps={sigma_eps}.")

    d_labels = [str(d) for (d, _, _) in configs]
    ess_key = "ESS_min_t" if use_min else "ESS_mean_t"

    # -------- BUILD HEATMAP DATA --------
    H_list = []
    for N in N_list:
        H = tf.stack(
            [tf.cast(results_all[cfg]["ESS"][N][ess_key], tf.float64) for cfg in configs],
            axis=0
        ).numpy()
        H_list.append(H)

    # -------- GLOBAL COLOR SCALE --------
    all_values = np.concatenate([H.ravel() for H in H_list])
    vmin = np.nanmin(all_values)
    vmax = np.nanmax(all_values)

    # -------- FIGURE (FIXED HEIGHT) --------
    fig, axes = plt.subplots(
        1, len(N_list),
        figsize=(6 * len(N_list), 4.5),
        sharey=True
    )

    if len(N_list) == 1:
        axes = [axes]

    im = None

    # -------- PLOT --------
    for ax, N, H in zip(axes, N_list, H_list):
        im = ax.imshow(
            H,
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis"
        )
        ax.set_title(f"N = {N}")
        ax.set_xlabel("Time")

    # -------- Y AXIS ONLY LEFT --------
    axes[0].set_yticks(np.arange(len(configs)))
    axes[0].set_yticklabels(d_labels)
    axes[0].set_ylabel("State dimension d")

    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False, left=False)

    # -------- COLORBAR --------
    fig.subplots_adjust(left=0.15, right=0.88, wspace=0.25)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(im, cax=cax, label="ESS")

    # -------- TITLE --------
    title_metric = "minimum" if use_min else "mean"
    fig.suptitle(
        f"{title_metric} ESS heatmap | φ={phi}, σe={sigma_eps}",
        y=0.98
    )

    plt.show()


def build_ESS_summary_dataframe(results_all):
    rows = []

    for (d, phi, sigma_eps), res in results_all.items():
        for N, met in res["metrics"].items():
            row = {
                "d": int(d),
                "phi": phi,
                "sigma_eps": sigma_eps,
                "N": int(N),
                "n_below_thr": int(met["ess_n_below"].numpy()),
                "frac_below_thr": float(met["ess_frac_below"].numpy()),
                "mean_ESS_norm": float(met["ess_mean"].numpy()),
#                "q05_ESS_norm": float(met["ess_q05"].numpy()),
                "min_ESS_norm": float(met["ess_min"].numpy()),
                "loglik_mean": float(met["loglik_mean"].numpy()),
                "rmse": float(tf.reduce_mean(tf.cast(met["rmse"], tf.float64)).numpy())
            }
            rows.append(row)

    return pd.DataFrame(rows)
