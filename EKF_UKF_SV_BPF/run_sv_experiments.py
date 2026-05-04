import os
import tensorflow as tf
import matplotlib.pyplot as plt

from sv_experiments import (
    compare_methods_one_config,
    plot_ESS_heatmap_over_d,
    plot_ESS_over_time_bpf,
    plot_metric_over_time_algorithms,
    plot_rmse_over_dimension,
    plot_benchmark_grouped,
)


##########
######### RUN EXPERIMENTS 

FIG_DIR = "figures_q2"
os.makedirs(FIG_DIR, exist_ok=True)

#N_list = [50, 200, 500]
d_list = [5, 10, 50]
results_all = {}

for d in d_list:
    for phi in [0.5, 0.95]:
        for sigma_eps in [1.0]:
            print(f"\nRunning d={d}, phi={phi}, sigma_eps={sigma_eps}")

            results_all[(d, phi, sigma_eps)] = compare_methods_one_config(
                d=d,
                phi=phi,
                sigma_eps=sigma_eps,
                sigma_eta=1.0,
                xi=1.0,
                Np=(50, 200, 500),
                T=100,
                N_MC=100,
                dtype_bpf=tf.float32,
                dtype_kf=tf.float64,
            )


####### PLOTS
plot_ESS_heatmap_over_d(
    results_all,
    N_list=[50, 200, 500],
    phi=0.95,
    sigma_eps=1.0,
    use_min=False
)
plt.savefig(os.path.join(FIG_DIR, "ess_heatmap_phi_05_sigmaeps_10.png"), dpi=300, bbox_inches="tight")
plt.close()

plot_ESS_over_time_bpf(
    results_all,
    phi_fixed=0.95,
    sigma_eps_fixed=1.0,
    N_bpf_list=[50, 200, 500],
    use_min=False,
)
plt.savefig(os.path.join(FIG_DIR, "ess_over_time_phi_05_sigmaeps_10.png"), dpi=300, bbox_inches="tight")
plt.close()


plot_metric_over_time_algorithms(
    results_all,
    phi_fixed=0.95,
    sigma_eps_fixed=1.0,
    d_list=d_list,
    metric="RMSE", 
    N_bpf_list=[50, 200, 500],
    reduce_mode="mean",
)
plt.savefig(os.path.join(FIG_DIR, "rmse_over_time_phi_095_sigmaeps_10.png"), dpi=300, bbox_inches="tight")
plt.close()

plot_rmse_over_dimension(
    results_all,
    phi_fixed=0.95,
    sigma_eps_fixed=1.0,
    N_list=[50, 200, 500],
    reduce_time="mean",
    reduce_coord="mean",
)
plt.savefig(os.path.join(FIG_DIR, "rmse_over_dimension_phi_095_sigmaeps_10.png"), dpi=300, bbox_inches="tight")
plt.close()

plot_benchmark_grouped(results_all)
plt.savefig(os.path.join(FIG_DIR, "benchmark_runtime_memory.png"), dpi=300, bbox_inches="tight")
plt.close()

# Log-likelihood comparison

for cfg, res in results_all.items():
    d, phi, sigma_eps = cfg

    print(f"\n=== d={d}, phi={phi}, sigma_eps={sigma_eps} ===")

    # BPF
    for N in res["metrics"]:
        print(f"BPF N={N}: loglik_mean = {res['metrics'][N]['loglik_mean']:.4f}")

    # EKF / UKF
    for method in res["KF"]:
        ll = res["KF"][method]["loglik"].numpy()
        print(f"{method}: loglik = {ll:.4f}")
