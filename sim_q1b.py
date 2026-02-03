import tensorflow as tf 
import math
import numpy as np
import time

from filters_utils_q1b import SV_model_sim_tf_h
from filters_utils_q1b import compute_jacobian_tf, compute_sigma_points_tf, extensionKF_uni_tf_consistent
from filters_utils_q1b import SIR_bootstrap_markov_tf_stat, profile_tf, compute_bias_rmse

import warnings
warnings.filterwarnings('ignore') 


# II)
## Monte Carlo experiment - NONLNEAR CASE to show degeneracy
# Parameters
T = 200
phi = 0.7
sigma_eta = 0.2
sigma_e = 0.1

tf.random.set_seed(42) 

# Initial state x[0] ~ N(0,1)
x0 = tf.random.normal([], mean=0.0, stddev=1.0, dtype=tf.float64)
x = tf.TensorArray(tf.float64, size=T)
x = x.write(0, x0)

# State recursion
for t in range(1, T):
    eta_t = tf.random.normal([], mean=0.0, stddev=sigma_eta, dtype=tf.float64)
    x_t = phi * x.read(t-1) + eta_t
    x = x.write(t, x_t)
x = x.stack()

# Observations
y = x**2 / 5 + tf.random.normal([T], mean=0.0, stddev=sigma_e, dtype=tf.float64)


# Define f() and h() for the nonlinear model
def f_fun_tf(x):
    return phi * x

def h_fun_tf(x):
    return x**2 / 5


# APPLICATION with a nonlinear model
# EKF 
res_EKF_tf = extensionKF_uni_tf_consistent(
    y=y,
    f_fun=f_fun_tf,
    h_fun=h_fun_tf,
    sigma_eta=sigma_eta,
    sigma_e=sigma_e,
    m0=tf.constant(0.5, dtype=tf.float64),
    P0=tf.constant(1.0, dtype=tf.float64),
    method="EKF"  
)

res_UKF_tf = extensionKF_uni_tf_consistent(
    y=y,
    f_fun=f_fun_tf,
    h_fun=h_fun_tf,
    sigma_eta=sigma_eta,
    sigma_e=sigma_e,
    m0=tf.constant(0.5, dtype=tf.float64),
    P0=tf.constant(1.0, dtype=tf.float64),
    method="UKF"  
)


# III)
### Simulate from the model SV above 
# Fixed parameters
phi = 0.95
sigma_eta = 0.2
xi = 1.0
sigma_eps = 1.0

# generate data
sim_sv_tf_out = SV_model_sim_tf_h(iT=500, phi=phi, sigma_eta=sigma_eta, xi=xi, sigma_eps=sigma_eps)
y_tf = sim_sv_tf_out["vY"]
h_tf = sim_sv_tf_out["h"]

# Apply SIR to compute ESS
SIR_markov_v1 = SIR_bootstrap_markov_tf_stat(N = 50, phi = phi, tau = sigma_eta, sigma = sigma_eps, squared=False, Y=y_tf)
SIR_markov_v2 = SIR_bootstrap_markov_tf_stat(N = 150, phi = phi, tau = sigma_eta, sigma = sigma_eps, squared=False, Y=y_tf)
SIR_markov_v3 = SIR_bootstrap_markov_tf_stat(N = 500, phi = phi, tau = sigma_eta, sigma = sigma_eps, squared=False, Y=y_tf)


#Visualise particle degeneracy 

# Extract ESS arrays from SIR outputs
ESS_50  = np.array(SIR_markov_v1["ESS"])
ESS_150 = np.array(SIR_markov_v2["ESS"])
ESS_500 = np.array(SIR_markov_v3["ESS"])
# Normalize ESS 
ESS_50_norm  = ESS_50 / 50
ESS_150_norm = ESS_150 / 150
ESS_500_norm = ESS_500 / 500


plt.figure(figsize=(8,4))
plt.plot(ESS_50_norm,  label="N=50")
plt.plot(ESS_150_norm, label="N=150")
plt.plot(ESS_500_norm, label="N=500")

# Shaded regions correspond to ESS/N < 0.2, severe degeneracy
for i, ESS_norm in enumerate([ESS_50_norm, ESS_150_norm, ESS_500_norm]):
    degenerate_indices = np.where(ESS_norm < 0.2)[0]
    for idx in degenerate_indices:
        plt.axvspan(idx-0.5, idx+0.5, color=f"C{i}", alpha=0.1)

plt.xlabel("Time")
plt.ylabel("Normalized ESS (ESS/N)")
plt.title("Normalized ESS with shaded severe degeneracy regions")
plt.legend()
plt.tight_layout()
plt.savefig("normalized_ESS.png", dpi=300)
plt.show()


# IV)
### BPF vs. EKF/UKF in the SV model
y = y_tf
z = tf.math.log(y**2 + 1e-8)  # approximate linear Gaussian model based on Harvey
z = tf.convert_to_tensor(z, dtype=tf.float64)

xi_tf = tf.constant(xi, dtype=tf.float64)
sigma_eps_tf = tf.constant(sigma_eps, dtype=tf.float64)
c0 = tf.math.log(xi_tf**2 * sigma_eps_tf**2) - tf.constant(1.2704, dtype=tf.float64) # see Harvey (1996)

# Functions to compute the Taylor approx.
def f_fun_sv(x):
    return phi * x

def h_fun_sv(x):
    return x + c0 # see Harvey (1996)


# Apply EKF
res_EKF_sv = extensionKF_uni_tf_consistent(y=z, f_fun=f_fun_sv, h_fun=h_fun_sv, sigma_eta=sigma_eta, sigma_e=sigma_eps_tf, m0=tf.constant(0.0, dtype=tf.float64), P0=tf.constant(1.0, dtype=tf.float64), method="EKF")

# Apply UKF
res_UKF_sv = extensionKF_uni_tf_consistent(y=z, f_fun=f_fun_sv, h_fun=h_fun_sv, sigma_eta=sigma_eta, sigma_e=sigma_eps_tf, m0=tf.constant(0.0, dtype=tf.float64), P0=tf.constant(1.0, dtype=tf.float64), method="UKF")


# COMPARE PERFORMANCE
SIR_results = [SIR_markov_v1, SIR_markov_v2, SIR_markov_v3]

# Compute and print bias/RMSE 
for i, res in enumerate(SIR_results, start=1):
    bias_rmse = compute_bias_rmse(h_tf, res['part_est'])
    print(f"SIR result {i}: {bias_rmse}")

print(compute_bias_rmse(h_tf, res_EKF_sv['mu_filt'])) #EKF
print(compute_bias_rmse(h_tf, res_UKF_sv['mu_filt'])) # UKF

#EKF/UKF
result_perf_ukf, stats_perf_ukf = profile_tf(extensionKF_uni_tf_consistent, y=z, f_fun=f_fun_sv, h_fun=h_fun_sv, sigma_eta=sigma_eta, sigma_e=sigma_eps_tf, m0=tf.constant(0.0, dtype=tf.float64), P0=tf.constant(1.0, dtype=tf.float64), method="UKF")

result_perf_ekf, stats_perf_ekf = profile_tf(extensionKF_uni_tf_consistent, y=z, f_fun=f_fun_sv, h_fun=h_fun_sv, sigma_eta=sigma_eta, sigma_e=sigma_eps_tf, m0=tf.constant(0.0, dtype=tf.float64), P0=tf.constant(1.0, dtype=tf.float64), method="EKF")

# SIR
Ns = [50, 100, 500, 1000]
sir_results = {}

for N in Ns:
    result, stats = profile_tf(
        SIR_bootstrap_markov_tf_stat,
        N=N,
        phi=phi,
        tau=sigma_eta,
        sigma=sigma_eps,
        squared=False,
        Y=y_tf
    )
    sir_results[N] = {"result": result, "stats": stats}

# print performance
for N in Ns:
    stats = sir_results[N]["stats"]
    print(f"Runtime SIR (N={N}): {stats['runtime_seconds']} seconds")
    print(f"GPU peak memory SIR (N={N}): {stats['gpu_memory']} MB")

print("Runtime EKF:", stats_perf_ukf["runtime_seconds"], "seconds")
print("GPU peak memory UKF:", stats_perf_ukf["gpu_memory"], "MB")
#
print("Runtime UKF:", stats_perf_ekf["runtime_seconds"], "seconds")
print("GPU peak memory UKF:", stats_perf_ekf["gpu_memory"], "MB")