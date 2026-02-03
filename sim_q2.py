import tensorflow as tf
import math

import matplotlib.pyplot as plt 
import seaborn as sns

from filters_utils_q2 import compute_Sigma_tf, Sim_HD_LGSSM, particle_flow_update_linear_new, update_weight_linear_new, particle_flow_pf_new
from filters_utils_q2 import bpf_multi_simple, ukf_predict, ukf_update, state_space_filter, esrf, sample_skewed_t, simulate_skewed_t_HDS 
from filters_utils_q2 import compute_kernel, grad_log_posterior, hu_particle_flow_update, hu_pff_filter
from filters_utils_q2 import matrix_valued_kernel, hu_pff_matrix_kernel_example, 

#a) 
# I have to run a series of simulations to show the results of the paper in terms of Bias, RMSE etc. but the functions exluding the UKF work
# Compute sigma spatial 
Sigma_tf = compute_Sigma_tf(d = 64)

# Application 
d = 64
T = 100
alpha = 0.9 # stationary but persistent AR process
sigma_z = 0.3

true_states, measurements = Sim_HD_LGSSM(d=d, T=T, alpha=alpha, sigma_z=sigma_z, Sigma_tf=Sigma_tf)

print(true_states.shape)   
print(measurements.shape)  

# Cast to float64 for Kalman/EKF
Y64 = tf.cast(tf.transpose(measurements), tf.float64)  # (d, T)

# EKF Jacobians (linear case)
def F_jac(x, t):
    return tf.eye(d, dtype=tf.float64) * alpha

def H_jac(x, t):
    return tf.eye(d, dtype=tf.float64)

# EKF works
m0_hd = tf.zeros(d, dtype=tf.float64)
P0_hd = tf.eye(d, dtype=tf.float64) * 0.1
Q_mat_hd = tf.eye(d, dtype=tf.float64) * 0.05
R_mat_hd = tf.eye(d, dtype=tf.float64) * sigma_z**2

ekf_result_hd = state_space_filter(
    Y=Y64,
    F_func=lambda x,t: alpha*x,
    H_func=lambda x,t: x,
    Q_mat=Q_mat_hd,
    R_mat=R_mat_hd,
    m0=m0_hd,
    P0=P0_hd,
    method="EKF",
    F_jac=F_jac,
    H_jac=H_jac
)

mu_ekf_hd = ekf_result_hd["mu_filt"]  


# KF 
m0_hd = tf.zeros(d, dtype=tf.float64)
P0_hd = tf.eye(d, dtype=tf.float64) * 0.1
Q_mat_hd = tf.eye(d, dtype=tf.float64) * 0.05
R_mat_hd = tf.eye(d, dtype=tf.float64) * sigma_z**2

F_tf_j = tf.eye(d, dtype=tf.float64) * alpha
H_tf_j = tf.eye(d, dtype=tf.float64)    

kf_result_hd = state_space_filter(
    Y=Y64,
    F_func=F_tf_j,
    H_func=H_tf_j,
    Q_mat=Q_mat_hd,
    R_mat=R_mat_hd,
    m0=m0_hd,
    P0=P0_hd,
    method="KF",
    F_jac=F_jac,
    H_jac=H_jac
)

mu_kf_hd = kf_result_hd["mu_filt"]

# UKF not working
ukf_result_hd = state_space_filter(
    Y=Y64,
    F_func=lambda x,t: alpha*x,
    H_func=lambda x,t: x,
    Q_mat=Q_mat_hd,
    R_mat=R_mat_hd,
    m0=m0_hd,
    P0=P0_hd,
    method="UKF",
    F_jac=F_jac,
    H_jac=H_jac
)

mu_ukf_hd = ukf_result_hd["mu_filt"]  

# Np = 200
esrf_new = esrf_filter(true_states, measurements, Np, F_tf, Q_tf, H_tf, R_tf)


# Simulation C)
sim_data_skewed = simulate_skewed_t_HDS(d=64, T=10, alpha=0.9, nu=5.0, num_runs=5) # num_runs = N, MC replicates
print("Simulated data shape:", sim_data_skewed.shape) 


# II)
# particles so to visualise the marginals
x_scalar, z_scalar, ests_hu_scalar, part_scalar = hu_pff_filter()
# same setting as for the scalar - in the example
T = 5 
Np = 20
# d = 64 # as at the beginning
x_matrix, z_matrix, ests_hu_matrix, part_matrix = hu_pff_matrix_kernel_example(T=T, Np=Np, d=d, alpha=1.0, eps=0.1)

# PLOTS 
obs_dims = [0, 1] # selecting the first two variables

plt.figure(figsize=(6,4))
plt.scatter(part_scalar[:, obs_dims[0]],
            part_scalar[:, obs_dims[1]],
            c='red', alpha=0.5, label='Scalar kernel')

plt.scatter(part_matrix[:, obs_dims[0]],
            part_matrix[:, obs_dims[1]],
            c='blue', alpha=0.5, label='Matrix-valued kernel')
plt.xlabel(f'x[{obs_dims[0]}]')
plt.ylabel(f'x[{obs_dims[1]}]')
plt.title('Posterior estimate')
plt.legend()
plt.grid(True)
plt.show()


def plot_particle_marginals(particles_scalar, particles_matrix, dims=(0, 1), bins=30):
    """
    Compare particle marginals for scalar vs matrix-valued kernel.
    
    particles_scalar: [Np, d] particles from scalar kernel
    particles_matrix: [Np, d] particles from matrix-valued kernel
    dims: tuple of two dimensions to plot (default first two)
    bins: number of bins for 2D histogram / contour
    """
    x_dim, y_dim = dims
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Scalar 
    x_s = particles_scalar[:, x_dim].numpy() if isinstance(particles_scalar, tf.Tensor) else particles_scalar[:, x_dim]
    y_s = particles_scalar[:, y_dim].numpy() if isinstance(particles_scalar, tf.Tensor) else particles_scalar[:, y_dim]
    
    # 2D histogram for density
    H_s, xedges_s, yedges_s = np.histogram2d(x_s, y_s, bins=bins, density=True)
    X_s, Y_s = np.meshgrid((xedges_s[1:] + xedges_s[:-1])/2, (yedges_s[1:] + yedges_s[:-1])/2)
    
    axes[0].contourf(X_s, Y_s, H_s.T, cmap='Blues', alpha=0.7)
    axes[0].scatter(x_s, y_s, color='blue', alpha=0.5, s=20)
    axes[0].set_title("Scalar Kernel (Hu)")
    axes[0].set_xlabel(f"x[{x_dim}]")
    axes[0].set_ylabel(f"x[{y_dim}]")
    axes[0].grid(True)
    
    # Matrix-valued kernel
    x_m = particles_matrix[:, x_dim].numpy() if isinstance(particles_matrix, tf.Tensor) else particles_matrix[:, x_dim]
    y_m = particles_matrix[:, y_dim].numpy() if isinstance(particles_matrix, tf.Tensor) else particles_matrix[:, y_dim]
    
    H_m, xedges_m, yedges_m = np.histogram2d(x_m, y_m, bins=bins, density=True)
    X_m, Y_m = np.meshgrid((xedges_m[1:] + xedges_m[:-1])/2, (yedges_m[1:] + yedges_m[:-1])/2)
    
    axes[1].contourf(X_m, Y_m, H_m.T, cmap='Reds', alpha=0.7)
    axes[1].scatter(x_m, y_m, color='red', alpha=0.5, s=20)
    axes[1].set_title("Matrix-valued Kernel")
    axes[1].set_xlabel(f"x[{x_dim}]")
    axes[1].set_ylabel(f"x[{y_dim}]")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

plot_particle_marginals(particles_scalar=part_scalar, 
                        particles_matrix=part_matrix, 
                        dims=(0,1),
                        bins=40)


# III) W.I.P.
d = 64
Np = 500
#P_pred_foo = Q_tf
P_pred_foo = tf.eye(d) * 0.1
R_tf_foo = tf.eye(d) * sigma_z**2

#
pfpf_edh = particle_flow_pf_new(true_states, measurements, Np = 200, P_pred = P_pred_foo, flow_type="EDH", use_weights=False)
pfpf_edh_w = particle_flow_pf_new(true_states, measurements, Np = 200, P_pred = P_pred_foo, flow_type="EDH", use_weights=True)
pfpf_ledh_w = particle_flow_pf_new(true_states, measurements, Np = 200, P_pred = P_pred_foo, flow_type="LEDH", use_weights=True)