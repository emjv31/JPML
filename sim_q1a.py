import tensorflow as tf
import matplotlib.pyplot as plt
import math
import numpy as np

from filters_utils_q1a import KF_multivariate_robust, Sim_mLGSSM 

import warnings
warnings.filterwarnings('ignore') 


## Simulations 
# 1) Example 1 
tf.random.set_seed(123)
T = 500
m0 = tf.constant([0.0, 0.0], dtype=tf.float64)
n_x = m0.shape[0]
n_y = n_x  

F_mat = tf.constant([[0.8, 0.1],
                     [0.0, 0.9]], dtype=tf.float64)
H_mat = tf.eye(n_y, dtype=tf.float64)

Q_mat = tf.constant([[0.05, 0.0],
                     [0.0, 0.05]], dtype=tf.float64)
R_mat = tf.constant([[0.3, 0.0],
                     [0.0, 0.3]], dtype=tf.float64) 
P0 = tf.eye(n_x, dtype=tf.float64)

# Run simulations
X_true, Y_sim = Sim_mLGSSM(T = T, m0 = m0, F_mat = F_mat, H_mat = H_mat, Q_mat = Q_mat, R_mat = R_mat, P0 = P0, seed=123)
# Results 
kf_out = KF_multivariate_robust(Y_sim, F_mat, H_mat, Q_mat, R_mat, m0, P0)

# Compare filtered means and covariances to the kalamn recursions
rec_kf = []
for t in range(T):
    mu_pred_kf= kf_out['mu_pred_before'][:, t]
    K_kf = kf_out['K'][t] 
    v_kf = kf_out['v'][:, t]
    mu_rec_kf = mu_pred_kf + tf.linalg.matvec(K_kf, v_kf)
    rec_kf.append(tf.norm(mu_rec_kf - kf_out['mu_filt'][:, t]))

rec_kf = tf.stack(rec_kf)
#
rec_P_kf = []
d = 2 
I = tf.eye(d, dtype=tf.float64)

for t in range(T):
    P_pred = kf_out['P_pred'][t]   
    K = kf_out['K'][t]
    # Joseph form
    P_joseph = ((I - K @ H_mat) @ P_pred @ tf.transpose(I - K @ H_mat)+ K @ R_mat @ tf.transpose(K))
    # reconstruction error
    P_filt = kf_out['P_filt_joseph'][t]
    rec_P_kf.append(tf.norm(P_joseph - P_filt))

rec_P_kf = tf.stack(rec_P_kf)

print("Max reconstruction error:", tf.reduce_max(rec_kf).numpy())
print("Mean reconstruction error:", tf.reduce_mean(rec_kf).numpy())
print("Max Joseph covariance reconstruction error:", tf.reduce_max(rec_P_kf).numpy())
print("Mean Joseph covariance reconstruction error:", tf.reduce_mean(rec_P_kf).numpy())

# Filtered means: bias and RMSE 
X_true_np = X_true.numpy()
X_filt_np = kf_out['mu_filt'].numpy()
print("Bias per state:", np.mean(X_filt_np - X_true_np, axis=1))
print("RMSE per state:", np.sqrt(np.mean((X_filt_np - X_true_np)**2, axis = 1)

## Numerical stability of the filter: deterministic diagnostic given the output from the filter
### Compute the CONDITIONING NUMBER using outputs from the filter                                         
# Innovation covariance matrix
S = kf_out["F_innov"]
# Eigenvalues of S_t
eigs_S = tf.linalg.eigvalsh(S)         
# Condition number over time
cond_number = eigs_S[:, -1] / (eigs_S[:, 0]) 

plt.figure()
plt.hist(cond_number.numpy(), bins=50, log=True)
plt.xlabel("Condition number")
plt.ylabel("Frequency (log scale)")
plt.title("Distribution of innovation covariance condition numbers")
plt.tight_layout()
plt.savefig("cond_number_hist.png", dpi=300)
plt.show()


#Example 2
           
Q_mat = tf.constant([[0.005, 0.0],
                     [0.0, 0.005]], dtype=tf.float64)
# the filtered values should not trust the observations too much
R_mat = tf.constant([[0.2, 0.0],
                     [0.0, 0.2]], dtype=tf.float64)

# Run simulations and repeat the analysis as above (from line 3) using kf_out_v1 rather than kf_out
X_true, Y_sim = Sim_mLGSSM(T = T, m0 = m0, F_mat = F_mat, H_mat = H_mat, Q_mat = Q_mat, R_mat = R_mat, P0 = P0, seed=123)
kf_out_v1 = KF_multivariate_robust(Y_sim, F_mat, H_mat, Q_mat, R_mat, m0, P0)
         