import math
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")


def KF_multivariate_robust(Y, F_mat, H_mat, Q_mat, R_mat, m0, P0):
    """
    Implementation of the Kalman filter for multivariate time series
    Inputs:
        Y       : (n_y, T) observation matrix
        F_mat   : (n_x, n_x) state transition matrix
        H_mat   : (n_y, n_x) observation matrix
        Q_mat   : (n_x, n_x) state noise covariance matrix
        R_mat   : (n_y, n_y) observation noise covariance matrix
        m0      : (n_x,) initial state mean 
        P0      : (n_x, n_x) initial state covariance
    Returns:
        Dictionary with: filtered & predicted means and covariances, prediction errors, Kalman gains, and log-likelihoods
    """
    # Convert input
    Y = tf.convert_to_tensor(Y, dtype=tf.float64)
    F_mat = tf.convert_to_tensor(F_mat, dtype=tf.float64)
    H_mat = tf.convert_to_tensor(H_mat, dtype=tf.float64)
    Q_mat = tf.convert_to_tensor(Q_mat, dtype=tf.float64)
    R_mat = tf.convert_to_tensor(R_mat, dtype=tf.float64)
    m0 = tf.convert_to_tensor(m0, dtype=tf.float64)
    P0 = tf.convert_to_tensor(P0, dtype=tf.float64)
    
    # Check input 
    n_y, T = Y.shape
    n_x = m0.shape[0]

    assert F_mat.shape == (n_x, n_x), f"F_mat shape {F_mat.shape} incompatible with state dimension {n_x}"
    assert H_mat.shape == (n_y, n_x), f"H_mat shape {H_mat.shape} incompatible with Y shape {Y.shape} and state {n_x}"
    assert Q_mat.shape == (n_x, n_x), f"Q_mat shape {Q_mat.shape} incompatible with state dimension {n_x}"
    assert R_mat.shape == (n_y, n_y), f"R_mat shape {R_mat.shape} incompatible with observation dimension {n_y}"
    assert P0.shape == (n_x, n_x), f"P0 shape {P0.shape} incompatible with state dimension {n_x}"

    # Check covariance matrices are symmetric and positive semi-definite
    for cov, name in [(Q_mat, "Q_mat"), (R_mat, "R_mat"), (P0, "P0")]:
        tf.debugging.assert_near(cov, tf.transpose(cov), atol=1e-10, message=f"{name} not symmetric")
        eigs = tf.linalg.eigvalsh(cov)
        tf.debugging.assert_non_negative(eigs, message=f"{name} not PSD")

    ####
    # Allocation space
    ###
    I = tf.eye(n_x, dtype=tf.float64)
    log2pi = tf.math.log(tf.constant(2*math.pi, dtype=tf.float64)) 

    mu_pred_before = tf.TensorArray(tf.float64, size=T)
    mu_pred_next = tf.TensorArray(tf.float64, size=T)
    P_pred = tf.TensorArray(tf.float64, size=T)
    mu_filt = tf.TensorArray(tf.float64, size=T)
    P_filt = tf.TensorArray(tf.float64, size=T)
    P_filt_joseph = tf.TensorArray(tf.float64, size=T)
    innovations = tf.TensorArray(tf.float64, size=T)
    innovation_covs = tf.TensorArray(tf.float64, size=T)
    gains = tf.TensorArray(tf.float64, size=T)
    loglik_vec = tf.TensorArray(tf.float64, size=T)
    
    #####
    # Initialisation
    #####
    mu_pred_before = mu_pred_before.write(0, m0)
    mu_pred_next = mu_pred_next.write(0, m0)
    P_pred = P_pred.write(0, P0)
    loglik = tf.constant(0.0, dtype=tf.float64)
    
    #####
    # Recursion
    ####
    for t in tf.range(T):
        x_pred = mu_pred_before.read(t)
        P_pred_t = P_pred.read(t)
        y_t = Y[:, t]
        
        # Innovation
        v_t = y_t - tf.linalg.matvec(H_mat, x_pred)
        innovations = innovations.write(t, v_t)
        
        # Innovation covariance
        S_t = H_mat @ P_pred_t @ tf.transpose(H_mat) + R_mat
        innovation_covs = innovation_covs.write(t, S_t)
        
        # Check positive definiteness
        eigs = tf.linalg.eigvalsh(S_t)
        tf.debugging.assert_non_negative(eigs, message=f"S_t not PSD at t={t}")
        
        # Kalman gain
        K_t = P_pred_t @ tf.transpose(H_mat) @ tf.linalg.inv(S_t)
        gains = gains.write(t, K_t)
        
        # Filtered state
        x_filt = x_pred + tf.linalg.matvec(K_t, v_t)
        mu_filt = mu_filt.write(t, x_filt)
        
        # Covariance update
        # Standard form
        P_f_std = (I - K_t @ H_mat) @ P_pred_t
        # Joseph form update
        KH = K_t @ H_mat
        P_f_joseph = (I - KH) @ P_pred_t @ tf.transpose(I - KH) + K_t @ R_mat @ tf.transpose(K_t)
        #
        P_filt = P_filt.write(t, P_f_std)
        P_filt_joseph = P_filt_joseph.write(t, P_f_joseph)
        tf.debugging.assert_all_finite(P_f_std, message=f"P_f contains NaN or inf at t={t}")
        tf.debugging.assert_all_finite(P_f_joseph, message=f"P_f contains NaN or inf at t={t}")

        # Log-likelihood
        v_col = tf.reshape(v_t, [n_y,1])
        quad = (tf.transpose(v_col) @ tf.linalg.inv(S_t) @ v_col)[0,0]
        logdet = tf.math.log(tf.linalg.det(S_t))
        ll = -0.5 * (tf.cast(n_y, tf.float64) * log2pi + logdet + quad)
        tf.debugging.assert_all_finite(ll, message=f"log-likelihood not finite at t={t}")
        loglik_vec = loglik_vec.write(t, ll)
        loglik += ll
        # One step-ahead prediction
        if t < T-1:
            x_pred_next = tf.linalg.matvec(F_mat, x_filt)
            mu_pred_next = mu_pred_next.write(t+1, x_pred_next)
            mu_pred_before = mu_pred_before.write(t+1, x_pred_next)
            P_pred = P_pred.write(t+1, F_mat @ P_f_joseph @ tf.transpose(F_mat) + Q_mat)
    ####
    # Output
    ####
    return {
        "mu_pred_before": tf.transpose(mu_pred_before.stack()),  # (n_x, T)
        "mu_pred_next": tf.transpose(mu_pred_next.stack()),      # (n_x, T)
        "P_pred": P_pred.stack(),                                # (T, n_x, n_x)
        "mu_filt": tf.transpose(mu_filt.stack()),                # (n_x, T)
        "P_filt": P_filt.stack(),                                # (T, n_x, n_x)
        "P_filt_joseph": P_filt_joseph.stack(),                  # Joseph form update
        "v": tf.transpose(innovations.stack()),                  # (n_y, T)
        "F_innov": innovation_covs.stack(),                      # (T, n_y, n_y)
        "K": gains.stack(),                                      # (T, n_x, n_y)
        "loglik_vec": loglik_vec.stack(),                        # (T,)
        "loglik": loglik                                         # scalar
    }



def Sim_mLGSSM(T, m0, F_mat, H_mat, Q_mat, R_mat, P0, seed=None):
    """
    Simulate a multivariate linear Gaussian state-space model:
        y_t = H x_t + eps_t,       eps_t ~ N(0, R)
        x_t = F x_{t-1} + eta_t,  eta_t ~ N(0, Q)
    
    Input:
        T: int, number of time steps
        m0: initial mean [n_x]
        F_mat: state transition matrix [n_x, n_x]
        H_mat: observation matrix [n_y, n_x]
        Q_mat: state noise covariance [n_x, n_x]
        R_mat: observation noise covariance [n_y, n_y]
        P0: initial covariance [n_x, n_x]
        seed: optional int for reproducibility
    
    Returns:
        X_true: latent states [n_x, T]
        Y: observations [n_y, T]
    """
    import tensorflow as tf

    if seed is not None:
        tf.random.set_seed(seed)

    # Convertion
    m0 = tf.convert_to_tensor(m0, dtype=tf.float64)
    F_mat = tf.convert_to_tensor(F_mat, dtype=tf.float64)
    H_mat = tf.convert_to_tensor(H_mat, dtype=tf.float64)
    Q_mat = tf.convert_to_tensor(Q_mat, dtype=tf.float64)
    R_mat = tf.convert_to_tensor(R_mat, dtype=tf.float64)
    P0 = tf.convert_to_tensor(P0, dtype=tf.float64)

    n_x = m0.shape[0]
    n_y = H_mat.shape[0]

    # Input check
    def is_symmetric(A, tol=1e-12):
        return tf.reduce_all(tf.abs(A - tf.transpose(A)) < tol)

    def is_pos_def(A):
        eigvals = tf.linalg.eigvalsh(A)
        return tf.reduce_all(eigvals > 0)

    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if F_mat.shape != (n_x, n_x):
        raise ValueError(f"F_mat must be [{n_x},{n_x}], got {F_mat.shape}")
    if H_mat.shape[1] != n_x:
        raise ValueError(f"H_mat must have {n_x} columns, got {H_mat.shape[1]}")
    if Q_mat.shape != (n_x, n_x):
        raise ValueError(f"Q_mat must have shape [{n_x},{n_x}], got {Q_mat.shape}")
    if R_mat.shape != (n_y, n_y):
        raise ValueError(f"R_mat must have shape [{n_y},{n_y}], got {R_mat.shape}")
    if P0.shape != (n_x, n_x):
        raise ValueError(f"P0 must have shape [{n_x},{n_x}], got {P0.shape}")

    for mat, name in [(Q_mat, "Q_mat"), (R_mat, "R_mat"), (P0, "P0")]:
        if not tf.reduce_all(tf.math.is_finite(mat)):
            raise ValueError(f"{name} contains NaN or Inf")
        if not is_symmetric(mat):
            raise ValueError(f"{name} must be symmetric")
        if not is_pos_def(mat):
            raise ValueError(f"{name} must be positive definite")

    # Cholesky decomposition for sampling
    LQ = tf.linalg.cholesky(Q_mat)
    LR = tf.linalg.cholesky(R_mat)
    LP0 = tf.linalg.cholesky(P0)

    #Allocation space
    X_true_list = []
    Y_list = []

    # Initialisation 
    x_prev = m0  
    y_prev = tf.linalg.matvec(H_mat, x_prev) + tf.squeeze(LR @ tf.random.normal([n_y, 1], dtype=tf.float64))

    X_true_list.append(x_prev)
    Y_list.append(y_prev)

    # Recursion
    for t in range(1, T):
        x_prev = tf.linalg.matvec(F_mat, x_prev) + tf.squeeze(LQ @ tf.random.normal([n_x, 1], dtype=tf.float64))
        y_prev = tf.linalg.matvec(H_mat, x_prev) + tf.squeeze(LR @ tf.random.normal([n_y, 1], dtype=tf.float64))

        X_true_list.append(x_prev)
        Y_list.append(y_prev)
        
    # Stack outputs
    X_true = tf.stack(X_true_list, axis=1)
    Y = tf.stack(Y_list, axis=1)

    # Degeneracy checks
    for Z, name in [(X_true, "Latent states"), (Y, "Observations")]:
        if tf.reduce_any(tf.math.is_nan(Z)) or tf.reduce_any(tf.math.is_inf(Z)):
            raise ValueError(f"{name} contain NaN or Inf")
        if tf.reduce_all(Z == Z[:, 0:1]):
            raise ValueError(f"{name} are degenerate (all values equal)")

    return X_true, Y


def reconstruct_and_check(mu_pred, K, v, mu_filt, P_pred, P_filt, H_mat, R_mat, d):
    """
    mu_pred: (n_x, T)
    K:       (T, n_x, n_y)
    v:       (n_y, T)
    mu_filt: (n_x, T)

    P_pred:  (T, n_x, n_x)
    P_filt:  (T, n_x, n_x)
    """
    # Only check ranks that matter for time dependence
    tf.debugging.assert_rank(mu_pred, 2)
    tf.debugging.assert_rank(v, 2)
    tf.debugging.assert_rank(K, 3)
    tf.debugging.assert_rank(P_pred, 3)
    tf.debugging.assert_rank(P_filt, 3)

    # H_mat and R_mat are allowed to be 2D
    tf.debugging.assert_rank(H_mat, 2)
    tf.debugging.assert_rank(R_mat, 2)
    # =========================
    # STATE RECONSTRUCTION
    # =========================

    v_T = tf.transpose(v)                    # (T, n_y)
    v_T = tf.expand_dims(v_T, axis=-1)

    Kv = tf.matmul(K, v_T)                   # (T, n_x, 1)
    Kv = tf.squeeze(Kv, axis=-1)             # (T, n_x)
    Kv = tf.transpose(Kv)                    # (n_x, T)

    mu_rec = mu_pred + Kv

    rec_mu = tf.norm(mu_rec - mu_filt, axis=0)
    
    rec_mu_by_state = tf.abs(mu_rec - mu_filt)       # (n_x, T)
    rec_mu = tf.norm(mu_rec - mu_filt, axis=0)       # (T,)

    # =========================
    # COVARIANCE (JOSEPH)
    # =========================

    KH = tf.matmul(K, H_mat)                 # (T, n_x, n_x)

    I = tf.eye(d, dtype=mu_pred.dtype)       # (n_x, n_x)

    I_KH = I - KH                       

    P_joseph = (tf.matmul(I_KH, tf.matmul(P_pred, tf.transpose(I_KH, perm=[0,2,1]))) + tf.matmul(K, tf.matmul(R_mat, tf.transpose(K, perm=[0,2,1]))))
    rec_P = tf.norm(P_joseph - P_filt, axis=[1,2])

    P_joseph_diag = tf.linalg.diag_part(P_joseph)     # (T, n_x)
    P_filt_diag = tf.linalg.diag_part(P_filt)         # (T, n_x)
    rec_P_diag = tf.abs(P_joseph_diag - P_filt_diag)  # (T, n_x)

    return {
        "mu_rec": mu_rec,
        "P_joseph": P_joseph,
        "rec_mu": rec_mu,                    # global over states, by time
        "rec_mu_by_state": rec_mu_by_state,  # by state, by time
        "rec_P": rec_P,                      # global matrix error, by time
        "rec_P_diag": rec_P_diag             # diagonal error, by state, by time
    }

def compute_kf_metrics(X_true, kf_out):
    X_true = tf.convert_to_tensor(X_true, dtype=tf.float64)

    mu_filt = kf_out["mu_filt"]          # (n_x, T)
    P_filt=kf_out["P_filt_joseph"]       # (T, n_x, n_x)
    v = kf_out["v"]                      # (n_y, T)
    S = kf_out["F_innov"]                # (T, n_y, n_y)
    loglik_vec = kf_out["loglik_vec"]

    err = mu_filt - X_true               # (n_x, T)

    # Error metrics
    bias_state = tf.reduce_mean(err, axis=1)                         # (n_x,)
    bias_time = tf.reduce_mean(err, axis=0)                          # (T,)

    rmse_state = tf.sqrt(tf.reduce_mean(tf.square(err), axis=1))     # (n_x,)
    rmse_time = tf.sqrt(tf.reduce_mean(tf.square(err), axis=0))      # (T,)

    # 95% confidence interval diagnostics
    var_filt = tf.linalg.diag_part(P_filt)                           # (T, n_x)
    sd_filt = tf.sqrt(tf.maximum(var_filt, 0.0))

    lower = tf.transpose(mu_filt) - 1.96 * sd_filt
    upper = tf.transpose(mu_filt) + 1.96 * sd_filt
    x_true_T = tf.transpose(X_true)

    inside = tf.cast((x_true_T >= lower) & (x_true_T <= upper), tf.float64)
    coverage95_state = tf.reduce_mean(inside, axis=0)                # (n_x,)
    avg_ci_width_state = tf.reduce_mean(2.0 * 1.96 * sd_filt, axis=0)

    # NIS
    v_T = tf.transpose(v)
    v_T = tf.expand_dims(v_T, axis=-1)                               # (T, n_y, 1)
    S_inv_v = tf.linalg.solve(S, v_T)
    nis = tf.squeeze(tf.matmul(v_T, S_inv_v, transpose_a=True), axis=[1, 2])   # (T,)

    # NEES
    e_T = tf.transpose(err)
    e_T = tf.expand_dims(e_T, axis=-1)                               # (T, n_x, 1)
    P_inv_e = tf.linalg.solve(P_filt, e_T)
    nees = tf.squeeze(tf.matmul(e_T, P_inv_e, transpose_a=True), axis=[1, 2])  # (T,)

    return {
        "bias_state": bias_state,
        "bias_time": bias_time,
        "rmse_state": rmse_state,
        "rmse_time": rmse_time,
        "coverage95_state": coverage95_state,
        "avg_ci_width_state": avg_ci_width_state,
        "nis": nis,
        "nees": nees,
        "nis_mean": tf.reduce_mean(nis),
        "nees_mean": tf.reduce_mean(nees),
        "loglik_total": kf_out["loglik"],
        "loglik_cum": tf.cumsum(loglik_vec)
    }


def run_experiments_KF(T, m0, F_mat, H_mat, Q_mat, R_mat, P0, label="experiment", seed=123):
    tf.random.set_seed(seed)
    # Simulate data
    X_true, Y_sim = Sim_mLGSSM(
        T=T,
        m0=m0,
        F_mat=F_mat,
        H_mat=H_mat,
        Q_mat=Q_mat,
        R_mat=R_mat,
        P0=P0,
        seed=seed
    )
    # --- Kalman filter ---
    kf_out = KF_multivariate_robust(
        Y_sim,
        F_mat,
        H_mat,
        Q_mat,
        R_mat,
        m0,
        P0
#        propagate_with="joseph"
    )
    
    # --- Reconstruction check ---
    rec = reconstruct_and_check(
        mu_pred=kf_out["mu_pred_before"],
        K=kf_out["K"],
        v=kf_out["v"],
        mu_filt=kf_out["mu_filt"],
        P_pred=kf_out["P_pred"],
        P_filt=kf_out["P_filt_joseph"],
        H_mat=H_mat,
        R_mat=R_mat,
        d=m0.shape[0]
    )
    # --- Metrics ---
    metrics = compute_kf_metrics(X_true, kf_out)

    return {
        "label": label,
        "X_true": X_true,
        "Y_sim": Y_sim,
        "kf": kf_out,
        "reconstruction": rec,
        "metrics": metrics
    }


import pandas as pd
import numpy as np

def build_global_metrics_dataframe(experiment_results):
    rows = []

    for res in experiment_results:
        met = res["metrics"]

        rows.append({
            "experiment": res["label"],
            "bias_global": float(np.mean(met["bias_state"].numpy())),
            "rmse_global": float(np.mean(met["rmse_state"].numpy())),
            "coverage_95_global": float(np.mean(met["coverage95_state"].numpy())),
            "ci_width_global": float(np.mean(met["avg_ci_width_state"].numpy())),
            "nis_mean": float(met["nis_mean"].numpy()),
            "nees_mean": float(met["nees_mean"].numpy()),
            "loglik": float(met["loglik_total"].numpy())
        })

    return pd.DataFrame(rows)


def print_reconstruction_summary(experiment_results):
    for res in experiment_results:
        rec = res["reconstruction"]

        print(f"\n========== {res['label']} ==========")
        print("Max reconstruction error mu:", tf.reduce_max(rec["rec_mu"]).numpy())
        print("Mean reconstruction error mu:", tf.reduce_mean(rec["rec_mu"]).numpy())
        print("Max reconstruction error mu by state:", tf.reduce_max(rec["rec_mu_by_state"], axis=1).numpy())
        print("Mean reconstruction error mu by state:", tf.reduce_mean(rec["rec_mu_by_state"], axis=1).numpy())

        print("Max reconstruction error P:", tf.reduce_max(rec["rec_P"]).numpy())
        print("Mean reconstruction error P:", tf.reduce_mean(rec["rec_P"]).numpy())
        print("Max diagonal reconstruction error P by state:", tf.reduce_max(rec["rec_P_diag"], axis=0).numpy())
        print("Mean diagonal reconstruction error P by state:", tf.reduce_mean(rec["rec_P_diag"], axis=0).numpy())





if __name__ == "__main__":
    ### EXPERIMENTS

    tf.random.set_seed(123)
    #T = 500
    m0 = tf.constant([0.0, 0.0], dtype=tf.float64)
    n_x = m0.shape[0]
    n_y = n_x  # states directly observed

    F_mat = tf.constant([[0.8, 0.1],
                     [0.0, 0.9]], dtype=tf.float64)
    H_mat = tf.eye(n_y, dtype=tf.float64)

    P0 = tf.eye(n_x, dtype=tf.float64)


    experiments = [
        {
            "T": 500,
            "Q_mat": tf.constant([[0.05, 0.0],[0.0, 0.05]], dtype=tf.float64),
            "R_mat": tf.constant([[0.3, 0.0],[0.0, 0.3]], dtype=tf.float64),
            "label": "Exp 1: Noisy measurements"   # State noise is moderate, Measurement noise is relatively large
                                               # Measurements are not very informative (they are noisy)
        },
        {
            "T": 500, 
            "Q_mat": tf.constant([[0.005, 0.0],[0.0, 0.005]], dtype=tf.float64), # State noise is very small
            "R_mat": tf.constant([[0.2, 0.0],[0.0, 0.2]], dtype=tf.float64),     # Measurement noise is smaller than Exp 1, but still larger than Q
            "label": "Exp 2: System is stable/predictable"  
        }, 
            {
            "T": 500,
            "Q_mat": tf.constant([[0.05, 0.0],  [0.0, 0.05]], dtype=tf.float64),
            "R_mat": tf.constant([[0.01, 0.0], [0.0, 0.01]], dtype=tf.float64),
            "label": "Exp 3: Very informative measurements"
        }
    ]

    all_results = []
    
    for exp in experiments:
        res = run_experiments_KF(
            T=exp["T"],
            m0=m0,
            F_mat=F_mat,
            H_mat=H_mat,
            Q_mat=exp["Q_mat"],
            R_mat=exp["R_mat"],
            P0=P0,
            label=exp["label"],
            seed=123
        )
        all_results.append(res)

    df_global = build_global_metrics_dataframe(all_results)
    print(df_global.round(4))

    print_reconstruction_summary(all_results)