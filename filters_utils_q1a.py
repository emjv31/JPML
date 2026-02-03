import tensorflow as tf
import numpy as np
import math

import warnings
warnings.filterwarnings('ignore') 


def KF_multivariate_robust(Y, F_mat, H_mat, Q_mat, R_mat, m0, P0):
    """
    Kalman filter implementation for multivariate time series
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
    log2pi = tf.math.log(tf.constant(2*math.pi, dtype=tf.float64)) # tf.math.log(tf.constant(2.0 * 3.141592653589793, tf.float64))

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
    for t in range(T):
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
    # Results
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

