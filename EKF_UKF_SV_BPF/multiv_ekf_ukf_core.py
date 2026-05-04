import numpy as np
import tensorflow as tf


#### EKF/UKF CORE 
@tf.function(reduce_retracing=True)
def unscented_sigma_points_batch(x_particles, P, alpha, beta, kappa): 
    """
    x_particles: (Np, d) particle states
    P: (d, d) covariance matrix
    Returns:
        sigma_points: (Np, 2d+1, d)
        Wm: (2d+1,)
        Wc: (2d+1,)
    """
    dtype = x_particles.dtype
    P = tf.cast(P, dtype)

    shape = tf.shape(x_particles)
    Np = shape[0]
    d = shape[1]
    d_float = tf.cast(d, dtype)

    lam = alpha**2 * (d_float + kappa) - d_float
    c = d_float + lam

    # Weights
    Wm = tf.concat([[lam / c], tf.fill([2*d], 1/(2*c))], axis=0)
    Wc = tf.concat([[lam / c + (1 - alpha**2 + beta)], tf.fill([2*d], 1/(2*c))], axis=0)

    # Cholesky scaled
    sqrtP = tf.linalg.cholesky(c * P)  # (d,d)
    
    # Reshape for broadcasting over Np particles
    sqrtP = tf.reshape(sqrtP, (1, d, d))  # (1,d,d)
    x_exp = tf.expand_dims(x_particles, 1) # (Np,1,d)
    
    sigma_plus = x_exp + tf.transpose(sqrtP, perm=[0,2,1])  # (Np,d,d)
    sigma_minus = x_exp - tf.transpose(sqrtP, perm=[0,2,1])

    sigma_points = tf.concat([x_exp, sigma_plus, sigma_minus], axis=1)  # (Np, 2d+1, d)

    return sigma_points, Wm, Wc


def unscented_sigma_points(x, P, alpha, beta, kappa):
    sigma, Wm, Wc = unscented_sigma_points_batch(x[None, :], P[None, :, :],alpha, beta, kappa) #unscented_sigma_points_batch_strict() not working for UKF
    
    return sigma[0], Wm, Wc


def unscented_transform(sigma, Wm, Wc, noise_cov=None):
    
    mean = tf.reduce_sum(sigma * Wm[:, None], axis=0)
    diff = sigma - mean
    cov = tf.einsum('i,ij,ik->jk', Wc, diff, diff) 

    if noise_cov is not None:
        cov = cov + noise_cov

    return mean, cov, diff



@tf.function(reduce_retracing=True)
def ukf_predict(x, P, f, Q, alpha, beta, kappa, t):

    # -------------------------
    # Input validation 
    # -------------------------
    tf.debugging.assert_all_finite(x, "x contains NaN or Inf")
    tf.debugging.assert_all_finite(P, "P contains NaN or Inf")
    tf.debugging.assert_all_finite(Q, "Q contains NaN or Inf")

    tf.debugging.assert_equal(tf.shape(x)[0], tf.shape(P)[0], message="x and P dimension mismatch")
    tf.debugging.assert_equal(tf.shape(P)[0], tf.shape(P)[1], message="P must be square")

    # Symmetry check (cheap + useful)
    tf.debugging.assert_near(P, tf.transpose(P), atol=1e-8, message="P is not symmetric")

    
    dtype = x.dtype
    sigma, Wm, Wc = unscented_sigma_points(x, P, alpha, beta, kappa)
    
    # Transform sigma points 
    sigma_f = tf.vectorized_map(lambda s: f(s, t), sigma)
    sigma_f = tf.cast(sigma_f, dtype)
    
    x_pred, P_pred, _ = unscented_transform(sigma_f, Wm, Wc, Q)
    return x_pred, P_pred


@tf.function(reduce_retracing=True)
def ukf_update_check(x_pred, P_pred, y, h, R, alpha=1e-3, beta=2.0, kappa=0.0, t=0):

    # --------------------------------------------------------
    # Input validation 
    # --------------------------------------------------------
    if not tf.is_tensor(x_pred):
        raise TypeError("x_pred must be a TensorFlow tensor")

    if not tf.is_tensor(P_pred):
        raise TypeError("P_pred must be a TensorFlow tensor")

    if not tf.is_tensor(y):
        raise TypeError("y must be a TensorFlow tensor")

    if not tf.is_tensor(R):
        raise TypeError("R must be a TensorFlow tensor")

    if not callable(h):
        raise TypeError("h must be a callable function")

    if len(x_pred.shape) != 1:
        raise ValueError("x_pred must be a 1D tensor")

    if len(P_pred.shape) != 2:
        raise ValueError("P_pred must be a 2D tensor")

    if P_pred.shape[0] != P_pred.shape[1]:
        raise ValueError("P_pred must be a square matrix")

    if x_pred.shape[0] != P_pred.shape[0]:
        raise ValueError("x_pred and P_pred dimension mismatch")

    if y.shape[0] < 1:
        raise ValueError("y must be a non-empty vector")

    # --------------------------------------------------------
    # Cast 
    # --------------------------------------------------------
    dtype = x_pred.dtype
    P_pred = tf.cast(P_pred, dtype)
    y = tf.cast(y, dtype)
    R = tf.cast(R, dtype)

    n = tf.shape(x_pred)[0]

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------
    tf.debugging.assert_all_finite(x_pred, "x_pred contains NaNs or Infs")
    tf.debugging.assert_all_finite(P_pred, "P_pred contains NaNs or Infs")
    tf.debugging.assert_all_finite(y, "y contains NaNs or Infs")
    tf.debugging.assert_all_finite(R, "R contains NaNs or Infs")

    tf.debugging.assert_near(
        P_pred, tf.transpose(P_pred), atol=1e-8,
        message="P_pred is not symmetric"
    )

    # PD check
    sqrt_test = tf.linalg.cholesky(P_pred)  

    # --------------------------------------------------------
    # UKF parameters
    # --------------------------------------------------------
    lam = alpha**2 * (tf.cast(n, dtype) + kappa) - tf.cast(n, dtype)
    c = tf.cast(n, dtype) + lam

    Wm = tf.concat([[lam / c], tf.fill([2*n], 1/(2*c))], axis=0)
    Wc = tf.concat([[lam / c + (1 - alpha**2 + beta)], tf.fill([2*n], 1/(2*c))], axis=0)

    Wm = tf.cast(Wm, dtype)
    Wc = tf.cast(Wc, dtype)

    # --------------------------------------------------------
    # Sigma points
    # --------------------------------------------------------
    sqrtP = tf.linalg.cholesky(c * P_pred)

    sigma = tf.concat([
        x_pred[None, :],
        x_pred[None, :] + tf.transpose(sqrtP),
        x_pred[None, :] - tf.transpose(sqrtP)
    ], axis=0)

    # --------------------------------------------------------
    # Nonlinear transformation
    # --------------------------------------------------------
    sigma_y = tf.vectorized_map(lambda s: h(s, t), sigma)

    y_pred = tf.reduce_sum(sigma_y * Wm[:, None], axis=0)

    dy = sigma_y - y_pred
    dx = sigma - x_pred

    # --------------------------------------------------------
    # Covariances
    # --------------------------------------------------------
    S = tf.einsum('i,ij,ik->jk', Wc, dy, dy) + R
    Pxz = tf.einsum('i,ij,ik->jk', Wc, dx, dy)

    tf.debugging.assert_all_finite(S, "Innovation covariance S contains NaNs/Infs")

    # --------------------------------------------------------
    # Kalman gain
    # --------------------------------------------------------
    K = tf.transpose(tf.linalg.solve(S, tf.transpose(Pxz)))

    v_t = y - y_pred

    # --------------------------------------------------------
    # State update
    # --------------------------------------------------------
    x_filt = x_pred + tf.linalg.matvec(K, v_t)

    # --------------------------------------------------------
    # Covariance update
    # --------------------------------------------------------
    P_filt = P_pred - K @ S @ tf.transpose(K)

    return x_filt, P_filt, v_t, S, K



@tf.function(reduce_retracing=True)
def _filter_core(
    Y,
    predict_fn,
    update_fn,
    R_mat,
    m0,
    P0,
    measurement_type,
    dtype
):
    # --------------------------------------------------------
    # Input validation 
    # --------------------------------------------------------
    if not tf.is_tensor(Y):
        raise TypeError("Y must be a TensorFlow tensor")

    if not tf.is_tensor(R_mat):
        raise TypeError("R_mat must be a TensorFlow tensor")

    if not tf.is_tensor(m0):
        raise TypeError("m0 must be a TensorFlow tensor")

    if not tf.is_tensor(P0):
        raise TypeError("P0 must be a TensorFlow tensor")

    if not callable(predict_fn):
        raise TypeError("predict_fn must be callable")

    if not isinstance(update_fn, dict):
        raise TypeError("update_fn must be a dictionary")

    if "h" not in update_fn or "step" not in update_fn:
        raise ValueError("update_fn must contain 'h' and 'step'")

    if len(Y.shape) != 2:
        raise ValueError("Y must be a 2D tensor (n_y, T)")

    if len(m0.shape) != 1:
        raise ValueError("m0 must be a 1D tensor")

    if len(P0.shape) != 2:
        raise ValueError("P0 must be a 2D tensor")

    if P0.shape[0] != P0.shape[1]:
        raise ValueError("P0 must be square")

    if m0.shape[0] != P0.shape[0]:
        raise ValueError("m0 and P0 dimension mismatch")
    
    Y = tf.cast(Y, dtype)
    R_mat = tf.cast(R_mat, dtype)
    m0 = tf.cast(m0, dtype)
    P0 = tf.cast(P0, dtype)

    # --------------------------------------------------------
    # Input checks
    # --------------------------------------------------------
    tf.debugging.assert_all_finite(Y, "Y contains NaN or Inf")
    tf.debugging.assert_all_finite(R_mat, "R_mat contains NaN or Inf")
    tf.debugging.assert_all_finite(m0, "m0 contains NaN or Inf")
    tf.debugging.assert_all_finite(P0, "P0 contains NaN or Inf")


    n_y, T = Y.shape
    n_x = m0.shape[0]
    
    # Allocation 
    mu_filt = tf.TensorArray(dtype, size=T)
    P_filt = tf.TensorArray(dtype, size=T)
    P_pred_store = tf.TensorArray(dtype, size=T)   # New

    # Initialisation
    x_filt = m0
    P_filt_t = P0
    loglik = tf.constant(0.0, dtype=dtype)

    # Recursion
    for t in tf.range(T):
        y_t = Y[:, t]

        # -------- PREDICT --------
        x_pred, P_pred = predict_fn(x_filt, P_filt_t, t)

        # STORE PREDICTED COVARIANCE
        P_pred_store = P_pred_store.write(t, P_pred)

        # -------- R handling --------
        # Gaussian case: use fixed observation covariance R_mat.
        # Poisson case, included to support the Li example:
        # the observation variance equals the predicted intensity,
        # so the effective covariance is R_t = diag(h(x_pred, t)).
        if measurement_type == "poisson":
            y_pred = update_fn["h"](x_pred, t)
            R_foo = tf.linalg.diag(tf.reshape(y_pred, [-1]))
        else:
            R_foo = R_mat

        # -------- UPDATE --------
        x_filt, P_filt_t, v, S = update_fn["step"](x_pred, P_pred, y_t, R_foo, t)

        # -------- STORE --------
        mu_filt = mu_filt.write(t, x_filt)
        P_filt = P_filt.write(t, P_filt_t)

        # -------- LOG-LIK --------
        v_col = tf.reshape(v, (-1, 1))
        n_y_tf = tf.cast(tf.shape(Y)[0], dtype)
        log2pi = tf.math.log(tf.constant(2.0 * np.pi, dtype=dtype))

        ll = -0.5 * (n_y_tf * log2pi + tf.math.log(tf.linalg.det(S)) + tf.transpose(v_col) @ tf.linalg.solve(S, v_col))[0, 0]

        loglik += ll

    return {
        "mu_filt": tf.transpose(mu_filt.stack()),
        "P_filt": P_filt.stack(),
        "P_pred": P_pred_store.stack(),   
        "loglik": loglik
    }


def make_ukf_kernels(F_func, H_func, Q, alpha, beta, kappa):

    @tf.function(reduce_retracing=True)
    def predict_fn(x, P, t):
        dtype = x.dtype
        
        x = tf.cast(x, dtype)
        P = tf.cast(P, dtype)
        Q_ = tf.cast(Q, dtype)

        x_pred, P_pred = ukf_predict(x, P, F_func, Q_, alpha, beta, kappa, t)

        return tf.cast(x_pred, dtype), tf.cast(P_pred, dtype)

    @tf.function(reduce_retracing=True)
    def update_step(x_pred, P_pred, y, R, t):
        dtype = x_pred.dtype

        x_pred = tf.cast(x_pred, dtype)
        P_pred = tf.cast(P_pred, dtype)
        y = tf.cast(y, dtype)
        R_ = tf.cast(R, dtype)

        x, P, v, S, _ = ukf_update_check(
            x_pred, P_pred, y,
            H_func, R_,
            alpha, beta, kappa, t
        )

        return tf.cast(x, dtype), tf.cast(P, dtype), tf.cast(v, dtype), tf.cast(S, dtype)

    return predict_fn, {
        "step": update_step,
        "h": H_func
    }

# EKF

def make_ekf_kernels(F_func, H_func, F_jac, H_jac, Q):

    @tf.function(reduce_retracing=True)
    def predict_fn(x, P, t):
        dtype = x.dtype

        F = tf.cast(F_jac(x, t), dtype)
        x_pred = tf.cast(F_func(x, t), dtype)
        Q_ = tf.cast(Q, dtype)

        P_pred = F @ P @ tf.transpose(F) + Q_

        return x_pred, P_pred

    @tf.function(reduce_retracing=True)
    def update_step(x_pred, P_pred, y, R, t):
        dtype = x_pred.dtype

        H = tf.cast(H_jac(x_pred, t), dtype)
        y_pred = tf.cast(H_func(x_pred, t), dtype)
        R_ = tf.cast(R, dtype)
        
        # innovation
        v = y - y_pred
        S = H @ P_pred @ tf.transpose(H) + R_
        #kalman gain
        K = tf.transpose(tf.linalg.solve(S, tf.transpose(P_pred @ tf.transpose(H))))

        x = x_pred + tf.linalg.matvec(K, v)

        I = tf.eye(tf.shape(x_pred)[0], dtype=dtype)
        KH = K @ H
        # covariance
        P = (I - KH) @ P_pred @ tf.transpose(I - KH) + K @ R_ @ tf.transpose(K)

        return x, P, v, S

    return predict_fn, {
        "step": update_step,
        "h": H_func
    }
