import sys
import os

import tensorflow as tf
import math
import time 

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings('ignore') 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from EKF_UKF_SV_BPF.multiv_sv_bpf_core import (
    multinomial_resampling,
#    bpf_generic_resampling,
)

from EKF_UKF_SV_BPF.multiv_ekf_ukf_core import (
    _filter_core,
    make_ekf_kernels,
    make_ukf_kernels,
    ukf_predict,
    ukf_update_check,
    unscented_sigma_points_batch,
    unscented_sigma_points,
    unscented_transform,
)

from simulator_model_comps import gaussian_logpdf_batch


def make_kf_kernels(F_mat, H_mat, Q):

    @tf.function(reduce_retracing=True)
    def predict_fn(x, P, t):
        dtype = x.dtype

        F = tf.cast(F_mat, dtype)
        Q_ = tf.cast(Q, dtype)

        # Predicted state mean
        x_pred = tf.linalg.matvec(F, x)
        # Predicted state covariance
        P_pred = F @ P @ tf.transpose(F) + Q_

        return x_pred, P_pred

    @tf.function(reduce_retracing=True)
    def update_step(x_pred, P_pred, y, R, t):
        dtype = x_pred.dtype

        H = tf.cast(H_mat, dtype)
        R_ = tf.cast(R, dtype)

        # Innovation error 
        v = y - tf.linalg.matvec(H, x_pred)

        # Innovation covariance
        S = H @ P_pred @ tf.transpose(H) + R_
        
        # Cross-covariance between state and observation
        PHt = P_pred @ tf.transpose(H)
        
        # Kalman gain
        K = tf.transpose(tf.linalg.solve(S, tf.transpose(PHt)))

        # Updated filtered state mean
        x = x_pred + tf.linalg.matvec(K, v)

        # Joseph-form covariance update 
        I = tf.eye(tf.shape(x_pred)[0], dtype=dtype)
        KH = K @ H
        P = (I - KH) @ P_pred @ tf.transpose(I - KH) + K @ R_ @ tf.transpose(K)

        return x, P, v, S

    return predict_fn, {
        "step": update_step,
         # Observation function h(x) = Hx, used by the generic filter core
        # e.g. for predicted observations or Poisson covariance construction
        "h": lambda x, t: tf.linalg.matvec(H_mat, x)
    }



def esrf_filter(measurements, Np, F_func, Q, H_func, R, measurement_type = "gaussian", dtype=tf.float32):

    # --------------------------------------------------------
    # Input checks
    # --------------------------------------------------------
    if not tf.is_tensor(measurements):
        raise TypeError("measurements must be a TensorFlow tensor")

    if not tf.is_tensor(Q):
        raise TypeError("Q must be a TensorFlow tensor")

    if not tf.is_tensor(R):
        raise TypeError("R must be a TensorFlow tensor")

    if not callable(F_func):
        raise TypeError("F_func must be callable")

    if not callable(H_func):
        raise TypeError("H_func must be callable")

    if not isinstance(Np, int) or Np <= 1:
        raise ValueError("Np must be an integer > 1")

    if len(measurements.shape) != 2:
        raise ValueError("measurements must be of shape (T, n_y)")

    if len(Q.shape) != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square")

    if len(R.shape) != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be square")

    if measurement_type.lower() not in [None, "gaussian", "poisson"]: 
        raise ValueError("measurement_type must be None, 'gaussian' or 'poisson'")

    # -----------------------------------
    # Casting
    # -----------------------------------

    measurements = tf.cast(measurements, dtype)
    Q = tf.cast(Q, dtype)
    R = tf.cast(R, dtype)

    state_dim = Q.shape[0]
    T = measurements.shape[0]

    particles = tf.random.normal((Np, state_dim), dtype=dtype)
    L_Q = tf.linalg.cholesky(Q)

    ests = []

    for t in range(T):

        z = measurements[t]

        # ----- Forecast -----
        eps = tf.random.normal((Np, state_dim), dtype=dtype)
        process_noise = tf.matmul(eps, tf.transpose(L_Q))
        particles = F_func(particles) + process_noise

        # ----- Ensemble mean / anomalies -----
        x_mean = tf.reduce_mean(particles, axis=0, keepdims=True)
        X = particles - x_mean

        y_pred = H_func(particles)
        y_mean = tf.reduce_mean(y_pred, axis=0, keepdims=True)
        Y = y_pred - y_mean

        if measurement_type.lower() == "gaussian":
            R_foo = R
        elif measurement_type.lower()=="poisson":
            R_foo = tf.linalg.diag(tf.squeeze(y_mean) + 1e-8)
        else:
            raise ValueError("Unknown measurement type")

        # ----- Covariances -----
        P_xy = tf.matmul(X, Y, transpose_a=True) / (Np - 1)
        P_yy = tf.matmul(Y, Y, transpose_a=True) / (Np - 1) + R_foo # R

        # ----- Kalman gain -----
        K = tf.matmul(P_xy, tf.linalg.inv(P_yy))

        # ----- Mean update -----
        x_mean_a = x_mean + tf.matmul((z - y_mean), tf.transpose(K))

        # ----- ESRF transform -----
        R_inv = tf.linalg.inv(R_foo)

        S = tf.eye(Np, dtype=dtype) + tf.matmul(tf.matmul(Y, R_inv), Y, transpose_b=True) / (Np - 1)

        eigvals, eigvecs = tf.linalg.eigh(S)
        S_inv_sqrt = eigvecs @ tf.linalg.diag(1.0 / tf.sqrt(eigvals)) @ tf.transpose(eigvecs)

        X_a = tf.matmul(S_inv_sqrt, X)

        # ----- Updated ensemble -----
        particles = x_mean_a + X_a

        ests.append(tf.squeeze(x_mean_a))

    return tf.stack(ests)



def upf_filter(
    Y,
    Np,
    alpha_dyn,
    Sigma,
    gamma,
    nu,
    Q_proposal,
    transition_logpdf_fn, 
    log_likelihood_fn,
    transition_mean_fn,
    resampling_fn=multinomial_resampling,
    dtype=tf.float64,
    alpha=1e-3, # UKF tuning parameters
    beta=2.0,
    kappa=0.0,
    resample_threshold = False
):
    """
    Unscented Particle Filter (UPF) for general nonlinear/non-Gaussian state-space models, using sigma-point proposals and importance sampling; 
    Gaussian filtering case can be recovered as a limiting case (set nu -> infty, e.g. 200) depending on the choice of likelihood and transition
    
    Inputs:
        Y: (T,d) measurements
        Np: number of particles
        alpha_dyn: autoregressive parameter
        Sigma: transition covariance (d,d)
        gamma: skew term (d,)
        nu: Student-t degrees of freedom
        Q_proposal: process noise covariance (d,d)
        transition_logpdf_fn: function(x_new, x_prev, L_Sigma, nu) -> (Np,) log-density
        log_likelihood_fn: function(particles, y_t) -> (Np,) log-likelihood
        transition_mean_fn: function(sigma, alpha_dyn, gamma) -> (Np,2d+1,d) - propagated mean

    Returns:
        ests: (T,d) posterior means
        ESSs: (T,) effective sample sizes
    """

    # -------------------------
    # Input validation 
    # -------------------------
    if not tf.is_tensor(Y):
        raise TypeError("Y must be a TensorFlow tensor")

    if len(Y.shape) != 2:
        raise ValueError("Y must have shape (T, d)")
    if Y.shape[0] <1:
        raise ValueError("Y must have at least one observation")

    if not isinstance(Np, int) or Np <= 1:
        raise ValueError("Np must be an integer > 1")

    if not callable(transition_logpdf_fn):
        raise TypeError("transition_logpdf_fn must be callable")

    if not callable(log_likelihood_fn):
        raise TypeError("log_likelihood_fn must be callable")

    if not callable(transition_mean_fn):
        raise TypeError("transition_mean_fn must be callable")

    if not callable(resampling_fn):
        raise TypeError("resampling_fn must be callable")

    if not tf.is_tensor(Sigma) or len(Sigma.shape) != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square matrix")

    if not tf.is_tensor(Q_proposal) or len(Q_proposal.shape) != 2 or Q_proposal.shape[0] != Q_proposal.shape[1]:
        raise ValueError("Q_proposal must be a square matrix")

    if measurement_type := None:  
        pass
    if measurement_type not in {None, "gaussian", "poisson"}:
        raise ValueError("measurement_type must be 'gaussian' or 'poisson'")


    # --------------------------------------------------------
    # Input checks 
    # --------------------------------------------------------
    for name, val in {
        "Y": Y,
        "Sigma": Sigma,
        "Q_proposal": Q_proposal,
    }.items():
        if not tf.is_tensor(val):
            raise TypeError(f"{name} must be a TensorFlow tensor")

    for name, fn in {
        "transition_logpdf_fn": transition_logpdf_fn,
        "log_likelihood_fn": log_likelihood_fn,
        "transition_mean_fn": transition_mean_fn,
    }.items():
        if not callable(fn):
            raise TypeError(f"{name} must be callable")

    if not isinstance(Np, int) or Np <= 1:
        raise ValueError("Np must be an integer > 1")

    if len(Y.shape) != 2:
        raise ValueError("Y must have shape (T, d)")

    if len(Sigma.shape) != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be square")

    if len(Q_proposal.shape) != 2 or Q_proposal.shape[0] != Q_proposal.shape[1]:
        raise ValueError("Q_proposal must be square")

    # ======= Cast to float64 - UKF can be fragile =======
    Y = tf.cast(Y, dtype)
    Sigma = tf.cast(Sigma, dtype)
    Q_proposal = tf.cast(Q_proposal, dtype)
    gamma = tf.cast(gamma, dtype)
    alpha_dyn = tf.cast(alpha_dyn, dtype)
    nu = tf.cast(nu, dtype)

    T, d = Y.shape

    particles = tf.zeros((Np, d), dtype=dtype)
    weights = tf.ones((Np,), dtype=dtype) / Np
    ests, ESSs = [], []
    
    threshold_value = Np/2.0
    
    start_time = time.time()
    
    # ======= UKF weights =======
    lam = alpha**2 * (d + kappa) - d
    c = d + lam

    lam = tf.cast(lam, dtype)
    c = tf.cast(c, dtype)

    Wm = tf.concat([tf.reshape(lam / c, (1,)), tf.fill((2*d,), tf.cast(1.0/(2*c), dtype))], axis=0)

    Wc = tf.identity(Wm)
    Wc = tf.tensor_scatter_nd_add(Wc, [[0]], [tf.cast(1.0 - alpha**2 + beta, dtype)])

    Wm = tf.reshape(Wm, (1, 2*d+1, 1))
    Wc = tf.reshape(Wc, (1, 2*d+1, 1, 1))

    # Precompute Cholesky factors
    L_Q = tf.linalg.cholesky(Q_proposal)  # (d,d)
    L_Sigma = tf.linalg.cholesky(Sigma) 
    log2pi = tf.math.log(2.0 * np.pi)
    
    # Time series recursion
    for t in range(T):
        y_t = Y[t]  # current observation

        # ---------------- Sigma points ----------------
        sigma, Wm, Wc = unscented_sigma_points_batch(particles, Q_proposal, alpha, beta, kappa) 
        Wm = tf.reshape(Wm, (1, -1, 1))
        # ---------------- Propagate sigma points ----------------
        sigma_f = transition_mean_fn(sigma, alpha_dyn) # (Np,2d+1,d)

        # ---------------- Compute (UKF) mean ----------------
        x_pred = tf.reduce_sum(Wm * sigma_f, axis=1)  # (Np,d)

        # ---------------- Compute the (UKF) covariance ----------------
        diff = sigma_f - tf.expand_dims(x_pred, 1)   # (Np,2d+1,d)
        # diff: (Np, K, d)
        Wc_flat = tf.reshape(Wc, (1, -1, 1))   # (1, K, 1)
        diff_w = diff * Wc_flat                # (Np, K, d)
        P_pred = tf.matmul(diff_w, diff, transpose_a=True) #+ Q_proposal # (Np,d,d)

        
        # ---------------- Sample Gaussian proposal ----------------
        eps = tf.random.normal((Np, d), dtype=dtype)
        
        #Lp = tf.linalg.cholesky(P_pred)
        #
        P_pred = 0.5 * (P_pred + tf.transpose(P_pred, perm=[0,2,1]))  # enforce symmetry
        Lp = tf.linalg.cholesky(tf.cast(P_pred, dtype))
        Lp = tf.cast(Lp, dtype)  

        noise = tf.matmul(Lp, eps[..., None])  # (Np,d,1)
        noise = tf.squeeze(noise, axis=2)
        x_new = x_pred + noise  # (Np,d)

        # ---------------- Measurement log-likelihood ----------------
        log_meas = log_likelihood_fn(x_new, y_t)  # (Np,)
        
        # ---------------- Transition log-density ----------------
        log_trans = transition_logpdf_fn(x_new, particles, L_Sigma) 
        # ---------------- Proposal log-density ----------------
        log_prop = gaussian_logpdf_batch(x_new, x_pred, P_pred)  # (Np,) old version


        # ---------------- Importance weights ----------------
        log_w = log_meas + log_trans - log_prop
        log_w = tf.where(tf.math.is_finite(log_w), log_w, tf.fill(tf.shape(log_w), tf.constant(-1e30, dtype=dtype))) # numerical stability
        log_w -= tf.reduce_logsumexp(log_w)
        weights = tf.exp(log_w)

        # Estimate
        est = tf.reduce_sum(x_new * weights[:, None], axis=0)
        ests.append(est)
        
        # ESS monitoring 
        ESS = 1.0 / tf.reduce_sum(weights**2)
        ESSs.append(ESS)

        # ---------------- Resampling ----------------
        do_resample = ESS < threshold_value if resample_threshold else True
        if do_resample:
            particles, weights = resampling_fn(x_new, weights)        
        else:
            particles = x_new

    exec_time = time.time() - start_time
    print(f"UPF executed in {exec_time: .3f}s")

    return tf.stack(ests), tf.stack(ESSs)


def gsmc_general(
    Y,                 # [T,d] measurements
    Np,                # number of particles
    prop_fn,           # lambda for particle propagation: x_new = prop_fn(x)
    log_likelihood_fn, # lambda for log-likelihood: log_w = log_likelihood_fn(particles, y_t)
    resampling_fn = multinomial_resampling, 
    resample_threshold=False,
    dtype=tf.float32
):
    """
    Generic particle filter:
        - Vectorized propagation
        - Vectorized log-likelihood
        - Resampling ste - default type is Multinomial
        - Returns estimates and ESS
    """
    # --------------------------------------------------------
    # Input validation 
    # --------------------------------------------------------

    # --- Tensor checks ---
    if not tf.is_tensor(Y):
        raise TypeError("Y must be a TensorFlow tensor")

    if len(Y.shape) != 2:
        raise ValueError("Y must have shape (T, d)")

    if Y.shape[0] < 1:
        raise ValueError("Y must contain at least one observation")

    # --- Np ---
    if not isinstance(Np, int) or Np <= 1:
        raise ValueError("Np must be an integer larger than 1")

    # --- Functions ---
    for name, fn in {
        "prop_fn": prop_fn,
        "log_likelihood_fn": log_likelihood_fn,
        "resampling_fn": resampling_fn,
    }.items():
        if not callable(fn):
            raise TypeError(f"{name} must be callable")

    Y = tf.cast(Y, dtype)
    T = tf.shape(Y)[0]
    d = tf.shape(Y)[1]

    # Initialize particles and weights
    particles = tf.zeros([Np, d], dtype=dtype)
    weights = tf.ones([Np], dtype=dtype) / Np
    
    ests, ESSs = [], []
    threshold_value = Np/2.0

    start_time = time.time()
    
    for t in tf.range(T):
        y_t = Y[t]

        # -----------------
        # Propagate particles
        # -----------------
        #if t > 0:
        particles = prop_fn(particles)  
            
        # -----------------
        # Compute log-likelihood 
        # -----------------
        log_w = log_likelihood_fn(particles, y_t)
        log_w -= tf.reduce_logsumexp(log_w)
        weights = tf.exp(log_w)

        # ----------------
        # State estimate
        # ----------------
        est = tf.reduce_sum(particles * tf.reshape(weights, [Np, 1]), axis=0)
        ests.append(est)
        
        # ---------------
        # Compute ESS
        # ---------------
        ESS = 1.0 / tf.reduce_sum(weights**2)
        ESSs.append(ESS)

        # ---------------
        # Resampling
        # ---------------
        do_resample = ESS < threshold_value if resample_threshold else True
        if do_resample:
            particles, weights = resampling_fn(particles, weights)

    exec_time = time.time()-start_time
    print(f"GSMC execution time: {exec_time: .3f}s")

    return tf.stack(ests), tf.stack(ESSs)



def helper_hmc_rej_vectorized(
    particles,          # [Np, d]
    y_t,                # current observation [d]
    log_likelihood_fn,  # function: log_likelihood_fn(particles, y_t) -> [Np]
    leapfrog_steps=5,
    epsilon=0.05,
    dtype=tf.float32
):
    """
    Vectorized HMC Rejuvenation helper.
    Fully vectorized over particles for speedup.
    """
    Np, d = particles.shape
    x = tf.identity(particles)                   # [Np, d]
    p = tf.random.normal((Np, d), dtype=dtype)  # [Np, d]

    x_new = tf.identity(x)
    p_new = tf.identity(p)

    # Helper: batch log posterior
    def log_post_batch(x_batch):
        lp = log_likelihood_fn(x_batch, y_t)   # [Np]
        return lp

    for _ in range(leapfrog_steps):
        # --- First half momentum step ---
        with tf.GradientTape() as tape:
            tape.watch(x_new)
            lp = log_post_batch(x_new)
            # sum to get scalar for batch grad
            lp_sum = tf.reduce_sum(lp)
        grad = tape.gradient(lp_sum, x_new)
        if grad is None:
            grad = tf.zeros_like(x_new)
        p_new = p_new + 0.5 * epsilon * grad

        # --- Full position step ---
        x_new = x_new + epsilon * p_new
        x_new = tf.clip_by_value(x_new, -50.0, 50.0)

        # --- Second half momentum step ---
        with tf.GradientTape() as tape:
            tape.watch(x_new)
            lp = log_post_batch(x_new)
            lp_sum = tf.reduce_sum(lp)
        grad = tape.gradient(lp_sum, x_new)
        if grad is None:
            grad = tf.zeros_like(x_new)
        p_new = p_new + 0.5 * epsilon * grad

    # --- Metropolis-Hastings accept/reject ---
    log_post_old = log_post_batch(x)
    log_post_new = log_post_batch(x_new)
    K_old = 0.5 * tf.reduce_sum(p**2, axis=1)
    K_new = 0.5 * tf.reduce_sum(p_new**2, axis=1)
    log_alpha = log_post_new - log_post_old - (K_new - K_old)

    # uniform [0,1] per particle
    u = tf.math.log(tf.random.uniform((Np,), dtype=dtype))
    accept = u < log_alpha
    accept = tf.cast(accept, dtype)

    # choose new or old particle
    x_final = accept[:, None] * x_new + (1 - accept)[:, None] * x
    return x_final

    
def smhmc_helper(
    Y,
    Np,
    prop_fn,
    log_likelihood_fn,
    resampling_fn = multinomial_resampling,
    leapfrog_steps=5,
    epsilon=0.05,
    resample_threshold=False,
    dtype=tf.float32
):
    """
    Sequential Monte Carlo with HMC rejuvenation (SmHMC)

    Consistent interface with bpf_generic:
        - prop_fn(x): next state sample
        - log_likelihood_fn(particles, y) log p(y|x)

    Returns:
        ests        : state estimates [T,d]
        ESS_list    : effective sample size [T]
        trace_list  : predictive covariance trace [T]
    """
    if len(Y.shape) != 2:
        raise ValueError("Y must have shape (T, d)")

    if Y.shape[0] < 1:
        raise ValueError("Y must contain at least one observation")

    # --- Np ---
    if not isinstance(Np, int) or Np <= 1:
        raise ValueError("Np must be an integer larger than 1")

    # --- Functions ---
    for name, fn in {
        "prop_fn": prop_fn,
        "log_likelihood_fn": log_likelihood_fn,
        "resampling_fn": resampling_fn,
    }.items():
        if not callable(fn):
            raise TypeError(f"{name} must be callable")

    # --- HMC params ---
    if not isinstance(leapfrog_steps, int) or leapfrog_steps <= 0:
        raise ValueError("leapfrog_steps must be a positive integer")

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")


#    start_time = time.time()

    Y = tf.cast(tf.convert_to_tensor(Y), dtype)
    
    T = tf.shape(Y)[0]
    d = tf.shape(Y)[1]

    particles = tf.zeros([Np, d], dtype=dtype)
    weights = tf.ones([Np], dtype=dtype) / tf.cast(Np, dtype)

    ests = tf.TensorArray(dtype, size=T)
    ESS_list = tf.TensorArray(dtype, size=T)

    threshold_value = Np/2.0

    for t in tf.range(T):
        y_t = Y[t]

        # -------------------
        # Propagate particles
        # -------------------
        particles_pred = tf.TensorArray(dtype, size=Np)

        for i in tf.range(Np):
            particles_pred = particles_pred.write(i, prop_fn(particles[i]))

        particles_pred = particles_pred.stack()

        # ------------------
        # Weight update
        # ------------------
        log_w = log_likelihood_fn(particles_pred, y_t)
        log_w -= tf.reduce_logsumexp(log_w)
        weights = tf.exp(log_w)

        # ------------------
        # Estimation 
        # ------------------
        est = tf.reduce_sum(particles_pred * weights[:, None], axis=0)
        ests = ests.write(t, est)

        # ------------------
        # ESS
        # ------------------
        ESS = 1.0 / tf.reduce_sum(weights**2)
        ESS_list = ESS_list.write(t, ESS)

        # ------------------
        # Resampling
        # ------------------
        do_resample = ESS < threshold_value if resample_threshold else True
        if do_resample:
            particles, weights = resampling_fn(particles_pred, weights)
        else:
            particles = particles_pred

        ### REJUVENATION
        particles = helper_hmc_rej_vectorized(
            particles=particles,
            y_t=y_t,
            log_likelihood_fn=log_likelihood_fn,
            leapfrog_steps=leapfrog_steps,
            epsilon=epsilon,
            dtype=dtype
        )

    return ests.stack(), ESS_list.stack() 


def bpf_block(
    Y,
    Np,
    prop_fn,
    log_likelihood_fn,
    resampling_fn=multinomial_resampling,
    block_size=3,
    resample_threshold=False,
    dtype=tf.float32,
):

    # --- Tensor checks ---
    if not tf.is_tensor(Y):
        raise TypeError("Y must be a TensorFlow tensor")

    if len(Y.shape) != 2:
        raise ValueError("Y must have shape (T, d)")

    if Y.shape[0] < 1:
        raise ValueError("Y must contain at least one observation")

    # --- Np ---
    if not isinstance(Np, int) or Np <= 1:
        raise ValueError("Np must be an integer larger than 1")

    # --- block_size ---
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be an integer >= 1")

    # --- Functions ---
    for name, fn in {
        "prop_fn": prop_fn,
        "log_likelihood_fn": log_likelihood_fn,
        "resampling_fn": resampling_fn,
    }.items():
        if not callable(fn):
            raise TypeError(f"{name} must be callable")

    
    Y = tf.cast(tf.convert_to_tensor(Y), dtype)
#    T, d = Y.shape
    T = tf.shape(Y)[0]
    d=tf.shape(Y)[1]

    particles = tf.zeros([Np, d], dtype=dtype)
    weights = tf.ones([Np], dtype=dtype) / tf.cast(Np, dtype)

    ests = tf.TensorArray(dtype=dtype, size=T)
    ESSs = tf.TensorArray(dtype=dtype, size=T)

    start_time = time.time()

    t = 0
    while t < T:

        block_end = min(t + block_size, T)

        # ---- accumulate block weights ----
#        log_w_block = tf.zeros([Np], dtype=dtype)
        log_w_block = tf.math.log(weights)

        k = t
        while k < block_end:

            y_t = tf.cast(Y[k], dtype)

            # propagation
            particles = tf.vectorized_map(prop_fn, particles)

            # likelihood
            log_w_block += tf.cast(log_likelihood_fn(particles, y_t), dtype)

            k += 1

        # ---- normalize weights after block ----
        log_w_block -= tf.reduce_logsumexp(log_w_block)
        weights = tf.exp(log_w_block)

        # ---- estimate ----
        est = tf.reduce_sum(particles * weights[:, None], axis=0)

        ESS = 1.0 / tf.reduce_sum(weights**2)

        ests = ests.write(t, est)
        ESSs = ESSs.write(t, ESS)

        # ---- Resampling ----
        particles, weights = resampling_fn(particles, weights)

        t = block_end

    exec_time = time.time() - start_time
    print(f"Block PF executed in {exec_time:.3f}s")

    return ests.stack(), ESSs.stack()



def compute_spectral_norm(A, n_iter=5):
    """
    Approximation of spectral norm ||A||_2
    """
    d = tf.shape(A)[0]
    tf.debugging.assert_rank(A, 2)
    tf.debugging.assert_equal(d, tf.shape(A)[1])
    tf.debugging.assert_greater(n_iter, 0)
    
    v = tf.random.normal((d, 1), dtype=A.dtype)

    for _ in range(n_iter):
        v = tf.matmul(A, v)
        v /= tf.norm(v) #+ 1e-12

    Av = tf.matmul(A, v)
    return tf.norm(Av)


def particle_flow_pf_update_beta_batch(
    particles,      # [Np, d]
    P_pred,         # [d, d]
    R_mat,          # [d, d]
    z,              # [d]
    flow_type="EDH",
    N_lambda=50,
    dtype=tf.float32,
    dl=None,
    beta=None,      # optional non-uniform integration schedule
    h_func=None,
    jacobian_func=None,
    diagnostics=True
):
    """
    LEDH batch update: fully vectorized over particles and integration steps.
    EDH remains the same (global Ai/bi).
    Approximate EDH/LEDH particle flow using a frozen linearization and fixed flow field over lambda.

    Input
    ----------
    particles : tf.Tensor, shape [Np, d]
        Predictive particles before the flow update.

    P_pred : tf.Tensor, shape [d, d]
        Predictive covariance matrix, usually obtained from EKF/UKF/KF prediction.

    R_mat : tf.Tensor, shape [d, d]
        Observation noise covariance matrix.

    z : tf.Tensor, shape [d]
        Current observation.

    flow_type : {"EDH", "LEDH"}, default="EDH"
        Type of particle flow to apply.

    N_lambda : int, default=50
        Number of integration steps for the artificial flow time.

    dtype : tf.DType, default=tf.float32
        Numerical dtype used internally.

    dl : float or None, default=None
        Uniform step size. If None, uses 1 / N_lambda.

    beta : tf.Tensor or None, shape [N_lambda + 1], default=None
        Optional non-uniform integration schedule from 0 to 1.
        If provided, integration steps are computed as beta[1:] - beta[:-1].

    h_func : callable
        Observation function h(x). Must accept a state or batch of states.

    jacobian_func : callable
        Jacobian of the observation function. Must return H(x).

    diagnostics : bool, default=True
        If True, returns flow diagnostics in addition to updated particles.

    Output
    -------
    If diagnostics=True:
        eta : tf.Tensor, shape [Np, d]
            Updated particles after the flow.

        theta : tf.Tensor, scalar
            Approximate Jacobian correction factor.

        logdet_J : tf.Tensor, scalar
            Log-determinant of the approximate flow Jacobian.

        flow_norm : tf.Tensor, shape [Np]
            Particle-wise flow magnitude diagnostic.

        spec_J : tf.Tensor, scalar
            Spectral norm diagnostic of the linearized flow matrix.

        cond_J : tf.Tensor, scalar
            Condition number diagnostic of the linearized flow matrix.

    If diagnostics=False:
        eta : tf.Tensor, shape [Np, d]
            Updated particles after the flow.

        theta : tf.Tensor, scalar
            Approximate Jacobian correction factor.
    
    """
    # ----- INPUT VALIDATION and CASTING ----- #
    if beta is not None and len(beta.shape) != 1:
        raise ValueError("beta must be a 1D tensor")

    if flow_type.upper() not in ["EDH", "LEDH"]:
        raise ValueError("flow_type must be 'EDH' or 'LEDH'")

    
    eta = tf.cast(particles, dtype)
    P_pred = tf.cast(P_pred, dtype)
    z = tf.cast(z, dtype)
    R_mat = tf.cast(R_mat, dtype)
    Np, d = tf.shape(eta)[0], tf.shape(eta)[1]

    # ------------------------------
    # Integration schedule in lambda ∈ [0,1]
    # ------------------------------
    if beta is not None:
        # Non-uniform schedule (e.g. optimized homotopy)
        beta = tf.cast(beta, dtype)
        steps = beta[1:] - beta[:-1]  
        N_lambda = tf.shape(steps)[0]
    else:
        if dl is None:
            dl = 1.0 / N_lambda
        steps = tf.ones(N_lambda, dtype=dtype) * dl  # uniform steps

    # ------------------------------
    # Linearization of the observation model
    # ------------------------------
    # h(x) linearised around the particle mean (EKF-style)
    # This is the key approximation: H is kept fixed across lambda
    x_mean = tf.reduce_mean(eta, axis=0)
    h_mean = h_func(x_mean)
    H_mean = jacobian_func(x_mean)
    
    # Innovation covariance (same as Kalman filter)
    S_global = H_mean @ P_pred @ tf.transpose(H_mean) + R_mat
    
    # Cross covariance
    PHt_global = P_pred @ tf.transpose(H_mean)
    
    # Drift matrix of the flow (Daum-Huang linear term)
    Ai_global = -0.5 * PHt_global @ tf.linalg.inv(S_global) @ H_mean

    
    # ===============================
    # EDH: global affine flow
    # ===============================
    if flow_type.upper() == "EDH":
        # Global innovation evaluated at mean
        dz_global = tf.reshape(z - h_mean, (d, 1))
        
        # Constant shift term b
        bi_global = tf.reshape(P_pred @ tf.transpose(H_mean) @ tf.linalg.solve(S_global, dz_global), (d,))
        
        # Flow is affine: f(x) = A x + b
        flow = eta @ tf.transpose(Ai_global) + bi_global[None, :]
        
        # Explicit Euler integration of dx/dλ = f(x)
        # RMK: flow is NOT recomputed - frozen dynamics approximation
        for k in tf.range(N_lambda):
            eta += steps[k] * flow

    # ===============================
    # LEDH: particle-wise correction
    # ===============================
    else:
        # Kalman gain (used for particle-wise residuals)
        K_global = PHt_global @ tf.linalg.inv(S_global)
        
        Ai_T = tf.transpose(Ai_global)
        KT = tf.transpose(K_global)

        # Evaluate observation function per particle
        h_particles = h_func(eta)

        # Particle-wise innovations
        dz_particles = z[None, :] - h_particles       # [Np, d]
        # Particle-dependent shift term
        bi_particles = dz_particles @ KT              # [Np, d]

        # Compute flow ONCE: still linear in x but with particle-specific bias
        flow = eta @ Ai_T + bi_particles   # [Np, d]

        # AgaFrozen flow, no recomputation across lambda
        # Sequential integration over β
        for k in tf.range(N_lambda):
            eta = eta + steps[k] * flow

    # ------------------------------
    # Jacobian determinant approximation
    # ------------------------------
    # True flow requires exp(∫ A dλ), but we approximate:
    # exp(A Δλ) ≈ I + Δλ A  (first-order Taylor)
    J_total = tf.eye(d, dtype=dtype) + tf.reduce_sum(steps) * Ai_global
    _, logdet_J = tf.linalg.slogdet(J_total)
    theta = tf.exp(logdet_J)   

    # ------------------------------
    # Diagnostics (stability + stiffness)
    # ------------------------------
    if diagnostics:
        # Norm of linear flow contribution
        flow_norm = tf.norm(eta @ tf.transpose(Ai_global), axis=1)

        # Linearized Jacobian of flow
        J_tilde = Ai_global if flow_type.upper() == "EDH" else Ai_global - K_global @ H_mean

        # Spectral norm: stiffness indicator
        spec_J = compute_spectral_norm(J_tilde)

        # Condition number: numerical stability
        svals_j = tf.linalg.svd(J_tilde, compute_uv=False)
        cond_J = svals_j[0] / (svals_j[-1] + 1e-12)
        return eta, theta, logdet_J, flow_norm, spec_J, cond_J
    else:
        return eta, theta



def particle_flow_pf_vectorized_propagation(
    measurements,      
    Np,
    P_pred,            
    R_mat,             # [d,d] or [Np,d,d]
    prop_fn,
    flow_update_fn,
    log_likelihood_fn,
    h_func, 
    jacobian_func,
    flow_type="EDH",
    use_weights=False,
    measurement_type="gaussian",
    diagnostics=True,
    prop_fn_stoch_drift=None,
    resampling_fn=None, 
    flow_constant=True,
    use_fixed_prop_noise=False,
    prop_noise_bank=None,
    loglik_weight_helper=None,
    collect_resampling_examples=None,
    dtype=tf.float32
):

    """
    Unified approximate EDH/LEDH particle-flow particle filter with optional weighting,
    resampling, stochastic propagation, fixed-noise propagation, and diagnostics.

    Input
    ----------
    measurements : tf.Tensor, shape [T, d]

    Np : int
        Number of particles.

    P_pred : tf.Tensor, shape [T, d, d] or compatible
        Predictive covariance matrices, usually supplied by KF/EKF/UKF.
        At time t, P_pred[t] is passed to the flow update.

    R_mat : tf.Tensor, shape [d, d] or [Np, d, d]
        Observation covariance. For Gaussian observations this is used directly.
        For Poisson observations, the function builds a local diagonal covariance
        from the predicted observation intensity.

    prop_fn : callable
        Deterministic propagation function applied particle-wise.
        Expected signature:
            prop_fn(x) -> x_next
        where x has shape [d].

    flow_update_fn : callable
        Particle-flow update function. Expected to implement the EDH/LEDH update.
        Expected signature:
            flow_update_fn(particles, P_pred_t, R_mat, z, flow_type, h_func, jacobian_func, diagnostics)

    log_likelihood_fn : callable
        Log-likelihood function used for optional PF-PF weighting.
        Expected signature:
            log_likelihood_fn(particles, z) -> log_like
        particles has shape [Np, d] and log_like has shape [Np].

    h_func : callable
        Observation function h(x). Must support both single-particle and batched
        particle inputs depending on the flow/measurement mode.

    jacobian_func : callable
        Jacobian of the observation function. Expected signature:
            jacobian_func(x, t) -> H_t
        or compatible through the internal lambda wrapper.

    flow_type : {"EDH", "LEDH"}, default="EDH"
        Type of deterministic particle flow.
        - "EDH": global flow based on a global linearization/covariance.
        - "LEDH": locally adapted flow; if a particle-wise R is available, the
          update is applied particle by particle.

    use_weights : bool, default=False
        If False, returns an unweighted particle-flow filter estimate.
        If True, applies likelihood-based importance weights after the flow,
        including the approximate Jacobian correction theta.

    measurement_type : {"gaussian", "poisson"}, default="gaussian"
        Observation model used to construct the covariance in the flow step.
        - "gaussian": use R_mat directly.
        - "poisson": build a diagonal local covariance from h_func(particles).

    diagnostics : bool, default=True
        If True, collects flow diagnostics:
        flow norm, spectral norm, condition number, log determinant, and loglik.

    prop_fn_stoch_drift : callable or None, default=None
        Optional stochastic propagation function.
        If provided, it replaces prop_fn during propagation.
        Expected signatures:
            prop_fn_stoch_drift(x, t)
        or, when use_fixed_prop_noise=True:
            prop_fn_stoch_drift(x, t, eps_t)

    resampling_fn : callable or None, default=None
        Optional resampling function applied after weighting.
        Expected signature:
            resampling_fn(particles, weights) -> new_particles, new_weights

    flow_constant : bool, default=True
        Deprecated/backward-compatibility argument. It is ignored.

    use_fixed_prop_noise : bool, default=False
        If True, propagation uses pre-generated noise from prop_noise_bank.
        This is useful for common-random-number experiments or reproducibility.

    prop_noise_bank : tf.Tensor or None, shape [T, Np, d], default=None
        Fixed propagation noise, can be used only when use_fixed_prop_noise=True.

    loglik_weight_helper : callable or None, default=None
        Optional custom weighting/log-likelihood correction.
        Expected signature:
            loglik_weight_helper(log_like, theta, Np, dtype) -> weights, loglik_t
        Useful for non-standard proposal corrections or stochastic-flow variants.

    collect_resampling_examples : list or None, default=None
        Optional Python list used to store `(particles, weights)` pairs before
        resampling. Useful for neural or accelerated resampling experiments.
        Prefer eager execution if using this option.

    dtype : tf.DType, default=tf.float32
        Numerical dtype used internally.

    Returns
    -------
    ests : tf.Tensor, shape [T, d]
        Filtered state estimates.

    ESS : tf.Tensor or None, shape [T]
        Effective sample size over time. Returned only if use_weights=True;
        otherwise None.

    particles : tf.Tensor, shape [Np, d]
        Final particle ensemble.

    diagnostics_dict : dict
        Dictionary containing:
            "loglik" : scalar tf.Tensor
        and, if diagnostics=True:
            "flow_norm" : tf.Tensor
            "spec_J"    : tf.Tensor
            "cond_J"    : tf.Tensor
            "logdet_J"  : tf.Tensor

    Notes
    -----
    This implementation is approximate because the underlying flow update 
    relies on a frozen linearization and an approximate Jacobian correction.
    """
    
    del flow_constant  # kept only for backward compatibility

    # ------------------------------
    # Input validation
    # ------------------------------
    if len(measurements.shape) != 2:
        raise ValueError("measurements must have shape (T, d)")

    if measurements.shape[0] < 1:
        raise ValueError("measurements must contain at least one observation")

    if not isinstance(Np, int) or Np <= 1:
        raise ValueError("Np must be an integer > 1")

    for name, fn in {
        "prop_fn": prop_fn,
        "flow_update_fn": flow_update_fn,
        "log_likelihood_fn": log_likelihood_fn,
        "h_func": h_func,
        "jacobian_func": jacobian_func,
    }.items():
        if not callable(fn):
            raise TypeError(f"{name} must be callable")

    if not isinstance(use_weights, bool):
        raise TypeError("use_weights must be boolean")

    if not isinstance(diagnostics, bool):
        raise TypeError("diagnostics must be boolean")

    if flow_type.upper() not in ["EDH", "LEDH"]:
        raise ValueError("flow_type must be 'EDH' or 'LEDH'")

    if measurement_type.lower() not in ["gaussian", "poisson"]:
        raise ValueError("measurement_type must be 'gaussian' or 'poisson'")

    if use_fixed_prop_noise and prop_noise_bank is None:
        raise ValueError("prop_noise_bank must be provided when use_fixed_prop_noise=True")

    if use_fixed_prop_noise and prop_fn_stoch_drift is None:
        raise ValueError("prop_fn_stoch_drift must be provided when use_fixed_prop_noise=True")

    
    measurements = tf.cast(measurements, dtype)
    P_pred = tf.cast(P_pred, dtype)
    R_mat = tf.cast(R_mat, dtype)

    if use_fixed_prop_noise:
        prop_noise_bank = tf.cast(prop_noise_bank, dtype)
        if len(prop_noise_bank.shape) != 3:
            raise ValueError("prop_noise_bank must have shape [T, Np, d]")

    T = tf.shape(measurements)[0]
    d = tf.shape(measurements)[1]
    
    # Particles initialised at zero.
    # The model dynamics are then applied through prop_fn or prop_fn_stoch_drift.
    particles = tf.zeros((Np, d), dtype=dtype)
    
    # Allocation
    ests = tf.TensorArray(dtype, size=T)
    ESS_list = tf.TensorArray(dtype, size=T)

    flow_norm_ta = tf.TensorArray(dtype, size=T)
    spec_J_ta = tf.TensorArray(dtype, size=T)
    cond_J_ta = tf.TensorArray(dtype, size=T)
    logdet_J_ta = tf.TensorArray(dtype, size=T)

    total_loglik = tf.constant(0.0, dtype=dtype)

    for t in tf.range(T):
        z = measurements[t]
        
        # P_pred in the paper is obtained via EKF/UKF prediction.
        # It provides the Gaussian covariance used by the flow update.
        P_pred_t = tf.gather(P_pred, t)

        # ------------------------
        # Conditional propagation
        # ------------------------
        # For Gaussian observations, the first update can be applied directly
        # to the initial particles. For Poisson observations, propagation is
        # also applied at t=0 to avoid degeneracy at the origin.
        propagate_now = measurement_type.lower() == "poisson" or t > 0

        if propagate_now:
            if prop_fn_stoch_drift is not None:
                # Optional stochastic propagation with explicit noise input.
                # Useful when we want common random numbers or fixed noise.
                # This will be used later for the DBP-HMC
                if use_fixed_prop_noise:
                    eps_t = prop_noise_bank[t]
                    particles = tf.map_fn(
                        lambda args: prop_fn_stoch_drift(args[0], t, args[1]),
                        (particles, eps_t),
                        fn_output_signature=tf.TensorSpec(shape=(None,), dtype=dtype)
                    )
                else:
                    particles = tf.vectorized_map(lambda x: prop_fn_stoch_drift(x, t), particles)
            else:
                # Standard propagation.
                particles = tf.vectorized_map(prop_fn, particles)

        # ------------------------
        # Determine measurement covariance matrix
        # ------------------------
        if measurement_type.lower() == "gaussian":
            R_use = R_mat
            
        else:  # Poisson
            # Use a local Gaussian approximation: Var(y | x) ≈ E[y | x].
            y_pred = h_func(particles)
            
            if flow_type.upper() == "EDH":
                # EDH uses one global covariance based on the ensemble mean rate
                R_use = tf.linalg.diag(tf.reduce_mean(y_pred, axis=0))
            else:  
                # LEDH may use particle-specific covariance matrices.
                R_use = tf.linalg.diag(y_pred)

        # ------------------------
        # Flow update
        # ------------------------
        if flow_type.upper() == "LEDH" and len(R_use.shape) == 3 and R_use.shape[0] == Np:

            def update_single_particle(args):
                particle_i, R_i = args
                # LEDH particle-specific update:
                # each particle gets its own local observation covariance R_i.
                result = flow_update_fn(
                    tf.expand_dims(particle_i, 0),
                    P_pred_t,
                    R_mat=R_i,
                    z=z,
                    flow_type="LEDH",
                    h_func=h_func,
                    jacobian_func=lambda x: jacobian_func(x, t),
                    diagnostics=diagnostics
                )

                if diagnostics:
                    updated_particle, updated_theta, logdet_J, flow_norm, spec_J, cond_J = result

                    logdet_J = tf.reshape(logdet_J, [1]) if len(logdet_J.shape) == 0 else logdet_J
                    flow_norm = tf.reshape(flow_norm, [1]) if len(flow_norm.shape) == 0 else flow_norm
                    spec_J = tf.reshape(spec_J, [1]) if len(spec_J.shape) == 0 else spec_J
                    cond_J = tf.reshape(cond_J, [1]) if len(cond_J.shape) == 0 else cond_J

                    return (
                        tf.squeeze(updated_particle, 0),
                        updated_theta,
                        logdet_J,
                        flow_norm,
                        spec_J,
                        cond_J
                    )
                else:
                    updated_particle, updated_theta = result
                    return tf.squeeze(updated_particle, 0), updated_theta

            if diagnostics:
                updated_results = tf.map_fn(
                    update_single_particle,
                    (particles, R_use),
                    dtype=(dtype, dtype, dtype, dtype, dtype, dtype)
                )
                particles = updated_results[0]
                theta = updated_results[1]
                logdet_J, flow_norm, spec_J, cond_J = updated_results[2:6]
            else:
                updated_particles, updated_thetas = tf.map_fn(
                    update_single_particle,
                    (particles, R_use),
                    dtype=(dtype, dtype)
                )
                particles = updated_particles
                theta = updated_thetas

                logdet_J = tf.constant(float('nan'), dtype=dtype)
                flow_norm = tf.fill([Np], tf.constant(float('nan'), dtype=dtype))
                spec_J = tf.constant(float('nan'), dtype=dtype)
                cond_J = tf.constant(float('nan'), dtype=dtype)

        else:  
            # EDH or LEDH with global covariance.
            # The supplied flow_update_fn performs the approximate EDH/LEDH transport
            result = flow_update_fn(
                particles,
                P_pred_t,
                R_mat=R_use,
                z=z,
                flow_type=flow_type,
                h_func=h_func,
                jacobian_func=lambda x: jacobian_func(x, t),
                diagnostics=diagnostics
            )

            if diagnostics:
                particles, theta, logdet_J, flow_norm, spec_J, cond_J = result
            else:
                particles, theta = result

                logdet_J = tf.constant(float('nan'), dtype=dtype)
                flow_norm = tf.fill([Np], tf.constant(float('nan'), dtype=dtype))
                spec_J = tf.constant(float('nan'), dtype=dtype)
                cond_J = tf.constant(float('nan'), dtype=dtype)

        # ------------------------
        # Optional Weighting
        # ------------------------
        if use_weights:
            # Evaluate likelihood after the deterministic flow.
            # The theta factor accounts approximately for the flow Jacobian.
            log_like = tf.cast(log_likelihood_fn(particles, z), dtype)
            tf.debugging.assert_all_finite(particles, "particles non-finite before weighting")
            tf.debugging.assert_all_finite(log_like, "log_like non-finite before weighting")

            if loglik_weight_helper is None:
                # BASELINE BEHAVIOR
                # normalize likelihoods and multiply by flow Jacobian correction.
                loglik_t = tf.reduce_logsumexp(log_like) - tf.math.log(tf.cast(Np, dtype=dtype))
                total_loglik += loglik_t

                log_w = log_like - loglik_t
                weights = tf.exp(log_w) * theta
                weights /= tf.reduce_sum(weights)

            else:
                # OPTIONAL EXTERNAL LIKELIHOOD CORRECTION 
                # e.g. for alternative proposals or stochastic-flow/HMC-style weighting.
                # expected output: weights, loglik_t
                weights, loglik_t = loglik_weight_helper(
                    log_like=log_like,
                    theta=theta,
                    Np=Np,
                    dtype=dtype,
                )
                total_loglik += loglik_t

            ESS_list = ESS_list.write(t, 1.0 / tf.reduce_sum(weights**2))

            # Optional samples collection for neural/accelerated resampling training data
            if collect_resampling_examples is not None:
                collect_resampling_examples.append((
                    tf.identity(particles),
                    tf.identity(weights)
                ))

            # optional resampling
            if resampling_fn is not None:
                particles, weights = resampling_fn(particles, weights)

        else:
            # Unweighted particle flow filter:
            # all particles contribute equally after transport.
            weights = tf.ones(Np, dtype=dtype) / tf.cast(Np, dtype)

        # ------------------------
        # State estimate
        # ------------------------
        ests = ests.write(t, tf.reduce_sum(particles * tf.reshape(weights, (Np, 1)), axis=0))

        # ------------------------
        # Collect diagnostics
        # ------------------------
        if diagnostics:
            flow_norm_ta = flow_norm_ta.write(t, flow_norm)
            spec_J_ta = spec_J_ta.write(t, spec_J)
            cond_J_ta = cond_J_ta.write(t, cond_J)
            logdet_J_ta = logdet_J_ta.write(t, logdet_J)

    diagnostics_dict = None
    if diagnostics:
        diagnostics_dict = {
            "flow_norm": flow_norm_ta.stack(),
            "spec_J": spec_J_ta.stack(),
            "cond_J": cond_J_ta.stack(),
            "logdet_J": logdet_J_ta.stack(),
            "loglik": total_loglik
        }
    else:
        diagnostics_dict = {"loglik": total_loglik}

    return ests.stack(), ESS_list.stack() if use_weights else None, particles, diagnostics_dict



# PRE-EXECUTION FUNCTIONS

def run_ekf_wrap(Y, m0, P0, Q, R, F, H, F_jac, H_jac, measurement_type="gaussian"):
    # 1. Build EKF kernels
    predict_fn, update_fn = make_ekf_kernels(
        F_func=F,
        H_func=H,
        F_jac=F_jac,
        H_jac=H_jac,
        Q=Q
    )
    # 2. Run filter
    result = _filter_core(
        Y=Y,
        predict_fn=predict_fn,
        update_fn=update_fn,
        R_mat=R,
        m0=m0,
        P0=P0,
        measurement_type=measurement_type,
        dtype=Y.dtype
    )

    return result 


def run_kf_wrap(Y, m0, P0, Q, R, F_mat, H_mat, measurement_type="gaussian"):
    # 1. Build KF kernels
    predict_fn, update_fn = make_kf_kernels(
        F_mat,
        H_mat,
        Q
    )
    # 2. Run filter
    result = _filter_core(
        Y=Y,
        predict_fn=predict_fn,
        update_fn=update_fn,
        R_mat=R,
        m0=m0,
        P0=P0,
        measurement_type=measurement_type,
        dtype=Y.dtype
    )
    # 3. Return filtered mean - transpose so to be compatible with the simulated model
    return tf.transpose(result["mu_filt"])

def run_ukf_wrap(
    Y, m0, P0, Q, R,
    F, H,
    measurement_type="gaussian",
    alpha=1e-3, beta=2.0, kappa=0.0,
    dtype=tf.float64
):
    # 1. Build UKF kernels
    ukf_predict_fn, ukf_update_fn = make_ukf_kernels(
        F_func=F,
        H_func=H,
        Q=Q,
        alpha=alpha,
        beta=beta,
        kappa=kappa
    )
    # 2. Run filter
    result = _filter_core(
        Y=Y,
        predict_fn=ukf_predict_fn,
        update_fn=ukf_update_fn,
        R_mat=R,
        m0=m0,
        P0=P0,
        measurement_type=measurement_type,
        dtype=dtype
    )
    # 3. Return filtered mean - transpose so to be compatible with the simulated model
    return tf.transpose(result["mu_filt"])


def run_bpf(Y, Np, prop_fn, log_likelihood_fn, resampling_fn=multinomial_resampling):
    return bpf_generic_resampling( 
        Y=Y,
        Np=Np,
        resampling_fn=resampling_fn,
        prop_fn=prop_fn, 
        log_likelihood_fn=log_likelihood_fn
    )[:2]


def run_upf(Y, Np, Sigma, alpha, nu, gamma, transition_mean_fn, transition_logpdf_fn, log_likelihood_fn):
    return upf_filter(
        Y=Y,
        Np=Np,
        alpha_dyn=alpha,
        Sigma=Sigma,
        gamma=gamma,
        nu=nu,
        Q_proposal=Sigma,
        transition_mean_fn=transition_mean_fn, 
        transition_logpdf_fn=transition_logpdf_fn, 
        log_likelihood_fn=log_likelihood_fn 
    )

def run_gsmc(Y, Np, prop_fn, log_likelihood_fn):
    """
    Wrapper for gsmc_general using a linear Gaussian propagation
    """
    return gsmc_general(
        Y=Y,
        Np=Np,
        prop_fn=prop_fn,  
        log_likelihood_fn=log_likelihood_fn  
    )

def run_smhmc(Y, Np, prop_fn, log_likelihood_fn, leapfrog_steps=5, epsilon=0.05, resample_threshold=True):  
    """
    Wrapper smhmc_helper filter
    """
    return smhmc_helper(
        Y=Y,
        Np=Np,
        prop_fn=prop_fn,
        log_likelihood_fn=log_likelihood_fn,
        leapfrog_steps=leapfrog_steps,
        epsilon=epsilon,
        resample_threshold=resample_threshold,
        dtype=dtype
    )

def run_esrf(Y, Np, F_func, Q, H_func, R, measurement_type="gaussian", dtype=None):
    """
    Wrapper for esrf_filter using given transition and observation functions
    """
    # Make dtype optional and in case infer from data (pipeline-independent)
    if dtype is None:
        dtype = Y.dtype  

    return esrf_filter(
        measurements=Y,
        Np=Np,
        F_func=F_func,
        Q=Q,
        H_func=H_func,
        R=R,
        measurement_type=measurement_type,
        dtype=dtype   
    )


def run_pfpf_fn(flow_type="EDH", measurement_type="gaussian", use_weights=True, beta=None, resampling_fn=None, diagnostics=True,
    prop_fn_stoch_drift=None,
    use_fixed_prop_noise=False,
    prop_noise_bank=None,
    loglik_weight_helper=None,
    collect_resampling_examples=None,
    compile_tf=True,
):
    """
    Factory for building a configured Particle Flow Particle Filter (PFPF).

    Returns a callable `pfpf_fn` that runs a full filtering pass using EDH or LEDH
    flows, with optional weighting, resampling, and custom propagation.

    Input
    ----------
    flow_type : str
        "EDH" (global) or "LEDH" (local linearization).
    measurement_type : str
        "gaussian" or "poisson".
    use_weights : bool
        Enable importance weighting (and optional resampling).
    beta : tf.Tensor or None
        Optional non-uniform flow integration schedule.
    resampling_fn : callable or None
        Optional resampling method.
    diagnostics : bool
        If True, return flow-related diagnostics.
    compile_tf : bool
        If True, compile with `tf.function`.

    Output
    -------
    pfpf_fn : callable
        Filter runner returning (ests, ESS, particles, diagnostics).

    Notes
    -----
    - This is a factory: it configures the filter but does not run it.
    - Set `compile_tf=False` for debugging or samples collection.
    """

    if collect_resampling_examples is not None and compile_tf:
        raise ValueError(
            "collect_resampling_examples is intended for eager mode. "
            "Use compile_tf=False."
        )

    def pfpf_fn(
        Y,
        Np,
        P_pred,
        R_mat,
        prop_fn,
        log_likelihood_fn,
        h_func,
        jacobian_func,
        resampling_fn=resampling_fn,
        prop_fn_stoch_drift=prop_fn_stoch_drift,
        use_fixed_prop_noise=use_fixed_prop_noise,
        prop_noise_bank=prop_noise_bank,
        diagnostics=diagnostics,
        loglik_weight_helper=loglik_weight_helper,
        collect_resampling_examples=collect_resampling_examples,
    ):
        
        flow_update_fn_beta = lambda particles, P_pred_t, R_mat, z, flow_type, h_func, jacobian_func, diagnostics: \
            particle_flow_pf_update_beta_batch(
                particles,
                P_pred_t,
                R_mat,
                z,
                flow_type=flow_type,
                beta=beta,
                h_func=h_func,
                jacobian_func=jacobian_func,
                diagnostics=diagnostics
            )

        return particle_flow_pf_vectorized_propagation(
            measurements=Y,
            Np=Np,
            P_pred=P_pred,
            R_mat=R_mat,
            prop_fn=prop_fn,
            prop_fn_stoch_drift=prop_fn_stoch_drift,
            flow_update_fn=flow_update_fn_beta,
            log_likelihood_fn=log_likelihood_fn,
            h_func=h_func,
            jacobian_func=jacobian_func,
            flow_type=flow_type,
            use_weights=use_weights,
            measurement_type=measurement_type,
            diagnostics=diagnostics,
            resampling_fn=resampling_fn,
            use_fixed_prop_noise=use_fixed_prop_noise,
            prop_noise_bank=prop_noise_bank,
            loglik_weight_helper=loglik_weight_helper,
            collect_resampling_examples=collect_resampling_examples
        )

    return tf.function(pfpf_fn, reduce_retracing=True) if compile_tf else pfpf_fn