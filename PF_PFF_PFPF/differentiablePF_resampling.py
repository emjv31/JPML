import os
import time
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from simulator_model_comps import pairwise_distance
from replicate_Li_filters import make_kf_kernels
from EKF_UKF_SV_BPF.multiv_ekf_ukf_core import _filter_core

from EKF_UKF_SV_BPF.multiv_sv_bpf_core import (
    multinomial_resampling,
    bpf_generic_resampling,
)



# RESAMPLING METHODS 

def mixture_unif_multinomial_resampling(particles, weights, alpha = 0.35):  
    particles = tf.convert_to_tensor(particles)
    dtype = particles.dtype
    weights = tf.convert_to_tensor(weights, dtype=dtype)

    N = tf.shape(particles)[0] 
    
    # Normalise weights
    w = weights/tf.reduce_sum(weights)
    unif = tf.ones_like(w)/tf.cast(N, dtype)
    w_mixture = (1-alpha)*w + alpha*unif

    return multinomial_resampling(particles, w_mixture)

def no_resampling(particles, weights):
    return particles, weights

def soft_resampling_pfnet(particles, weights, alpha=0.35, eps=1e-12):
    dtype = particles.dtype
    N = tf.shape(particles)[0]

    w = weights / (tf.reduce_sum(weights) + eps)
    u = tf.ones_like(w) / tf.cast(N, dtype)
    q = (1.0 - alpha) * w + alpha * u   # mixture weights

    idx = tf.random.categorical(tf.math.log(q[None, :] + eps), N)[0]
    new_particles = tf.gather(particles, idx)

    new_weights = tf.gather(w / (q + eps), idx)
    new_weights = new_weights / (tf.reduce_sum(new_weights) + eps)

    return new_particles, new_weights


#################################
### ROBUST OT COST
#################################

def robustify_cost(C, lambda_robust=5.0, mode="smooth_clip"):
    """
    Robust transformation of the OT cost matrix for Robust Optimal Transport (ROBOT)
    see ``Inference via robust optimal transportation: theory and methods'', by Y. Ma, H. Liu, 
    D. La Vecchia and M. Lerasle. Both Y. Ma for details.

    This allows robustness to heavy-tailed particles and outliers by
    bounding the transport cost.

    smooth_clip:
        C_rob = lambda * (1 - exp(-C / lambda))
        -> smooth and fully differentiable

    hard_clip:
        C_rob = min(C, lambda)
        -> stronger robustness but non-smooth

    none:
        standard OT cost
    """
    C = tf.convert_to_tensor(C)
    dtype = C.dtype
    lam = tf.cast(lambda_robust, dtype)

    tf.debugging.assert_positive(lam, message="lambda_robust must be > 0")

    if mode == "smooth_clip":
        return lam * (1.0 - tf.exp(-C / lam))

    elif mode == "hard_clip":
        return tf.minimum(C, lam)

    elif mode == "none":
        return C

    else:
        raise ValueError("robust_mode must be 'smooth_clip', 'hard_clip', or 'none'")


def sinkhorn_log_general(
    a,
    b,
    C,
    epsilon,
    n_iter,
    log_u_init=None,
    log_v_init=None,
    normalize_cost=False,
    clip_logK=None,
    return_duals=False,
):
    """
    Log-domain Sinkhorn solver for entropy-regularized OT.

    Supporting both:
      1) standard Sinkhorn
      2) neural-accelerated Sinkhorn via warm-start duals

    Input
    ----
    a, b : [N]
        Source and target weights, nonnegative, sum to 1.
    C : [N, N]
        Cost matrix.
    epsilon : scalar
        Entropic regularization strenght.
    n_iter : int
        Number of Sinkhorn iterations.
    log_u_init, log_v_init : [N] or None
        Optional warm start for dual variables.
    normalize_cost : bool
        If True, divide C by its mean.
    clip_logK : float or None
        Optional clip for log_K = -C/epsilon.
    return_duals : bool
        If True, return (T, log_u, log_v).

    Returns
    -------
     T : tf.Tensor, shape (N, N)
        Transport matrix.
    optionally:
        log_u, log_v
    """
    a = tf.convert_to_tensor(a)
    dtype = a.dtype

    b = tf.convert_to_tensor(b, dtype=dtype)
    C = tf.convert_to_tensor(C, dtype=dtype)
    epsilon = tf.cast(epsilon, dtype)

    atol = tf.cast(1e-5, dtype)
    tiny = tf.cast(1e-12, dtype)

    # -------------------------
    # Input checks
    # -------------------------
    tf.debugging.assert_positive(epsilon, message="epsilon must be > 0")

    tf.debugging.assert_equal(
        tf.shape(C)[0], tf.shape(C)[1],
        message="Cost matrix C must be square"
    )

    tf.debugging.assert_equal(
        tf.shape(a)[0], tf.shape(b)[0],
        message="a and b must have same length"
    )

    tf.debugging.assert_equal(
        tf.shape(C)[0], tf.shape(a)[0],
        message="Dimensions of C, a, b must match"
    )

    tf.debugging.assert_greater_equal(a, tf.cast(0.0, dtype), message="a must be >= 0")
    tf.debugging.assert_greater_equal(b, tf.cast(0.0, dtype), message="b must be >= 0")

    # -----------------------------------------------
    # Normalization of marginals
    # -----------------------------------------------
    a = tf.maximum(a, tiny)
    b = tf.maximum(b, tiny)
    a = a / tf.reduce_sum(a)
    b = b / tf.reduce_sum(b)

    tf.debugging.assert_near(
        tf.reduce_sum(a), tf.cast(1.0, dtype), atol=atol,
        message="a must sum to 1"
    )

    tf.debugging.assert_near(
        tf.reduce_sum(b), tf.cast(1.0, dtype), atol=atol,
        message="b must sum to 1"
    )
    # ------------------------------------------------------------------
    # Build log-kernel log_K = -C / epsilon
    # Optional cost normalization / clipping improves numerical stability
    # ------------------------------------------------------------------
    if normalize_cost:
        C = C / (tf.reduce_mean(C) + tiny)

    log_K = -C / epsilon

    if clip_logK is not None:
        clip_logK = tf.cast(clip_logK, dtype)
        log_K = tf.clip_by_value(log_K, -clip_logK, clip_logK)

    log_a = tf.math.log(a)
    log_b = tf.math.log(b)

    # --------------------------------------------
    # Initialise duals: Warm start or zero start
    # --------------------------------------------
    if log_u_init is None:
        log_u = tf.zeros_like(a)
    else:
        log_u = tf.cast(log_u_init, dtype)

    if log_v_init is None:
        log_v = tf.zeros_like(b)
    else:
        log_v = tf.cast(log_v_init, dtype)

    # -------------------------------------
    # Sinkhorn iterations in log-domains
    # ------------------------------------
    for _ in range(n_iter):
        log_u = log_a - tf.reduce_logsumexp(log_K + log_v[None, :], axis=1)
        log_v = log_b - tf.reduce_logsumexp(log_K + log_u[:, None], axis=0)

    # Recover transport plan
    log_T = log_K + log_u[:, None] + log_v[None, :]
    T = tf.exp(log_T)

    if return_duals:
        return T, log_u, log_v
    return T


##################
####  ROBOT 
##################
def soft_resample_ot(
    particles,
    weights,
    epsilon,
    sinkhorn_iters,
    log_u_init=None,
    log_v_init=None,
    normalize_cost=False,
    clip_logK=None,
    return_transport=False,
    return_duals=False,
    robust_cost=False,
    lambda_robust=5.0,
    robust_mode="smooth_clip",
):
    """
    Differentiable OT resampling via entropy-regularized Sinkhorn algorithm.

    This function supports:
      - Standard OT resampling
      - Neural-accelerated Sinkhorn via warm-started dual variables
      - Robust OT resampling (optional)

    In the robust setting, the quadratic transport cost C is replaced by a
    bounded transformation rho_lambda(C), which limits the influence of
    outlier particles and improves stability in heavy-tailed settings.

    Parameters
    ----------
    particles : tf.Tensor, shape (N, d)
        Particle ensemble.

    weights : tf.Tensor, shape (N,)
        Source particle weights.

    epsilon : scalar
        Entropic regularization strength.

    sinkhorn_iters : int
        Number of Sinkhorn iterations.

    log_u_init, log_v_init : tf.Tensor or None
        Optional warm-start dual variables (for neural acceleration).

    normalize_cost : bool
        If True, rescales the cost matrix by its mean.

    clip_logK : float or None
        Optional clipping of log_K = -C / epsilon.

    return_transport : bool
        If True, also return the transport matrix.

    return_duals : bool
        If True, also return final Sinkhorn duals.

    robust_cost : bool
        If True, apply robust cost transformation before Sinkhorn.

    lambda_robust : float
        Robustness parameter. Smaller values increase robustness but introduce
        more bias. Larger values recover standard OT behaviour.

    robust_mode : {"smooth_clip", "hard_clip", "none"}
        Type of robust cost transformation. "smooth_clip" is recommended
        as it preserves differentiability.

    Returns
    -------
    particles_new : tf.Tensor, shape (N, d)
        Resampled particles after barycentric transport.

    weights_new : tf.Tensor, shape (N,)
        Uniform weights after resampling.

    Optionally:
        T : transport matrix
        log_u, log_v : Sinkhorn dual variables
    """
    particles = tf.convert_to_tensor(particles)
    dtype = particles.dtype

    weights = tf.convert_to_tensor(weights, dtype=dtype)
    Np = tf.shape(particles)[0]

    # Normalize source weights
    w = tf.maximum(weights, tf.cast(1e-12, dtype))
    w = w / tf.reduce_sum(w)

    # Uniform target weights after resampling
    b = tf.ones_like(w) / tf.cast(Np, dtype)

    # Pairwise squared cost matrix
    C = pairwise_distance(particles, squared=True)

    # Robust cost transformation
    if robust_cost:
        C = robustify_cost(
            C,
            lambda_robust=lambda_robust,
            mode=robust_mode,
        )

    # Solve entropy-regularized OT
    outputs = sinkhorn_log_general(
        a=w,
        b=b,
        C=C,
        epsilon=epsilon,
        n_iter=sinkhorn_iters,
        log_u_init=log_u_init,
        log_v_init=log_v_init,
        normalize_cost=normalize_cost,
        clip_logK=clip_logK,
        return_duals=return_duals,
    )

    if return_duals:
        T, log_u, log_v = outputs
    else:
        T = outputs

    # Deterministic barycentric projection
    particles_new = tf.matmul(tf.transpose(T), particles) * tf.cast(Np, dtype)
    weights_new = b

    if return_transport and return_duals:
        return particles_new, weights_new, T, log_u, log_v
    elif return_transport:
        return particles_new, weights_new, T
    elif return_duals:
        return particles_new, weights_new, log_u, log_v
    else:
        return particles_new, weights_new




#################################
### TUNING the PARS of Sinkhorn
#################################

def run_bpf_ot(
    resampling_fn,
    Y=None,
    Np_value=None,
    prop_fn=None,
    log_likelihood_fn=None,
    dtype=tf.float32,
):
    if Y is None:
        Y = globals().get("measurements", None)
    if Np_value is None:
        Np_value = globals().get("Np", None)
    if prop_fn is None:
        prop_fn = globals().get("prop_fn_b", None)
    if log_likelihood_fn is None:
        log_likelihood_fn = globals().get("llk_fn_b", None)

    if Y is None or Np_value is None or prop_fn is None or log_likelihood_fn is None:
        raise ValueError(
            "run_bpf_ot needs Y, Np_value, prop_fn, and log_likelihood_fn. "
            "Pass them explicitly when running outside the notebook."
        )

    return bpf_generic_resampling(
        Y=Y,
        Np=Np_value,
        prop_fn=prop_fn,
        log_likelihood_fn=log_likelihood_fn,
        resampling_fn=resampling_fn,
        resample_threshold=False,
        dtype=dtype,
    )[:2]



### ROBUST TUNING 
@tf.function(reduce_retracing=True)
def _eval_ot_candidate(
    filter_fn,
    true_state,
    Np_t,
    epsilon,
    sinkhorn_iters,
    lambda_ess,
    robust_cost=False,
    lambda_robust=5.0,
    robust_mode="smooth_clip",
):
    """
    Evaluate one OT configuration.
    """
    
    ests, ESS_list = filter_fn(
        resampling_fn=lambda p, w: soft_resample_ot(
            p,
            w,
            epsilon=epsilon,
            sinkhorn_iters=sinkhorn_iters,
            normalize_cost=True,
            robust_cost=robust_cost,
            lambda_robust=tf.cast(lambda_robust, p.dtype),
            robust_mode=robust_mode,
        )
    )

    ests = tf.cast(ests, true_state.dtype)
    ESS_list = tf.cast(ESS_list, true_state.dtype)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ests - true_state)))
    mean_ess = tf.reduce_mean(ESS_list)
    ess_ratio = mean_ess / Np_t

    quality = rmse + lambda_ess * (1.0 - ess_ratio)

    return quality, rmse, mean_ess, ess_ratio, ests, ESS_list


def tune_ot_entropy_regularized(
    filter_fn,
    true_state,
    Np,
    niter_grid=(4, 6, 8, 10, 12),
    eps_grid=(1e-3, 1e-2, 1e-1, 1.0),
    lambda_ess=0.95,
    lambda_speed=0.05,
    n_repeats=1,
    robust_cost=False,
    lambda_robust=5.0,
    robust_mode="smooth_clip",
    dtype=tf.float32,
    seed=123,
):
    """
    Tune entropy-regularized Optimal Transport (OT) resampling parameters
    for a differentiable particle filter.

    This function performs a grid search over:
        - the Sinkhorn iteration count (niter_grid)
        - the entropic regularization strength epsilon (eps_grid)

    Optionally, a robust variant of OT can be used, where the transport
    cost is modified to reduce the influence of outlier particles.

    For each configuration, the particle filter is run (optionally multiple times)
    and evaluated using a trade-off between:
        - estimation accuracy (RMSE w.r.t. true_state)
        - particle diversity (via effective sample size, ESS)
        - computational cost (proxied by number of Sinkhorn iterations)

    The objective minimized is:
        score = RMSE + lambda_ess * (1 - ESS / Np) + lambda_speed * normalized_cost

    where:
        - RMSE measures state estimation error
        - ESS / Np measures particle degeneracy
        - normalized_cost = n_iter / max(niter_grid)

    If robust_cost=True, the OT cost matrix is transformed via a robust
    mapping (e.g., smooth clipping), which improves stability under
    heavy-tailed or outlier-contaminated particle distributions.

    The best configuration is selected based on the lowest score observed
    across all runs.

    Parameters
    ----------
    filter_fn : callable
        Function implementing the particle filter.
        Must accept `resampling_fn=` argument and return (estimates, ESS).
    true_state : array-like, shape (T, d)
        Ground truth latent states used to compute RMSE.
    Np : int
        Number of particles.
    niter_grid : iterable of int
        Candidate numbers of Sinkhorn iterations.
    eps_grid : iterable of float
        Candidate entropy regularization strengths.
    lambda_ess : float
        Weight for ESS penalty (higher → more emphasis on particle diversity).
    lambda_speed : float
        Weight for computational cost penalty.
    n_repeats : int
        Number of Monte Carlo repetitions per configuration.

    robust_cost : bool, optional
        If True, use robust OT cost instead of standard quadratic cost.
    lambda_robust : float, optional
        Robustness parameter controlling the degree of cost clipping/smoothing.
    robust_mode : str, optional
        Type of robust transformation:
            - "smooth_clip" (default)
            - "hard_clip"
            - "none"

    dtype : tf.DType
        TensorFlow dtype used for computations.
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    best_params : dict
        Best-performing hyperparameters and associated metrics.
        Includes robust parameters if enabled.
    best_results : tuple
        (estimates, ESS) corresponding to the best single run.
    results_table : list of dict
        Aggregated performance metrics for each (epsilon, n_iter) pair.
    """
    if true_state is None:
        raise ValueError("true_state cannot be None")

    if Np is None or Np <= 1:
        raise ValueError("Np must be > 1")

    if not niter_grid:
        raise ValueError("niter_grid cannot be empty")

    if not eps_grid:
        raise ValueError("eps_grid cannot be empty")

    true_state = tf.cast(true_state, dtype)

    if len(true_state.shape) != 2:
        raise ValueError("true_state must have shape [T, d]")

    Np_t = tf.cast(Np, dtype)
    max_iter = max(niter_grid)

    best_score = float("inf")
    best_params = None
    best_results = None
    results_table = []

    for n_iter in niter_grid:
        speed_penalty = float(n_iter / max_iter)

        for eps in eps_grid:
            scores = []
            rmses = []
            esses = []

            for rep in range(n_repeats):
                tf.random.set_seed(seed + 1000 * n_iter + 10 * rep)

                eps_t = tf.cast(eps, dtype)

                quality, rmse, mean_ess, ess_ratio, ests, ESS_list = _eval_ot_candidate(
                    filter_fn=filter_fn,
                    true_state=true_state,
                    Np_t=Np_t,
                    epsilon=eps_t,
                    sinkhorn_iters=n_iter,
                    lambda_ess=tf.cast(lambda_ess, dtype),
                    robust_cost=robust_cost,
                    lambda_robust=tf.cast(lambda_robust, dtype),
                    robust_mode=robust_mode,
                )

                final_score = quality + tf.cast(lambda_speed * speed_penalty, dtype)

                score_value = float(final_score.numpy())
                rmse_value = float(rmse.numpy())
                ess_value = float(mean_ess.numpy())

                scores.append(score_value)
                rmses.append(rmse_value)
                esses.append(ess_value)

                if score_value < best_score:
                    best_score = score_value
                    best_params = {
                        "epsilon": float(eps),
                        "sinkhorn_iters": int(n_iter),
                        "lambda_ess": float(lambda_ess),
                        "lambda_speed": float(lambda_speed),
                        "robust_cost": bool(robust_cost),
                        "lambda_robust": float(lambda_robust),
                        "robust_mode": robust_mode,
                        "rmse": rmse_value,
                        "mean_ess": ess_value,
                        "score": score_value,
                    }
                    best_results = (tf.identity(ests), tf.identity(ESS_list))

            results_table.append({
                "epsilon": float(eps),
                "sinkhorn_iters": int(n_iter),
                "robust_cost": bool(robust_cost),
                "lambda_robust": float(lambda_robust),
                "robust_mode": robust_mode,
                "score_mean": sum(scores) / len(scores),
                "rmse_mean": sum(rmses) / len(rmses),
                "ess_mean": sum(esses) / len(esses),
            })

    return best_params, best_results, results_table




########################################################
##  Use the Kalman filter for benchmarking purposes
########################################################

def kalman_loglik_alpha(Y, rho, Sigma, sigma_z, dtype=tf.float32):
    """
    Exact log-likelihood via the Kalman Filter.

    Y is expected in PF layout: (T, d)
    _filter_core expects (d, T), so we transpose internally.
    """
    Y = tf.cast(Y, dtype)
    Sigma = tf.cast(Sigma, dtype)
    sigma_z = tf.cast(sigma_z, dtype)

    d = tf.shape(Y)[1]

    alpha = tf.cast(0.01 + 0.98 * tf.sigmoid(rho), dtype)
    I = tf.eye(d, dtype=dtype)

    F_mat = alpha * I
    H_mat = I
    Q = Sigma
    R_mat = (sigma_z ** 2) * I

    predict_fn, update_fn = make_kf_kernels(F_mat, H_mat, Q)

    out = _filter_core(
        Y=tf.transpose(Y),
        predict_fn=predict_fn,
        update_fn=update_fn,
        R_mat=R_mat,
        m0=tf.zeros([d], dtype=dtype),
        P0=tf.zeros([d, d], dtype=dtype),
        measurement_type="gaussian",
        dtype=dtype,
    )

    return out["loglik"]


# Transition parameterization
def prop_fn_alpha(x_prev, Sigma_chol, rho, dtype=tf.float32):
    x_prev = tf.cast(x_prev, dtype)
    Sigma_chol = tf.cast(Sigma_chol, dtype)

    alpha = tf.cast(0.01 + 0.98 * tf.sigmoid(rho), dtype)

    eps = tf.random.normal(tf.shape(x_prev), dtype=dtype)
    v = tf.linalg.matvec(Sigma_chol, eps)

    return alpha * x_prev + v


# Observation log-likelihood function
def log_likelihood_gaussian(particles, y_t, sigma_z, dtype=tf.float32):
    particles = tf.cast(particles, dtype)
    y_t = tf.cast(y_t, dtype)
    sigma_z = tf.cast(sigma_z, dtype)

    d = tf.cast(tf.shape(particles)[1], dtype)
    log_2pi = tf.math.log(tf.constant(2.0 * np.pi, dtype=dtype))

    diff = y_t[None, :] - particles
    sq = tf.reduce_sum(tf.square(diff), axis=1)

    return -0.5 * (
        d * (log_2pi + 2.0 * tf.math.log(sigma_z))
        + sq / (sigma_z ** 2)
    )

def make_prop_fn(Sigma_chol, rho, dtype=tf.float32):
    return lambda x: prop_fn_alpha(x, Sigma_chol, rho, dtype)

def make_llk_fn(sigma_z, dtype=tf.float32):
    return lambda particles, y: log_likelihood_gaussian(particles, y, sigma_z, dtype)

