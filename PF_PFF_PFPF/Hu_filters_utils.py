import tensorflow as tf
import math
import numpy as np

from replicate_Li_filters import compute_spectral_norm

def loglik_poisson_grad(particles, y, m1=1.0, m2=1/3):
    """
    Gradient of the Poisson log-likelihood w.r.t. particle states.
    
    particles: [Np, d]
    y: [d]
    returns: [Np, d]
    """
    dtype=particles.dtype
    y = tf.cast(y, dtype) 

    lam = m1 * tf.exp(m2 * particles)     # [Np, d]
    grad = m2 * (y[None, :] - lam)        # [Np, d]

    return grad


def hu_kernel_scalar(X, alpha, dtype=tf.float32):
    """
     Scalar Gaussian (Hu-style) kernel and its aggregated repulsion gradient.

    Input:
        X : tf.Tensor (Np, d) particle positions
        alpha : positive scalar bandwidth

    Output:
        K_scalar : (Np, Np) kernel matrix
        gradK    : (Np, d) aggregated repulsion term
    """
    X = tf.cast(X, dtype=dtype) 
    diff = X[:, None, :] - X[None, :, :]       # pairwise differences [Np, Np, d]
    dist2 = tf.reduce_sum(diff**2, axis=-1)   # squared distances [Np, Np]
    
    K_scalar = tf.exp(-dist2 / alpha)   # kernel [Np, Np]

    # repulsion: sum_j ∇_x_i k(x_i, x_j)
    gradK = -2.0 / alpha * tf.reduce_sum(diff * K_scalar[:, :, None], axis=1) # [Np, d]

    return K_scalar, gradK, None


def make_hu_matrix_kernel(Qinv, dtype=tf.float32):
    """
    Build a matrix-valued Hu kernel using an inverse covariance matrix.

    Input:
        Qinv : tf.Tensor (d, d) inverse metric

    Output:
        kernel_fn(X, alpha):
            K          : (Np, Np) kernel matrix
            None       : placeholder (no aggregated gradient)
            gradK_full : (Np, Np, d) pairwise gradients ∇_{x_i} k(x_i, x_j)

    Notes:
        Returns a Mahalanobis kernel using Qinv.
    """
    Qinv = tf.cast(Qinv, dtype)

    def kernel_fn(X, alpha):
        X = tf.cast(X, dtype)
        diff = X[:, None, :] - X[None, :, :]
        
        #Mahalanobis distance 
        temp = tf.matmul(diff, Qinv)
        dist2 = tf.reduce_sum(temp * diff, axis=-1)
        
        K = tf.exp(-dist2 / alpha)
        
        # pairwise gradients ∇_{x_i} k(x_i, x_j)
        gradK_full = -2.0 / alpha * temp * K[:, :, None]

        return K, None, gradK_full

    return kernel_fn


def make_flow_scalar(kernel_fn):
    """
    Factory for Hu flow with scalar kernel
    """
    def flow_fn(particles, grad_log_post, alpha):
        K, gradK, _ = kernel_fn(particles, alpha)
        attract = tf.matmul(K, grad_log_post)
        flow = (attract + gradK) / tf.cast(tf.shape(particles)[0], particles.dtype)
        return flow
    return flow_fn


def make_flow_matrix(kernel_fn, M):
    """
    Factory for Hu flow with matrix kernel:
    combines attraction and repulsion, then applies linear transform M
    """
    M = tf.cast(M, tf.float32)

    def flow_fn(particles, grad_log_post, alpha):
        K, _, gradK_full = kernel_fn(particles, alpha)
        
        attract = tf.matmul(K, grad_log_post)
        repel = tf.reduce_sum(gradK_full, axis=1)
        
        flow = (attract + repel) / tf.cast(tf.shape(particles)[0], particles.dtype)
        flow = tf.matmul(flow, tf.transpose(tf.cast(M, particles.dtype)))
        return flow

    return flow_fn


def median_alpha(X):
    X = tf.cast(X, tf.float32)

    diff = X[:, None, :] - X[None, :, :]
    dist2 = tf.reduce_sum(diff**2, axis=-1)

    mask = tf.logical_not(tf.eye(tf.shape(dist2)[0], dtype=tf.bool))
    dist_flat = tf.boolean_mask(dist2, mask)

    dist_sorted = tf.sort(dist_flat)
    N = tf.size(dist_sorted)
    mid = N // 2

    median = tf.cond(N % 2 == 0,lambda: 0.5 * (dist_sorted[mid - 1] + dist_sorted[mid]),lambda: dist_sorted[mid])
    
    return median



@tf.function(reduce_retracing=True)
def pff_filter_hu_new(
    measurements,
    Np,
    prop_fn,
    loglik_grad_fn,
    flow_step_fn,
    alpha_bandwidth_fn=median_alpha,
    spectral_norm_fn=compute_spectral_norm,
    n_steps=10,
    eps=0.01
):
    """
    Particle Flow Filter (Hu).

    Input:
        measurements: (T,d) data
        Np: number of particles
        prop_fn: dynamics
        loglik_grad_fn: ∇ log p(y|x)
        flow_step_fn: Hu flow update
        alpha_bandwidth_fn: kernel bandwidth
        n_steps, eps: flow iterations

    Returns:
        ests        : (T, d) filtered estimates
        particles   : (T, Np, d) particle estimates
        diagnostics : flow norms, conditioning, spectral info

    Notes:
        Propagates particles and applies Hu flow steps to approximate the posterior.
        Supports scalar or matrix-valued kernels via flow_step_fn.
    """
    
    dtype = measurements.dtype
    T = tf.shape(measurements)[0]
    d = tf.shape(measurements)[1]

    # --- checks ---
    tf.debugging.assert_greater(T, 0)
    tf.debugging.assert_greater(d, 0)

    if not isinstance(Np, int) or Np <= 0:
        raise ValueError("Np must be positive")

    if not callable(prop_fn):
        raise TypeError("prop_fn must be callable")

    if not callable(loglik_grad_fn):
        raise TypeError("loglik_grad_fn must be callable")

    # --- Particles ---
    particles = tf.random.normal((Np, d), dtype=dtype)

    # --- Allocation ---
    ests = tf.TensorArray(dtype, size=T)
    flow_norms = tf.TensorArray(dtype, size=T)
    grad_conds = tf.TensorArray(dtype, size=T)
    spec_J = tf.TensorArray(dtype, size=T)
    particles_all = tf.TensorArray(dtype, size=T)

    # --- Recursion ---
    for t in tf.range(T):
        y = measurements[t]

        particles = tf.vectorized_map(prop_fn, particles)
        # Bandwidth for kernel
        alpha = alpha_bandwidth_fn(particles)
        
        # Local storage for flow norms
        flow_norms_local = tf.TensorArray(dtype, size=n_steps)

        # Flow iterations 
        for k in tf.range(n_steps):
            grad_log_post = loglik_grad_fn(particles, y)
            
            # Hu flow step (attraction + repulsion)
            flow = flow_step_fn(particles, grad_log_post, alpha)
            flow = tf.where(tf.math.is_finite(flow), flow, tf.zeros_like(flow))

            particles += eps * flow

            flow_norms_local = flow_norms_local.write(k, tf.norm(flow, axis=1))

        flow_norm_all = flow_norms_local.stack()[-1]

        # --- Gradient conditioning ---
        grad_norm = tf.norm(grad_log_post, axis=1)
        cond = tf.reduce_max(grad_norm) / (tf.reduce_min(grad_norm) + 1e-8)

        # --- Compute local Hessian spectral norm (diagnostics) ---
        x_mean = tf.reduce_mean(particles, axis=0)

        with tf.GradientTape() as tape2:
            tape2.watch(x_mean)
            with tf.GradientTape() as tape1:
                tape1.watch(x_mean)
                log_post = tf.reduce_sum(
                    loglik_grad_fn(x_mean[None, :], y)
                )
            grad = tape1.gradient(log_post, x_mean)

        H_mean = tape2.jacobian(grad, x_mean)

        spec_H = tf.cond(tf.reduce_all(tf.math.is_finite(H_mean)),
            lambda: spectral_norm_fn(H_mean),
            lambda: tf.constant(0.0, dtype=dtype))

        # --- record ---
        ests = ests.write(t, tf.reduce_mean(particles, axis=0))
        flow_norms = flow_norms.write(t, flow_norm_all)
        grad_conds = grad_conds.write(t, cond)
        spec_J = spec_J.write(t, spec_H)
        particles_all = particles_all.write(t, particles)

    diagnostics_dict = {
        "flow_norm": flow_norms.stack(),
        "grad_cond": grad_conds.stack(),
        "spec_J": spec_J.stack()
    }

    return ests.stack(), particles_all.stack(), diagnostics_dict


def make_poisson_grad_wrapper(m1=1.0, m2=1.0 / 3.0):
    @tf.function
    def loglik_grad_fn(particles, y):
        return loglik_poisson_grad(
            particles,
            y,
            m1=m1,
            m2=m2
        )
    return loglik_grad_fn



def make_hu_pff_runners(
    Sigma_tf,
    prop_fn,
    loglik_grad_fn,
    n_steps=10,
    eps=0.5,
):
    scalar_flow_fn = make_flow_scalar(hu_kernel_scalar)

    Qinv = tf.linalg.inv(Sigma_tf)
    matrix_kernel_fn = make_hu_matrix_kernel(Qinv)
    matrix_flow_fn = make_flow_matrix(matrix_kernel_fn, Sigma_tf)

    def run_hu_pff_flow_matrix(Y, Np):
        return pff_filter_hu_new(
            measurements=Y,
            Np=Np,
            prop_fn=prop_fn,
            flow_step_fn=matrix_flow_fn,
            loglik_grad_fn=loglik_grad_fn,
            n_steps=n_steps,
            eps=eps,
        )

    def run_hu_pff_flow_scalar(Y, Np):
        return pff_filter_hu_new(
            measurements=Y,
            Np=Np,
            prop_fn=prop_fn,
            flow_step_fn=scalar_flow_fn,
            loglik_grad_fn=loglik_grad_fn,
            n_steps=n_steps,
            eps=eps,
        )

    return {
        "matrix": run_hu_pff_flow_matrix,
        "scalar": run_hu_pff_flow_scalar,
    }