import numpy as np
import tensorflow as tf


def compute_Sigma_tf(d, alpha0=0.3, alpha1=0.01, beta=20.0, dtype=tf.float32):
    """
    Compute spatially correlated covariance matrix Sigma
    based on the paper "Particle Filtering with Invertible Particle Flow"
    
    Input:
    d: state dimension
    alpha0, alpha1, beta: parameters (int). Values set as in "Particle Filtering with Invertible Particle Flow"
    
    Output: Sigma [d,d] tf.Tensor
    """
    # Check Input 
    try:
        d = int(d)
    except Exception:
        raise ValueError(f"d must be an integer, got {type(d)}")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
        
    # Create a 2D positions grid, given that d = 64 
    side = int(np.sqrt(d))
    rows = tf.range(d, dtype=dtype) // side 
    cols = tf.range(d, dtype=dtype) % side 
    positions = tf.stack([rows, cols], axis=1)  # [d,2]
    
    # Compute pairwise squared distances
    dist2=pairwise_distance(positions, squared=True)
    # exponential decay
    Sigma = alpha0 * tf.exp(-dist2 / beta)
    # diagonal matrix
    Sigma += tf.eye(d) * alpha1

    return Sigma


def pairwise_distance(X, Y=None, squared=True):
    
    if Y is None:
        Y = X

    XX = tf.reduce_sum(X * X, axis=1, keepdims=True)
    YY = tf.reduce_sum(Y * Y, axis=1, keepdims=True)

    D = XX - 2 * tf.matmul(X, Y, transpose_b=True) + tf.transpose(YY)
    # numerical stability
    D = tf.maximum(D, 0.0)

    if not squared:
        D = tf.sqrt(D + 1e-12)

    return D



### EXAMPLE B of the paper by Li

def Sim_HD_LGSSM(d, T, alpha, sigma_z, Sigma_tf, dtype=tf.float32):
    """
    Simulate a high-dimensional linear Gaussian state space model:
        x_k = alpha x_{k-1} + v_k,   v_k ~ N(0, Sigma)
        z_k = x_k + w_k,             w_k ~ N(0, sigma_z^2 I)
    Based on the paper "Particle Filtering with Invertible Particle Flow"
    
    Input:
        d: state dimension (int >= 1)
        T: number of time steps (int >= 1)
        alpha: scalar state transition coefficient
        sigma_z: measurement noise std (scalar > 0)
        Sigma_tf: [d,d] process noise covariance (tf.Tensor)

    Output:
        true_states: tf.Tensor shape (T, d)
        measurements: tf.Tensor shape (T, d)
    """
    ### Check Input
    try:
        d = int(d)
        T = int(T)
    except Exception:
        raise ValueError("d and T must be integers")
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")

    alpha = float(alpha)
    sigma_z = float(sigma_z)
    
    if not tf.math.is_finite(alpha):
        raise ValueError("alpha must be finite")
    if sigma_z <= 0 or not tf.math.is_finite(sigma_z):
        raise ValueError("sigma_z must be positive and finite")

    Sigma_tf = tf.convert_to_tensor(Sigma_tf, dtype=dtype)
    if Sigma_tf.shape != (d, d):
        raise ValueError(f"Sigma_tf shape {Sigma_tf.shape}, expected {(d, d)}")
    tf.debugging.assert_all_finite(Sigma_tf, "Sigma contains NaN/Inf")
    
    # Check positive definiteness
    eigs = tf.linalg.eigvalsh(Sigma_tf)
    if tf.reduce_any(eigs <= 0):
        raise ValueError("Sigma is not positive definite")

    # Allocation space and Choleski decomposition on the spatially correlated covariace matrix
    L = tf.linalg.cholesky(Sigma_tf)
    x_prev = tf.zeros(d, dtype=dtype)
    true_states = []
    measurements = []
    
    # Recursion
    for _ in range(T):
        # Transition noise
        eps_v = tf.random.normal((d,), dtype=dtype)
        v_k = tf.matmul(tf.reshape(eps_v, (1, d)), tf.transpose(L))
        v_k = tf.reshape(v_k, (-1,))
        
        # Measurement noise
        w_k = tf.random.normal((d,), stddev=sigma_z, dtype=dtype)
        
        # update
        x_k = alpha * x_prev + v_k
        z_k = x_k + w_k
        x_prev = x_k
        
        true_states.append(x_k)
        measurements.append(z_k)

    true_states = tf.stack(true_states)
    measurements = tf.stack(measurements)

    return true_states, measurements

def F_jac(x, t, alpha):
    """
    Linear transition Jacobian for KF/EKF/UKF.
    Automatically uses dtype of x.
    """
    dtype = x.dtype
    d = tf.shape(x)[0]
    return tf.eye(d, dtype=dtype) * tf.cast(alpha, dtype)


def H_jac(x, t):
    """
    Linear measurement Jacobian for EKF.
    """
    dtype = x.dtype
    d = tf.shape(x)[0]
    return tf.eye(d, dtype=dtype)

def prop_linear_gaussian(x, F, L):
    dtype = x.dtype

    eps = tf.random.normal(tf.shape(x), dtype=dtype)
    F = tf.cast(F, dtype)
    L = tf.cast(L, dtype)

    return tf.linalg.matvec(F, x) + tf.linalg.matvec(L, eps)
    

def loglik_gaussian(particles, y, sigma_z):
    
    dtype = particles.dtype
    y = tf.cast(y, dtype)
    sigma_z = tf.cast(sigma_z, dtype)

    dz = y[None, :] - particles

    #log-likelihood
    llk = -0.5 * tf.reduce_sum(dz**2, axis=1) / (sigma_z**2)
    
    # finiteness check 
    tf.debugging.assert_all_finite(llk, "log-likelihood contains NaN or Inf")

    return llk


### FUNCTIONS FOR UPF - GAUSSIAN EXAMPLE
def gaussian_logpdf_batch(x, mean, cov): 

    dtype=x.dtype
    d = tf.cast(tf.shape(x)[-1], dtype)

    diff = x - mean
    L = tf.linalg.cholesky(cov)

    v = tf.linalg.triangular_solve(L, diff[..., None])
    quad = tf.reduce_sum(v**2, axis=[-2, -1])

    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=1)

    return -0.5 * (d*tf.math.log(2.0*tf.constant(np.pi, dtype)) + logdet + quad) 


def gh_dynamics_mean(x, alpha): 
    return alpha * x 


def gaussian_logpdf_batch_from_chol(x, mean, L): 
    dtype = x.dtype
    d = tf.cast(tf.shape(x)[-1], dtype)

    diff = x - mean                         # (Np, d)

    v = tf.linalg.triangular_solve(L, tf.transpose(diff))                  # (d, Np)
    quad = tf.reduce_sum(v**2, axis=0)      # (Np,)
    

    logdet = tf.cast(2.0, dtype) * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
    log2pi = tf.math.log(tf.constant(2.0 * np.pi, dtype=dtype))

    return tf.cast(-0.5, dtype)*(d*log2pi + logdet + quad)


def make_transition_logpdf_gaussian(alpha, gamma):

    @tf.function(reduce_retracing=True)
    def transition_logpdf_fn(x_new, x_prev, L_Sigma):
        mean = alpha * x_prev + gamma
        return gaussian_logpdf_batch_from_chol(x_new, mean, L_Sigma)

    return transition_logpdf_fn

def make_gaussian_model_wrappers(
    alpha,
    F_tf,
    L_tf,
    sigma_z,
):
    @tf.function
    def prop_fn_b(x):
        return prop_linear_gaussian(x, F_tf, L_tf)

    @tf.function
    def h_func_b(x):
        return tf.squeeze(H_fct(tf.expand_dims(x, 0)))

    @tf.function
    def llk_fn_b(particles, y):
        return loglik_gaussian(particles, y, sigma_z)

    @tf.function
    def H_jac_tf(x, t):
        return H_jac(x, t)

    @tf.function
    def F_jac_tf(x, t):
        return F_jac(x, t, alpha)

    return {
        "prop_fn": prop_fn_b,
        "h_func": h_func_b,
        "llk_fn": llk_fn_b,
        "H_jac_fn": H_jac_tf,
        "F_jac_fn": F_jac_tf,
    }



# EXAMPLE C taken from the paper of Li
def sample_skewed_t_v1(x_prev, alpha, Sigma, gamma, nu, dtype=tf.float32):
    """
    Sample a high-dimensional skewed-t vector given previous state.
    Input: 
    x_prev: 
    Sigma: process covariance
    gamma: skew vector
    """
    # cast everything to float64
    x_prev = tf.cast(x_prev, dtype)
    Sigma = tf.cast(Sigma, dtype)
    gamma = tf.cast(gamma, dtype)
    alpha = tf.cast(alpha, dtype)
    nu = tf.cast(nu, dtype)
    
    d = x_prev.shape[0]
    
    # w ~ Inverse-Gamma per t-scaling
    w = tf.random.gamma(shape=[], alpha=nu/2.0, beta=nu/2.0, dtype=dtype)
    
    # z ~ N(0, I)
    z = tf.random.normal(shape=[d,1], dtype=dtype)
    
    # Cholesky
    L = tf.linalg.cholesky(Sigma)
    
    # skewness
    gamma = tf.reshape(gamma, (d,1))
    
    # Skewed-t transform
    mu_k = alpha * x_prev
    x_k = mu_k[:, None] + (1/tf.sqrt(w)) * (L @ z + gamma)
    x_k = tf.reshape(x_k, (-1,))
    
    return x_k


def generate_skt_poi_data(T, d, alpha, Sigma_proc, gamma, nu, m1=1.0, m2=1.0/3.0, dtype=tf.float32, seed=None): 
    """
    Generate synthetic data for Example C:
    - Latent dynamics: stationary AR(1) GH skewed-t process
    - Measurements: Poisson counts
    
    Parameters
    ----------
    T : int
        Number of time steps (T > 0).

    d : int
        State dimension (d > 0).

    alpha : float or tf.Tensor (scalar)
        AR(1) coefficient (typically |alpha| < 1).

    Sigma_proc : tf.Tensor of shape (d, d)
        Process covariance (symmetric positive definite, finite).

    gamma : float or tf.Tensor of shape (d,)
        Skewness parameter (scalar is broadcast to (d,)).

    nu : float
        Degrees of freedom (nu > 0).

    m1 : float, optional
        Positive scaling of Poisson rate.

    m2 : float, optional
        Scaling inside exponential link: λ = m1 * exp(m2 * x).

    dtype : tf.DType, optional
        Computation dtype.

    seed : int or None, optional
        Random seed.

    Returns
    -------
    true_states : tf.Tensor (T, d)
        Latent states.

    measurements : tf.Tensor (T, d)
        Poisson observations.
    """
    # ---- Input checks ----
    if not isinstance(T, int) or T <= 0:
        raise ValueError("T must be a positive integer")

    if not isinstance(d, int) or d <= 0:
        raise ValueError("d must be a positive integer")

    if nu <= 0:
        raise ValueError("nu must be positive")

    if m1 <= 0:
        raise ValueError("m1 must be positive (Poisson rate scaling)")
        
    if len(Sigma_proc.shape) != 2:
        raise ValueError("Sigma_proc must be a 2D matrix")
    if Sigma_proc.shape[0] != d or Sigma_proc.shape[1] != d:
        raise ValueError(f"Sigma_proc must have shape ({d}, {d})")
    if tf.reduce_any(tf.math.is_nan(Sigma_proc)):
        raise ValueError("Sigma_proc contains NaNs")
    # PD check
    try:
        tf.linalg.cholesky(Sigma_proc)
    except tf.errors.InvalidArgumentError:
        raise ValueError("Sigma_proc must be positive definite")

    if seed is not None:
        tf.random.set_seed(seed)
    
    Sigma_proc=tf.cast(Sigma_proc, dtype)
    gamma = tf.ones((d,), dtype=dtype) * gamma 
    gamma=tf.cast(gamma, dtype)
    
    true_states = []
    measurements = []
    
    # initial state
    x_prev = tf.zeros((d,), dtype=dtype)
    
    for t in range(T):
        # skewed-t dynamics
        x_t = sample_skewed_t_v1(x_prev, alpha, Sigma_proc, gamma, nu) 
        true_states.append(x_t)
        
        # Poisson measurement - based on Eq.45 of the paper
        lam = m1*tf.exp(m2*x_t)
        z_t = tf.random.poisson(lam=lam, shape=()) 
        measurements.append(tf.cast(z_t, dtype))
        
        x_prev = x_t
    
    return tf.stack(true_states), tf.stack(measurements)



def H_jac_t(x, t, m1=1.0, m2=1.0/3.0):
    """
    Function to compute the Jacobian of measurement mean, that is
        E[z|x] = h(x) = m1 * exp(m2 * x)

    Returns:
        H(x) = diag(m1 * m2 * exp(m2 * x))
    """
    dtype = x.dtype
    m1 = tf.cast(m1, dtype)
    m2 = tf.cast(m2, dtype)

    return tf.linalg.diag(m1 * m2 * tf.exp(m2 * x))

def F_jac_t(x, t):
    """
    Function to compute the Jacobian of the state transition equation
    Input:
        x : tf.Tensor of shape (d,)
        t : int or tf.Tensor (unused)

    Output:
        tf.Tensor of shape (d, d)

    Notes:
        Local linearization of dynamics F(x) = alpha * x,
        so the Jacobian is alpha * I.
    """
    dtype=x.dtype
    n = tf.shape(x)[0]
    
    return tf.eye(n, dtype=dtype) * alpha


def H_fct_esrf_t(particles, m1=1.0, m2=1.0/3.0): 
    """
    Compute the measurement mean for Poisson observations.
    
    Input:
        particles : tf.Tensor of shape (Np, d)

    Output:
        tf.Tensor of shape (Np, d)

    Notes:
        Applies h(x) = m1 * exp(m2 * x) elementwise.
        Used to map particles to Poisson intensities λ.
    """
    dtype=particles.dtype
    particles = tf.cast(particles, dtype)
    m1 = tf.cast(m1, dtype)
    m2 = tf.cast(m2, dtype)

    return m1 * tf.exp(m2 * particles)


def F_fct_esrf_t(particles):
    """
    State propagation function F(x).

    Input:
        particles : tf.Tensor of shape (Np, d)

    Output:
        tf.Tensor of shape (Np, d)
    """
    dtype=particles.dtype
    
    return tf.cast(alpha, dtype) * particles 


def prop_gh_skewed_t(x, alpha, Sigma, gamma, nu): 
    """
    Single-particle propagation under skewed-t dynamics.

    Input:
        x     : tf.Tensor of shape (d,)
        alpha : scalar (float or tf.Tensor)
        Sigma : tf.Tensor of shape (d, d)
        gamma : scalar or tf.Tensor of shape (d,)
        nu    : positive scalar

    Output:
        tf.Tensor of shape (d,)
    """

    dtype = x.dtype
    alpha = tf.cast(alpha, dtype)
    Sigma = tf.cast(Sigma, dtype)
    nu = tf.cast(nu, dtype)
    gamma = tf.convert_to_tensor(gamma, dtype=dtype)

    # Broadcast gamma if scalar
    gamma = tf.cond(
        tf.equal(tf.rank(gamma), 0),
        lambda: tf.fill(tf.shape(x), gamma),
        lambda: gamma
    )

    return sample_skewed_t_v1(x_prev=x, alpha=alpha, Sigma=Sigma, gamma=gamma, nu=nu)



def log_likelihood_poisson(particles, y_t, m1=1.0, m2=1/3):
    
    dtype = particles.dtype

    y_t = tf.cast(y_t, dtype)
    m1 = tf.cast(m1, dtype)
    m2 = tf.cast(m2, dtype)

    # lambda = m1 * exp(m2 x)
    lam = m1 * tf.exp(m2 * particles) # [Np, d]

    # log p(y|x)
    log_lik = y_t[None, :] * tf.math.log(lam) - lam # lam >0
    tf.debugging.assert_all_finite(log_lik, "log-likelihood contains NaN or Inf")

    return tf.reduce_sum(log_lik, axis=1)  # [Np]


# GSMC propagation function.
@tf.function(reduce_retracing=True)
def prop_fn_vec(particles):
    return tf.vectorized_map(
        lambda x: prop_gh_skewed_t(x=x, alpha=alpha, Sigma=Sigma_tf_c, gamma=gamma, nu=nu),
        particles)


# UPF
# Approximate GH-skewed-t transition density
def student_t_logpdf(x, mean, Sigma, nu): 
    dtype=x.dtype
    d = tf.cast(tf.shape(x)[-1], dtype)

    diff = x - mean
    L = tf.linalg.cholesky(Sigma)
    v = tf.linalg.triangular_solve(L, diff[..., None])
    quad = tf.reduce_sum(v**2, axis=[-2, -1])

    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

    return (
        tf.math.lgamma((nu + d)/2)
        - tf.math.lgamma(nu/2)
        - 0.5*logdet
        - (d/2)*tf.math.log(nu * tf.constant(np.pi, dtype)) 
        - ((nu + d)/2) * tf.math.log(1 + quad/nu)
    )

def skewt_transition_logpdf(x, x_prev, alpha, Sigma, gamma, nu): 
    """
    GH skewed-t transition log-density for variance mixture form.
    """
    dtype = x.dtype
    x = tf.cast(tf.reshape(x, [-1]), dtype)          # [d]
    x_prev = tf.cast(tf.reshape(x_prev, [-1]), dtype)  # [d]
    Sigma = tf.cast(Sigma, dtype)
    gamma = tf.cast(gamma, dtype)
    nu = tf.cast(nu, dtype)

    d = tf.shape(x)[0]

    # mean
    mu = alpha * x_prev

    # delta
    delta = x - mu - gamma
    delta = tf.reshape(delta, [d, 1])

    # Cholesky
    L = tf.linalg.cholesky(Sigma)

    # Solve Sigma^{-1} delta
    v = tf.linalg.triangular_solve(L, delta)
    quad = tf.reduce_sum(v**2)

    # log-determinant
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

    # Student-t type density (variance mixture)
    logZ = tf.math.lgamma((nu + tf.cast(d, dtype)) / 2) - tf.math.lgamma(nu / 2)
    logZ -= 0.5 * tf.cast(d, dtype) * tf.math.log(nu * tf.constant(np.pi, dtype))
    logZ -= 0.5 * logdet

    log_density = logZ - 0.5*(nu + tf.cast(d, dtype)) * tf.math.log(1 + quad/nu)
    return log_density


def gh_transition_logpdf(x_new, x_prev, alpha, Sigma, nu):
    mean = alpha * x_prev
    return student_t_logpdf(x_new, mean, Sigma, nu)


def student_t_logpdf_batch(x, mean, L_Sigma, nu): 
    """
    Batch student-t logpdf using precomputed Cholesky.
    x, mean: (Np, d)
    L_Sigma: (d, d)
    """
    dtype = x.dtype
    d = tf.cast(tf.shape(x)[-1], dtype)

    diff = x - mean                      # (Np,d)
    solved = tf.linalg.triangular_solve(
        L_Sigma,
        tf.transpose(diff)
    )                                     # (d,Np)

    quad = tf.reduce_sum(solved**2, axis=0)

    logdet = 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(L_Sigma))
    )

    logZ = (tf.math.lgamma((nu + d)/2) - tf.math.lgamma(nu/2) - 0.5*d*tf.math.log(nu*np.pi) - 0.5*logdet)

    return logZ - 0.5*(nu + d)*tf.math.log(1 + quad/nu)


@tf.function(reduce_retracing=True)
def skewt_transition_logpdf_batch_from_chol(x, x_prev, L_Sigma, alpha, gamma, nu):
    dtype = x.dtype

    # Casting 
    x = tf.cast(x, dtype)
    x_prev = tf.cast(x_prev, dtype)
    L_Sigma = tf.cast(L_Sigma, dtype)
    gamma = tf.cast(gamma, dtype)
    nu = tf.cast(nu, dtype)
    tf.debugging.assert_positive(nu, message="nu must be positive")
    d = tf.cast(tf.shape(x)[-1], dtype)

    mean = alpha * x_prev + gamma       # (Np,d)
    diff = x - mean                     # (Np,d)

    v = tf.linalg.triangular_solve(L_Sigma, tf.transpose(diff))  # (d,Np)
    
    quad = tf.reduce_sum(v**2, axis=0)

    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Sigma)))
    
    log2pi = tf.math.log(tf.constant(2.0 * np.pi, dtype=dtype)) 
    logZ = (tf.math.lgamma((nu + d)/2) - tf.math.lgamma(nu/2) - 0.5*d*log2pi  - 0.5*logdet) 

    return logZ - 0.5*(nu + d)*tf.math.log(1 + quad/nu)


def make_transition_logpdf_skewt(alpha, gamma, nu):

    @tf.function(reduce_retracing=True)
    def transition_logpdf_fn(x_new, x_prev, L_Sigma):
        return skewt_transition_logpdf_batch_from_chol(
            x_new, x_prev, L_Sigma, alpha, gamma, nu
        )

    return transition_logpdf_fn


def make_poisson_model_wrappers(
    alpha,
    Sigma_tf,
    gamma_vec,
    nu,
    m1=1.0,
    m2=1.0 / 3.0,
):
    @tf.function
    def prop_fn_fixed(x):
        return sample_skewed_t_v1(
            x,
            alpha=alpha,
            Sigma=Sigma_tf,
            gamma=gamma_vec,
            nu=nu
        )

    @tf.function
    def h_func_fixed(x):
        return m1 * tf.exp(m2 * x)

    @tf.function
    def llk_fn_poisson(particles, y):
        return log_likelihood_poisson(
            particles,
            y,
            m1=m1,
            m2=m2
        )

    @tf.function
    def H_jac_t_tf(x, t):
        return H_jac_t(
            x,
            t,
            m1=m1,
            m2=m2
        )

    return {
        "prop_fn": prop_fn_fixed,
        "h_func": h_func_fixed,
        "llk_fn": llk_fn_poisson,
        "H_jac_fn": H_jac_t_tf,
    }
