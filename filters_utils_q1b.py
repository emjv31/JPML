import tensorflow as tf 
import math
import numpy as np
import time

import warnings
warnings.filterwarnings('ignore') 

# I) 
def SV_model_sim_tf_h(iT, phi, sigma_eta, sigma_eps, xi, seed = 123):
    """
    Simulate a univariate stationary stochastic volatility
          y_t = exp(h_t / 2) * eps_t, eps_t is N(0, sigma_eps)
          h_t = phi * h_{t-1} + eta_t, eta_t is N(0, sigma_eta)
    with eps_t and eta_t independent at all leads and lags. 
    
    The case where the innovations are correlated is straightforward to obtain using the closure property of the Gaussian distribution.
    The timing of the correlation coefficient can be either contemporaneous as in Harvey (1996), e.g. rho = E[eta_t*eps_t], or intertemporal
    as in Yu (2005), e.g. rho = E[eta_{t-1}*eps_t]
    
    Input:
    ------
        iT : int, Number of observations.
        phi : float, AR(1) coefficient for the log-volatility. Should be |phi| < 1.
        sigma_eta : float, Standard deviation of eta_t.
        sigma_eps : float, Standard deviation of eps_t.
        xi : float, Scale parameter for the observations.
        seed : int, Random seed for reproducibility.

    Output:
    ------
        dict:
            vY : tf.Tensor, Simulated observations. (iT,)
            h  : tf.Tensor, Latent log-volatility. (iT, )  
    """
    # Check input
    if seed is not None:
        tf.random.set_seed(seed)
    if not isinstance(iT, int) or iT <= 0:
        raise ValueError("iT must be a positive integer")
    if not isinstance(phi, (float, int)) or abs(phi) >= 1:
        raise ValueError("phi must be a scalar in (-1, 1)")
    if not isinstance(sigma_eta, (float, int)) or sigma_eta <= 0:
        raise ValueError("sigma_eta must be positive")
    if not isinstance(sigma_eps, (float, int)) or sigma_eps <= 0:
        raise ValueError("sigma_eps must be positive")
    if not isinstance(xi, (float, int)) or xi <= 0:
        raise ValueError("xi must be positive")

    # Convert input parameters for modelling    
    phi, sigma_eta, sigma_eps, xi = map(lambda x: tf.constant(x, tf.float64), [phi, sigma_eta, sigma_eps, xi])

    # Innovations
    eta = tf.random.normal([iT], mean=0.0, stddev=sigma_eta, dtype=tf.float64)
    eps = tf.random.normal([iT], mean=0.0, stddev=sigma_eps, dtype=tf.float64)

    # h initialised to the stationary distribution of the process, i.e. h0 is N(0, sigma_eta^2 / (1 - phi^2))
    h0_std = sigma_eta / tf.sqrt(1.0 - phi**2)
    h0 = tf.random.normal([], mean=0.0, stddev=h0_std, dtype=tf.float64)

    # Helper function for the time series recursion 
    def helper_transition(h_prev, eta_t):
        return phi * h_prev + eta_t

    h0 = tf.constant(0.0, tf.float64)
    h = tf.scan(helper_transition, eta[1:], initializer=h0)
    h = tf.concat([[h0], h], axis=0)
    # observations
    y = xi*eps * tf.exp(h / 2.0)
    
    return {"vY": y, "h": h}



#II) 
def compute_jacobian_tf(fun, x, tol=1e-6):
    """
    Compute numerical derivative of fun at x using central difference.

    Input:
    ------
    fun : callable, function to differentiate
    x : float or tf.Tensor, point where derivative is evaluated 
    tol : float, small perturbation for finite difference (optional, set to 1e-6)

    Output:
    ------
        J : float or tf.Tensor, numerical derivative of fun at x
    """
    # Check float64
    if isinstance(x, tf.Tensor):
        x = tf.cast(x, dtype=tf.float64)
    else:
        x = tf.convert_to_tensor(x, dtype=tf.float64)

    # Check finiteness
    if not tf.math.is_finite(x):
        raise ValueError("Jacobian: x not finite")
    f1 = tf.convert_to_tensor(fun(x + tol), dtype=tf.float64)
    f2 = tf.convert_to_tensor(fun(x - tol), dtype=tf.float64)
    
    # Check function outputs finite
    if not (tf.math.is_finite(f1) and tf.math.is_finite(f2)):
        raise ValueError("Jacobian: function returned non-finite values")

    # Compute Numerical derivative
    J = (f1 - f2) / (2.0 * tol)

    # Check derivative is finite
    if not tf.math.is_finite(J):
        raise ValueError("Jacobian: non-finite derivative")
        
    # Check derivative is not numerically zero
    if tf.abs(J) < 1e-8:
        raise ValueError(f"Jacobian numerically zero at x = {x.numpy()}")

    return J


def compute_sigma_points_tf(mu, P, alpha=0.5, beta=2.0, kappa=0.0):
    """
    Computes sigma points and corresponding weights for the Unscented Transform
    Operates on a single scalar state; for multivariate states, apply independently per dimension.

    Input:
    ------
    mu : float or tf.Tensor, mean of the state
    P : float or tf.Tensor, variance of the state 
    alpha : float, spread scaling parameter (optional, set to 0.5)
    beta : float, distribution prior info parameter (optional, set to 2.0)
    kappa : float, secondary scaling parameter, (optional, set to 0.0)

    Output:
    ------
    dict: 
        "X" : tf.Tensor, sigma points [mu, mu+spread, mu-spread]  (shape [3])
        "Wm" :  tf.Tensor, weights for mean  (shape [3])
        "Wc" tf.Tensor, weights for covariance (shape [3])
    """
    # Conversion 
    mu = tf.convert_to_tensor(mu, dtype=tf.float64)
    P = tf.convert_to_tensor(P, dtype=tf.float64)
    alpha = tf.convert_to_tensor(alpha, dtype=tf.float64)
    beta = tf.convert_to_tensor(beta, dtype=tf.float64)
    kappa = tf.convert_to_tensor(kappa, dtype=tf.float64)

    # Check input 
    if not tf.math.is_finite(mu):
        raise ValueError("Sigma points: mu not finite")
    if not tf.math.is_finite(P) or P <= 0:
        raise ValueError("Sigma points: P must be positive")

    n = 1
    lambda_ = alpha**2 * (n + kappa) - n
    denom = n + lambda_

    if denom <= 0 or not tf.math.is_finite(denom):
        raise ValueError("Invalid scaling: n + lambda <= 0")

    gamma = tf.sqrt(denom)
    spread = gamma * tf.sqrt(P)

    # Robustness checks
    if not tf.math.is_finite(spread) or spread <= 0:
        raise ValueError("Sigma points collapsed: spread <= 0")
    # 
    if spread < 1e-14 * (tf.abs(mu) + 1):
        raise ValueError("Sigma points numerically degenerate (spread too small)")

    # Compute sigma points: in the univariate case three points are needed 
    X = tf.stack([mu, mu + spread, mu - spread])

    # Check degeneracy 
    tol = 1e-12
    if tf.reduce_max(tf.abs(X - X[0])) < tol:
        raise ValueError("Sigma points degenerate: all points coincide")

    # Compute weights
    Wm = tf.stack([lambda_ / denom, 1.0/(2*denom), 1.0/(2*denom)])
    Wc = tf.identity(Wm)
    Wc = tf.tensor_scatter_nd_add(Wc, [[0]], [1 - alpha**2 + beta])

    # Check weights finite
    if not tf.reduce_all(tf.math.is_finite(Wm)) or not tf.reduce_all(tf.math.is_finite(Wc)):
        raise ValueError("Non-finite UKF weights")

    return {"X": X, "Wm": Wm, "Wc": Wc}
    


def extensionKF_uni_tf_consistent(y, f_fun, h_fun, sigma_eta, sigma_e, m0=0.0, P0=1.0, method="EKF"):
    """
    Function to perform filtering supporting both EKF and UKF algorithms 
    Performs recursive prediction and update for a sequence of scalar observations.
    Consistent with compute_jacobian_tf and compute_sigma_points_tf.

    Input:
    ------
    y: array-like, Observations, shape = (T, )
    f_fun: callable, State transition function f(x) for x_{t} = f(x_{t-1}) + transition_noise.
    h_fun: callable, Observation function h(x) for y_t = h(x_t) + measurement_noise.
    sigma_eta: float, Standard deviation of the transition noise.
    sigma_e: float, Standard deviation of the measurement noise.
    m0: float, Initial state mean (set to 0.0)
    P0: float, Initial state variance (set to 1.0)
    method: str, optional
        Filtering method: "EKF" (Extended Kalman Filter) or "UKF" (Unscented Kalman Filter).
    
    Output
    -------
    dict:
        "mu_filt": filtered means
        "P_filt": filtered variances
        "mu_pred": predicted means
        "P_pred": predicted variances
        "llk": approximated log-likelihood 
    """
    # ---------- Method check ----------
    method = method.upper()
    if method not in ("EKF", "UKF"):
        raise ValueError("method must be 'EKF' or 'UKF'")

    # ---------- Convert inputs ----------
    y = tf.convert_to_tensor(y, dtype=tf.float64)
    m0 = tf.convert_to_tensor(m0, dtype=tf.float64)
    P0 = tf.convert_to_tensor(P0, dtype=tf.float64)
    sigma_eta = tf.convert_to_tensor(sigma_eta, dtype=tf.float64)
    sigma_e = tf.convert_to_tensor(sigma_e, dtype=tf.float64)

    # Check input 
    if y.shape.ndims != 1 or tf.size(y) < 2: #or y.shape[0] is None: 
        raise ValueError("y must be a 1D vector of length >= 2") ## 2 <= length < T = inf 
    if not tf.reduce_all(tf.math.is_finite(y)):
        raise ValueError("the values of y must be finite")
    # Parameters: finiteness and 1D
    for val, name in [(m0, "m0"), (P0, "P0"), # (y, "y"), 
                      (sigma_eta, "sigma_eta"), (sigma_e, "sigma_e")]:
        if not tf.reduce_all(tf.math.is_finite(val)):
            raise ValueError(f"{name} must be finite")
        if val.shape.ndims != 0:
            raise ValueError(f"{name} must be a scalar (0-dimensional tensor)")

    if P0 <= 0 or sigma_eta <= 0 or sigma_e <= 0:
        raise ValueError("P0, sigma_eta, and sigma_e must be positive")

    if not callable(f_fun):
        raise ValueError("f_fun must be a function")
    if not callable(h_fun):
        raise ValueError("h_fun must be a function")

    n = tf.shape(y)[0]  # number of observations
    pi_tf = tf.constant(math.pi, dtype=tf.float64)

    # Allocation space 
    mu_pred = tf.Variable(tf.zeros(n, dtype=tf.float64))
    mu_filt = tf.Variable(tf.zeros(n, dtype=tf.float64))
    P_pred  = tf.Variable(tf.zeros(n, dtype=tf.float64))
    P_filt  = tf.Variable(tf.zeros(n, dtype=tf.float64))
    v       = tf.Variable(tf.zeros(n, dtype=tf.float64))
    llk     = tf.Variable(0.0, dtype=tf.float64)

    mu_filt[0].assign(m0)
    P_filt[0].assign(P0)
    mu_pred[0].assign(m0)
    P_pred[0].assign(P0)
    v[0].assign(tf.constant(0.0, dtype=tf.float64))

    # Recursion
    t_indices = tf.range(1, n, dtype=tf.int32)

    for t in t_indices:
        ### Prediction
        if method == "EKF":
            Jf = compute_jacobian_tf(f_fun, mu_filt[t-1])
            mu_pred_t = f_fun(mu_filt[t-1])
            P_pred_t = Jf**2 * P_filt[t-1] + sigma_eta**2
        else:  # UKF
            sp = compute_sigma_points_tf(mu_filt[t-1], P_filt[t-1])
            X, Wm, Wc = sp["X"], sp["Wm"], sp["Wc"]
            X_pred = tf.stack([f_fun(xi) for xi in X])
            if not tf.reduce_all(tf.math.is_finite(X_pred)):
                raise ValueError(f"Non-finite prediction at t={t}")
            mu_pred_t = tf.reduce_sum(Wm * X_pred)
            P_pred_t = sigma_eta**2 + tf.reduce_sum(Wc * (X_pred - mu_pred_t)**2)
            
        # Check P_pred
        if not tf.math.is_finite(P_pred_t) or P_pred_t <= 0:
            raise ValueError(f"Invalid P_pred at t={t}")

        mu_pred[t].assign(mu_pred_t)
        P_pred[t].assign(P_pred_t)

        # Update
        if method == "EKF":
            Jh = compute_jacobian_tf(h_fun, mu_pred_t)
            v_t = y[t] - h_fun(mu_pred_t)
            S_t = Jh**2 * P_pred_t + sigma_e**2
            if not tf.math.is_finite(S_t) or S_t <= 0:
                raise ValueError(f"Invalid S_t at t={t}")
            K = P_pred_t * Jh / S_t
            mu_t_filt = mu_pred_t + K * v_t
            P_t_filt = (1 - K * Jh) * P_pred_t
        else:  # UKF
            sp = compute_sigma_points_tf(mu_pred_t, P_pred_t)
            X, Wm, Wc = sp["X"], sp["Wm"], sp["Wc"]
            Y = tf.stack([h_fun(xi) for xi in X])
            if not tf.reduce_all(tf.math.is_finite(Y)):
                raise ValueError(f"Non-finite observation mapping at t={t}")
            mu_y = tf.reduce_sum(Wm * Y)
            if tf.reduce_all(tf.abs(Y - mu_y) < 1e-12):
                raise ValueError(f"Observation sigma points collapsed at t={t}")
            S_t = sigma_e**2 + tf.reduce_sum(Wc * (Y - mu_y)**2)
            if not tf.math.is_finite(S_t) or S_t <= 0:
                raise ValueError(f"Invalid S_t at t={t}")
            C_xy = tf.reduce_sum(Wc * (X - mu_pred_t) * (Y - mu_y))
            if tf.abs(C_xy) < 1e-14 * (tf.abs(mu_pred_t) + 1):
                raise ValueError(f"UKF degeneracy detected: C_xy ≈ 0 at t={t}")
            K = C_xy / S_t
            v_t = y[t] - mu_y
            mu_t_filt = mu_pred_t + K * v_t
            P_t_filt = P_pred_t - K**2 * S_t
        # State checks
        if not tf.math.is_finite(mu_t_filt) or not tf.math.is_finite(P_t_filt) or P_t_filt <= 0:
            raise ValueError(f"Invalid filtered state at t={t}")
        mu_filt[t].assign(mu_t_filt)
        P_filt[t].assign(P_t_filt)
        v[t].assign(v_t)
        # Lielihood via prediction error decomposition 
        llk_inc = -0.5 * (tf.math.log(2.0 * pi_tf * S_t) + v_t**2 / S_t)
        if not tf.math.is_finite(llk_inc):
            raise ValueError(f"Invalid log-likelihood increment at t={t}")
        llk.assign_add(llk_inc)

    return {
        "mu_filt": mu_filt,
        "P_filt": P_filt,
        "mu_pred": mu_pred,
        "P_pred": P_pred,
        "llk": llk
    }


# III) 
def SIR_bootstrap_markov_tf_stat(Y, N, phi, tau, sigma, squared=False):
    """
    SIR bootstrap particle filter. 
    Particles initialised to the stationary distribution of the process.
    Exploiting the Markovian property: it only keeps the latest particles
    
    Input
    ------
    Y: array-like, Observations. Shape (T, )
    N: int, Number of particles.
    phi: float, AR(1) coefficient for latent variable. Must be |phi|<1
    tau: float, Std of the transition noise.
    sigma: float, Std of observation noise.
    squared: bool
        If True, estimate second moment of the state variable.
        Otherwise, estimate the mean of the state variable
    
    Output
    -------
    dict
        'part_est' : array-like, estimated expectation (or second moment if squared = True) (T,)
        'ESS' : array-like, effective sample size (T,)
        'loglikelihood' : scalar, approximate log-likelihood 
    """
    # Check input
    if N <= 0:
        raise ValueError("N must be positive")
    if not (-1 < phi < 1):
        raise ValueError("phi must be in (-1,1)")
    if tau <= 0 or sigma <= 0:
        raise ValueError("tau and sigma must be positive")
    if Y is None or len(Y) < 2:
        raise ValueError("Y must have at least 2 observations")
    if not isinstance(squared, bool):
        raise ValueError("squared must be a boolean")

    # Convert to float64 ---
    Y = tf.convert_to_tensor(Y, dtype=tf.float64)
    # Check Y is univariate 
    if Y.shape.ndims != 1:
        raise ValueError("Y must be a univariate time series")
    phi = tf.convert_to_tensor(phi, dtype=tf.float64)
    tau = tf.convert_to_tensor(tau, dtype=tf.float64)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float64)

    T = tf.shape(Y)[0]

    # Allocation space
    part_est = tf.Variable(tf.zeros(T, dtype=tf.float64))
    ESS = tf.Variable(tf.zeros(T, dtype=tf.float64))
    loglik = tf.constant(0.0, dtype=tf.float64)
    
    # Initialisation  
    # Sample particles from stationary distribution
    h0_std = tau / tf.sqrt(1.0 - phi**2)
    h0_std = tf.cast(h0_std, dtype=tf.float64)  
    X = tf.random.normal((N,), mean=tf.constant(0.0, dtype=tf.float64),
                         stddev=h0_std, dtype=tf.float64)
    # weights
    log_w_raw = -0.5 * ((Y[0] - X)/sigma)**2 - tf.math.log(sigma) - 0.5*tf.math.log(tf.constant(2*math.pi, dtype=tf.float64))
    log_w = log_w_raw - tf.reduce_logsumexp(log_w_raw)
    w = tf.exp(log_w)

    ESS[0].assign(1.0 / tf.reduce_sum(w**2))
    loglik += tf.reduce_logsumexp(log_w_raw) - tf.math.log(tf.cast(N, tf.float64))
    part_est[0].assign(tf.reduce_mean(X if not squared else X**2))

    # Recursion
    for t in tf.range(1, T):
        # State
        X = phi * X + tf.random.normal((N,), dtype=tf.float64) * tau
        # Log-weights - robust computation using LogSumExp
        log_w_raw = -0.5 * ((Y[t] - X)/sigma)**2 - tf.math.log(sigma) - 0.5*tf.math.log(tf.constant(2*math.pi, dtype=tf.float64))
        log_w = log_w_raw - tf.reduce_logsumexp(log_w_raw)
        w = tf.exp(log_w)
        # ESS and log-likelihood
        ESS[t].assign(1.0 / tf.reduce_sum(w**2))
        loglik += tf.reduce_logsumexp(log_w_raw) - tf.math.log(tf.cast(N, tf.float64))
        # Systematic resampling
        cdf = tf.cumsum(w)
        u0 = tf.random.uniform((), dtype=tf.float64) / tf.cast(N, tf.float64)
        u = u0 + tf.range(N, dtype=tf.float64) / tf.cast(N, tf.float64)
        idx = tf.searchsorted(cdf, u)
        X = tf.gather(X, idx)

        # Particle estimate
        part_est[t].assign(tf.reduce_mean(X if not squared else X**2))

    return {
        "part_est": part_est,
        "ESS": ESS,
        "loglikelihood": loglik
    }


# IV) 
def compute_bias_rmse(x_true, x_est):
    """
    Compute Bias and RMSE between the true and estimated values of two arrays.
    Input:
    ------
        x_true: array-like of shape [T] or [T, d], any numeric type
        x_est: array-like of shape [T] or [T, d], any numeric type
        
    Output:
    ------
        bias: scalar tensor
        rmse: scalar tensor
    """
    # Convert to tensors if needed
    if not isinstance(x_est, tf.Tensor):
        x_est = tf.convert_to_tensor(x_est, dtype=tf.float64)
    if not isinstance(x_true, tf.Tensor):
        x_true = tf.convert_to_tensor(x_true, dtype=x_est.dtype)
        
    # Check data type and shapes
    if x_true.dtype != x_est.dtype:
        x_true = tf.cast(x_true, dtype=x_est.dtype)
    x_true = tf.reshape(x_true, tf.shape(x_est))
    
    # Compute bias and RMSE
    bias = tf.reduce_mean(x_est - x_true)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(x_true - x_est)))
    
    return bias, rmse


def profile_tf(func, *args, warmup=True, **kwargs):
    # Warmup 
    if warmup:
        func(*args, **kwargs)
    # Reset GPU memory stats
    gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
    if gpu_available:
        try:
            tf.config.experimental.reset_memory_stats("GPU:0")
        except:
            pass
            
    # Tracking time 
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    # GPU memory 
    gpu_stats = None
    if gpu_available:
        try:
            mem = tf.config.experimental.get_memory_info("GPU:0")
            gpu_stats = {
                "current_MB": mem["current"] / 1024**2,
                "peak_MB": mem["peak"] / 1024**2,
            }
        except:
            pass

    return result, {
        "runtime_seconds": end - start,
        "gpu_memory": gpu_stats
    }