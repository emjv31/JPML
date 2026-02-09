import tensorflow as tf
import math



#a) 
def compute_Sigma_tf(d, alpha0=0.3, alpha1=0.01, beta=20.0):
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
    rows = tf.range(d, dtype=tf.float32) // 8
    cols = tf.range(d, dtype=tf.float32) % 8
    positions = tf.stack([rows, cols], axis=1)  # [d,2]
    # Compute pairwise squared distances
    diff = tf.expand_dims(positions, 1) - tf.expand_dims(positions, 0)  # [d,d,2]
    dist2 = tf.reduce_sum(diff**2, axis=2)  # [d,d]
    # exponential decay
    Sigma = alpha0 * tf.exp(-dist2 / beta)
    # diagonal matrix
    Sigma += tf.eye(d) * alpha1

    return Sigma


def Sim_HD_LGSSM(d, T, alpha, sigma_z, Sigma_tf):
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

    Sigma_tf = tf.convert_to_tensor(Sigma_tf, dtype=tf.float32)
    if Sigma_tf.shape != (d, d):
        raise ValueError(f"Sigma_tf shape {Sigma_tf.shape}, expected {(d, d)}")
    tf.debugging.assert_all_finite(Sigma_tf, "Sigma contains NaN/Inf")
    
    # Check positive definiteness
    eigs = tf.linalg.eigvalsh(Sigma_tf)
    if tf.reduce_any(eigs <= 0):
        raise ValueError("Sigma is not positive definite")
    #####
    # Simulation
    #####
    # Allocation space and Choleski decomposition on the spatially correlated input matrix
    L = tf.linalg.cholesky(Sigma_tf)
    x_prev = tf.zeros(d, dtype=tf.float32)
    true_states = []
    measurements = []
    
    # Recursion
    for _ in range(T):
        # Transition noise
        eps_v = tf.random.normal((d,), dtype=tf.float32)
        v_k = tf.matmul(tf.reshape(eps_v, (1, d)), tf.transpose(L))
        v_k = tf.reshape(v_k, (-1,))
        # Measurement noise
        w_k = tf.random.normal((d,), stddev=sigma_z, dtype=tf.float32)
        # update
        x_k = alpha * x_prev + v_k
        z_k = x_k + w_k
        x_prev = x_k
        true_states.append(x_k)
        measurements.append(z_k)
#        x_prev = x_k

    true_states = tf.stack(true_states)
    measurements = tf.stack(measurements)
    ###
    ### Check output
    tf.debugging.assert_all_finite(true_states, "true_states contain NaN/Inf")
    tf.debugging.assert_all_finite(measurements, "measurements contain NaN/Inf")

    return true_states, measurements

def particle_flow_update_linear_new(p, P_pred, z, flow_type="EDH", N_lambda=50, dl=None):
    """
    Linear particle flow update supporting EDH and LEDH
    p: particles [Np,d]
    P_pred: predicted covariance [d,d]
    z: measurement [d,]
    flow_type: "EDH" (global) or "LEDH" (particle-specific)
    """

    # Input check

    p = tf.convert_to_tensor(p, dtype=tf.float32)
    P_pred = tf.convert_to_tensor(P_pred, dtype=tf.float32)
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    if flow_type.upper() not in ["EDH", "LEDH"]:
        raise ValueError(f"Unknown flow_type {flow_type}")
    Np, d = p.shape
    if P_pred.shape != (d, d):
        raise ValueError(f"P_pred shape {P_pred.shape} incompatible with particle dim {d}")
    if z.shape[0] != d:
        raise ValueError(f"z shape {z.shape} incompatible with particle dim {d}")
    if dl is None:
        dl = 1.0 / N_lambda

    # Initialization
    eta = tf.identity(p)  # copy of particles
    S = P_pred + R_tf + 1e-6 * tf.eye(d)  # innovation covariance + stability
    S_inv = tf.linalg.inv(S)

    # Flow computation
    if flow_type.upper() == "EDH":
        Ai = - P_pred @ S_inv              # global flow 
        bi = tf.reshape(P_pred @ S_inv @ tf.expand_dims(z, 1), (-1,))
        for _ in range(N_lambda):          # numerical integration
            eta = eta + dl * (eta @ Ai.T + bi[None,:])
    else:  # LEDH
        Ai = - P_pred @ S_inv
        for _ in range(N_lambda):
            dz = z[None,:] - eta           # particle-specific innovation
            bi = dz @ tf.transpose(P_pred @ S_inv)
            eta = eta + dl * (eta @ Ai.T + bi)

    theta = 1.0  # flow weight factor (linear case)
    return eta, theta


# Weight update (log-space)
def update_weight_linear_new(p_new, theta, z):
    """
    Compute particle weights given measurement z
    """
    # Input check
    p_new = tf.convert_to_tensor(p_new, dtype=tf.float32)
    z = tf.convert_to_tensor(z, dtype=tf.float32)

    if p_new.ndim != 2:
        raise ValueError(f"p_new must be [Np,d], got {p_new.shape}")
    Np, d = p_new.shape
    if z.shape[0] != d:
        raise ValueError(f"z shape {z.shape} incompatible with particle dim {d}")
    if not isinstance(theta, (float, int)):
        raise ValueError(f"theta must be a scalar, got {type(theta)}")
    if R_tf.shape != (d, d):
        raise ValueError(f"R_tf must be [{d},{d}], got {R_tf.shape}")
    eigs = tf.linalg.eigvalsh(R_tf)
    if tf.reduce_any(eigs <= 0):
        raise ValueError("R_tf must be positive definite")
        
    # Compute weights
    dz = z[None,:] - p_new               # innovation
    L = tf.linalg.cholesky(R_tf)         # measurement covariance
    solved = tf.linalg.cholesky_solve(L, tf.transpose(dz))
    quad = tf.reduce_sum(dz * tf.transpose(solved), axis=1)
    log_w = -0.5 * quad
    log_w -= tf.reduce_max(log_w)        # robust version
    w = tf.exp(log_w)
    w /= tf.reduce_sum(w)
    w *= theta
    return w


def particle_flow_pf_new(true_states, measurements, Np, P_pred, flow_type="EDH", use_weights=False):
    """
     Particle Flow Filter supporting:
      - EDH: deterministic global flow (use_weights=False)
      - LEDH: particle-specific flow (use_weights=False)
      - PFPF-EDH: deterministic global flow + weights (use_weights=True)
      - PFPF-LEDH: particle-specific flow + weights
    """
    # Input check
    if not isinstance(Np, int) or Np < 2:
        raise ValueError(f"Np must be an integer >= 2, got {Np}")
    # Convert to tensors
    true_states = tf.convert_to_tensor(true_states, dtype=tf.float32)
    measurements = tf.convert_to_tensor(measurements, dtype=tf.float32)
    P_pred = tf.convert_to_tensor(P_pred, dtype=tf.float32)
    # Validations
    if flow_type.upper() not in ["EDH", "LEDH"]:
        raise ValueError(f"flow_type must be 'EDH' or 'LEDH', got {flow_type}")
    if not isinstance(use_weights, bool):
        raise ValueError(f"use_weights must be boolean, got {type(use_weights)}")
    T = len(measurements)
    if T != true_states.shape[0]:
        raise ValueError(f"true_states and measurements must have the same length, got "
                         f"{true_states.shape[0]} and {T}")
    d = true_states[0].shape[0]
    if P_pred.shape != (d, d):
        raise ValueError(f"P_pred must be square with shape [{d},{d}], got {P_pred.shape}")
    # Check positive definiteness of P_pred
    eigs = tf.linalg.eigvalsh(P_pred)
    if tf.reduce_any(eigs <= 0):
        raise ValueError("P_pred is not positive definite")

    ###
    # Initialisation
    ###
    particles = tf.random.normal((Np, d), dtype=tf.float32)
    ests = []

    ###
    # Recursion
    ###
    for t in range(T):
        z = tf.convert_to_tensor(measurements[t], dtype=tf.float32)
        # linear prediction
        L = tf.linalg.cholesky(P_pred)
        eps = tf.random.normal((Np, d), dtype=tf.float32)
        process_noise = eps @ tf.transpose(L)
        m_pred = particles @ tf.transpose(F_tf) + process_noise
        # particle flow update
        particles, theta = particle_flow_update_linear_new(m_pred, P_pred, z, flow_type=flow_type)
        # weights (optional)
        if use_weights:
            weights = update_weight_linear_new(particles, theta, z)
        else:
            weights = tf.ones(Np, dtype=tf.float32) / Np  # all equal
        # estimation
        est = tf.reduce_sum(particles * tf.reshape(weights, (-1, 1)), axis=0)
        ests.append(est)
        # propagation
        particles = particles  

    return tf.stack(ests)



def bpf_multi_simple(true_states, measurements, Np, F, Q, sigma_z, resample_threshold=0.5):
    """
    Multivariate bootstrap particle filter for an High-Dimensional Linear State Space Model
    Resampling threshold equal to 0.5 used to avoid particle degeneracy 
    
    Args:
        true_states: list/array of shape (T, state_dim)
        measurements: list/array of shape (T, meas_dim)
        Np: number of particles
        F: state transition matrix (state_dim x state_dim)
        Q: process noise covariance (state_dim x state_dim)
        sigma_z: measurement noise standard deviation (scalar)
        resample_threshold: fraction of Np for ESS-based resampling
        
    Returns:
        ests: (T, state_dim)
        ESSs: (T,)
    """
    #######
    # Input check
    ######
    T = len(measurements)
    state_dim = true_states[0].shape[0]

    F = tf.convert_to_tensor(F, tf.float32)
    Q = tf.convert_to_tensor(Q, tf.float32)

    if F.shape != (state_dim, state_dim):
        raise ValueError(f"F shape {F.shape}, expected {(state_dim,state_dim)}")
    if Q.shape != (state_dim, state_dim):
        raise ValueError(f"Q shape {Q.shape}, expected {(state_dim,state_dim)}")

    tf.debugging.assert_all_finite(F, "F contains NaNs/Infs")
    tf.debugging.assert_all_finite(Q, "Q contains NaNs/Infs")

    eigs = tf.linalg.eigvalsh(Q)
    if tf.reduce_any(eigs <= 0):
        raise ValueError("Q is not positive definite")
    ########
    # Initialization
    #######
    particles = tf.random.normal((Np, state_dim), dtype=tf.float32)
    weights = tf.ones(Np, dtype=tf.float32) / Np
    ests = []
    ESSs = []
    # Cholesky decomposition for noise covariance matrix
    L = tf.linalg.cholesky(Q)
    # Recursion
    for t in range(T):
        z = tf.convert_to_tensor(measurements[t], dtype=tf.float32)
        # Propagate particles
        eps = tf.random.normal((Np, state_dim), dtype=tf.float32)
        process_noise = tf.matmul(eps, tf.transpose(L))
        particles = tf.matmul(particles, tf.transpose(F)) + process_noise
        ###
        # Compute weights using the Gaussian log-Likelihood - LogSumExp trick for numerical stability
        ###
        dz = z - particles  # (Np, state_dim)
        log_w = -0.5 * tf.reduce_sum(dz**2, axis=1) / sigma_z**2
        # check loglik is finite
        tf.debugging.assert_all_finite(log_w, "log_w contains NaN/Inf")
        # compute weights
        log_w -= tf.reduce_max(log_w)  
        weights = tf.exp(log_w)
        weights /= tf.reduce_sum(weights)
        # check weights are finite and not degenerate
        tf.debugging.assert_all_finite(weights, "weights contain NaN/Inf") 
        if tf.reduce_max(weights) > 0.999:
            tf.print("Weight degenerate at time", t)
            
        # Compute ESS and resample if needed
        ESS = 1.0 / tf.reduce_sum(weights**2)
        ESSs.append(ESS)
        if ESS < 0.05 * Np:
            tf.print("Severe degeneracy at time", t, "ESS =", ESS)
        if ESS < resample_threshold * Np:
            idx = tf.random.categorical(tf.math.log([weights]), Np)
            idx = tf.reshape(idx, (-1,))
            particles = tf.gather(particles, idx)
            weights = tf.ones(Np, dtype=tf.float32) / Np

        # Estimate
        est = tf.reduce_sum(particles * tf.reshape(weights, (-1,1)), axis=0)
        ests.append(est)

    return tf.stack(ests), tf.stack(ESSs)


## KF, UKF and EKF. UKF is not working. to fix

def ukf_predict(x, P, f, Q, alpha=1e-3, beta=2.0, kappa=0.0, t=0):
    x = tf.cast(x, tf.float64)
    P = tf.cast(P, tf.float64)
    Q = tf.cast(Q, tf.float64)

    n = x.shape[0]
    lam = alpha**2 * (n + kappa) - n
    c = n + lam

    Wm = tf.concat([[lam/c], tf.fill([2*n], 1/(2*c))], axis=0)
    Wc = tf.concat([[lam/c + (1 - alpha**2 + beta)], tf.fill([2*n], 1/(2*c))], axis=0)
    Wm = tf.cast(Wm, tf.float64)
    Wc = tf.cast(Wc, tf.float64)

    sqrtP = tf.linalg.cholesky(c * P)
    sigma = tf.concat(
        [x[None, :],
         x + sqrtP,
         x - sqrtP], axis=0
    )

    sigma_f = tf.stack(
        [tf.cast(f(sigma[i], t), tf.float64) for i in range(2*n+1)],
        axis=0
    )

    x_pred = tf.reduce_sum(sigma_f * Wm[:, None], axis=0)

    diff = sigma_f - x_pred
    P_pred = Q + tf.reduce_sum(
        [Wc[i] * tf.tensordot(diff[i], diff[i], 0) for i in range(2*n+1)],
        axis=0
    )

    return x_pred, P_pred


def ukf_update(x_pred, P_pred, y, h, R, alpha=1e-3, beta=2.0, kappa=0.0, t=0):
    x_pred = tf.cast(x_pred, tf.float64)
    P_pred = tf.cast(P_pred, tf.float64)
    y = tf.cast(y, tf.float64)
    R = tf.cast(R, tf.float64)

    n = x_pred.shape[0]
    m = y.shape[0]

    lam = alpha**2 * (n + kappa) - n
    c = n + lam

    Wm = tf.concat([[lam/c], tf.fill([2*n], 1/(2*c))], axis=0)
    Wc = tf.concat([[lam/c + (1 - alpha**2 + beta)], tf.fill([2*n], 1/(2*c))], axis=0)
    Wm = tf.cast(Wm, tf.float64)
    Wc = tf.cast(Wc, tf.float64)

    sqrtP = tf.linalg.cholesky(c * P_pred)
    sigma = tf.concat([x_pred[None, :], x_pred + sqrtP, x_pred - sqrtP], axis=0)
    sigma_y = tf.stack([tf.cast(h(sigma[i], t), tf.float64) for i in range(2*n+1)], axis=0)
    y_pred = tf.reduce_sum(sigma_y * Wm[:, None], axis=0)

    dy = sigma_y - y_pred
    dx = sigma - x_pred

    S = R + tf.reduce_sum([Wc[i] * tf.tensordot(dy[i], dy[i], 0) for i in range(2*n+1)], axis=0)
    Pxz = tf.reduce_sum([Wc[i] * tf.tensordot(dx[i], dy[i], 0) for i in range(2*n+1)], axis=0)
    K = Pxz @ tf.linalg.inv(S)

    v_t = y - y_pred
    x_filt = x_pred + tf.linalg.matvec(K, y - y_pred)
    P_filt = P_pred - K @ S @ tf.transpose(K)

    return x_filt, P_filt, v_t, S, K


def state_space_filter(Y, F_func, H_func, Q_mat, R_mat, m0, P0, method="KF", **kwargs):
    """
    General multivariate state-space filter supporting:
        - KF: Kalman Filter (linear)
        - EKF: Extended Kalman Filter (nonlinear, linearize using Jacobians)
        - UKF: Unscented Kalman Filter (nonlinear, sigma points)
    
    Args:
        Y       : (n_y, T) observation matrix
        F_func  : function f(x, t) -> x_next (state transition)
                  for KF/EKF, can be constant matrix (linear)
        H_func  : function h(x, t) -> y (measurement)
                  for KF/EKF, can be constant matrix (linear)
        Q_mat   : (n_x, n_x) process noise covariance
        R_mat   : (n_y, n_y) measurement noise covariance
        m0      : (n_x,) initial mean
        P0      : (n_x, n_x) initial covariance
        method  : "KF", "EKF", or "UKF"
        kwargs  : extra parameters (e.g., UKF alpha, beta, kappa)
    
    Returns:
        Dictionary of filtered means, covariances, innovations, gains, log-likelihood
    """
    # Convert everything to float64 tensors
    Y = tf.convert_to_tensor(Y, dtype=tf.float64)
    Q_mat = tf.convert_to_tensor(Q_mat, dtype=tf.float64)
    R_mat = tf.convert_to_tensor(R_mat, dtype=tf.float64)
    m0 = tf.convert_to_tensor(m0, dtype=tf.float64)
    P0 = tf.convert_to_tensor(P0, dtype=tf.float64)
    
    n_y, T = Y.shape
    n_y = tf.cast(n_y, tf.float64)
    n_x = m0.shape[0]
    I = tf.eye(n_x, dtype=tf.float64)
    log2pi = tf.math.log(tf.constant(2.0 * np.pi, dtype=tf.float64))
    
    # Allocate tensors
    mu_pred = tf.TensorArray(tf.float64, size=T)
    P_pred = tf.TensorArray(tf.float64, size=T)
    mu_filt = tf.TensorArray(tf.float64, size=T)
    P_filt = tf.TensorArray(tf.float64, size=T)
    innovations = tf.TensorArray(tf.float64, size=T)
    F_innov = tf.TensorArray(tf.float64, size=T)
    K_arr = tf.TensorArray(tf.float64, size=T)
    loglik_vec = tf.TensorArray(tf.float64, size=T)
    
    # Init
    mu_pred = mu_pred.write(0, m0)
    P_pred = P_pred.write(0, P0)
    loglik = tf.constant(0.0, dtype=tf.float64)
    
    for t in range(T):
        x_pred = mu_pred.read(t)
        P_pred_t = P_pred.read(t)
        y_t = Y[:, t]
        
        if method == "KF":
            F_t = F_func if isinstance(F_func, tf.Tensor) else F_func(t)
            H_t = H_func if isinstance(H_func, tf.Tensor) else H_func(t)
            x_pred_vec = tf.linalg.matvec(F_t, x_pred)
            P_pred_t = F_t @ P_pred_t @ tf.transpose(F_t) + Q_mat
            v_t = y_t - tf.linalg.matvec(H_t, x_pred_vec)
            innovations = innovations.write(t, v_t)
            S_t = H_t @ P_pred_t @ tf.transpose(H_t) + R_mat
            F_innov = F_innov.write(t, S_t)
            K_t = P_pred_t @ tf.transpose(H_t) @ tf.linalg.inv(S_t)
            K_arr = K_arr.write(t, K_t)
            x_filt = x_pred_vec + tf.linalg.matvec(K_t, v_t)
            mu_filt = mu_filt.write(t, x_filt)
            P_filt = P_filt.write(t, (I - K_t @ H_t) @ P_pred_t)
        elif method == "EKF":
            # Linearize F and H using Jacobians at current estimate
            F_t = kwargs['F_jac'](x_pred, t)   
            H_t = kwargs['H_jac'](x_pred, t)   
            x_pred_vec = F_func(x_pred, t)     # nonlinear prediction
            P_pred_t = F_t @ P_pred_t @ tf.transpose(F_t) + Q_mat
            v_t = y_t - tf.linalg.matvec(H_t, x_pred_vec)
            innovations = innovations.write(t, v_t)
            S_t = H_t @ P_pred_t @ tf.transpose(H_t) + R_mat
            F_innov = F_innov.write(t, S_t)
            K_t = P_pred_t @ tf.transpose(H_t) @ tf.linalg.inv(S_t)
            K_arr = K_arr.write(t, K_t)
            x_filt = x_pred_vec + tf.linalg.matvec(K_t, v_t)
            mu_filt = mu_filt.write(t, x_filt)
            P_filt = P_filt.write(t, (I - K_t @ H_t) @ P_pred_t)
        elif method == "UKF":
            alpha = kwargs.get("alpha", 1e-3)
            beta = kwargs.get("beta", 2.0)
            kappa = kwargs.get("kappa", 0.0)
            x_pred, P_pred_t = ukf_predict(x_pred, P_pred_t, F_func, Q_mat, alpha, beta, kappa, t)
            x_filt, P_f, v_t, S_t, K_t = ukf_update(x_pred, P_pred_t, y_t, H_func, R_mat, alpha, beta, kappa, t)
            #P_filt = P_pred.write(t+1, P_f)   # filtered covariance from UKF
        else:
            raise ValueError(f"Unknown method {method}")
        
        # Log-likelihood
        v_col = tf.reshape(v_t, (-1,1))
        neg_half = tf.constant(-0.5, dtype=tf.float64)
        ll = neg_half * (tf.cast(n_y, tf.float64) * log2pi + tf.math.log(tf.linalg.det(S_t)) + tf.transpose(v_col) @ tf.linalg.inv(S_t) @ v_col)[0,0]
        #ll = -0.5 * (n_y * log2pi + tf.math.log(tf.linalg.det(S_t)) + tf.transpose(v_col) @ tf.linalg.inv(S_t) @ v_col)[0,0]
        loglik_vec = loglik_vec.write(t, ll)
        loglik += ll
        
        if t < T-1:
            mu_pred = mu_pred.write(t+1, x_filt)
            P_pred = P_pred.write(t+1, P_filt.read(t))
    
    return {
        "mu_filt": tf.transpose(mu_filt.stack()),
        "P_filt": P_filt.stack(),
        "mu_pred": tf.transpose(mu_pred.stack()),
        "P_pred": P_pred.stack(),
        "innovations": tf.transpose(innovations.stack()),
        "F_innov": F_innov.stack(),
        "K": K_arr.stack(),
        "loglik_vec": loglik_vec.stack(),
        "loglik": loglik
    }


def esrf(true_states, measurements, Np, F, Q, H, R):
    """
    Ensemble Square Root Filter (ESRF) for a linear Gaussian state-space models.

    Inputs:
        true_states : list or tensor of shape [T, n_x] 
                      (optional, only used for reference; does not affect filtering)
        measurements: list (or tensor) of observations at each time step [T, n_y]
        Np          : int, number of particles
        F           : state transition matrix [n_x, n_x]
        Q           : process noise covariance (PSD) matrix [n_x, n_x]
        H           : observation matrix [n_y, n_x]
        R           : observation noise covariance (PSD) matrix [n_y, n_y]

    Returns:
        estimated state at each time step (ensemble mean)
    """
    T = len(measurements)
    state_dim = true_states[0].shape[0]

    particles = tf.random.normal((Np, state_dim))
    L = tf.linalg.cholesky(Q)
    ests = []

    for t in range(T):
        z = measurements[t]  
        # propagate particles
        eps = tf.random.normal((Np, state_dim))
        process_noise = tf.matmul(eps, tf.transpose(L))
        particles = tf.matmul(particles, tf.transpose(F)) + process_noise
        # compute ensemble mean and covariance
        x_mean = tf.reduce_mean(particles, axis=0, keepdims=True)
        X = particles - x_mean
        P_ens = tf.matmul(tf.transpose(X), X) / (Np-1)
        # ESRF update - deterministic
        S = H @ P_ens @ tf.transpose(H) + R
        K = P_ens @ tf.transpose(H) @ tf.linalg.inv(S)
        # Compute innovations for each particle
        Hx = tf.transpose(tf.matmul(H, tf.transpose(particles)))  
        dz = z - Hx  
        # Update particles
        particles = particles + dz @ tf.transpose(K) 
        # Estimate
        ests.append(tf.reduce_mean(particles, axis=0))

    return tf.stack(ests)


#### Simulation setup for example C) Li's paper 
#### Large spatial sensor networks: Skewed-t dynamic model and count measurements

def sample_skewed_t(x_prev, alpha, Sigma, gamma, nu):
    """
    Sample a high-dimensional skewed-t vector given previous state.
    Input: 
    x_prev: 
    Sigma: process covariance
    gamma: skew vector
    """
    d = x_prev.shape[0]
    # linear dynamics
    mu_k = alpha * x_prev
    # sampling scaling factor w ~ Gamma
    w = tf.random.gamma(shape=(1,), alpha=nu/2.0, beta=nu/2.0, dtype=tf.float32)
    # Gaussian noise and Choleski decomp.
    z = tf.random.normal((d,1), dtype=tf.float32)      
    L = tf.linalg.cholesky(Sigma)    
    # skewness vector
    gamma = tf.reshape(gamma, (d,1))                  
    # skewed-t transformation
    x_k = mu_k[:, None] + (1/tf.sqrt(w)) * (L @ z + gamma) # skewed-t sample
    x_k = tf.reshape(x_k, (-1,))                      # back to [d]
    return x_k

# ----------------------------
# Simulation
# ----------------------------
def simulate_skewed_t_HDS(d=144, T=10, alpha=0.9, nu=5.0, num_runs=1):
    """
    Simulate high-dimensional skewed-t state variable (GH distribution) 
    in a dynamic linear model with a spatial correlated process noise
    Input: 
    d: int, Dimension of the state vector
    T: int, Number of observations
    alpha: float, autoregression coefficient. State process stationary if |alpha|<1
    nu : float, Degrees of freedom of the skewed-t distribution
    num_runs : int, Number of Monte Carlo repetitions
    Outputs:
        all_runs: tf.Tensor of simulated states with shape [num_runs, T, d] 
    """
    gamma = tf.ones((d,1), dtype=tf.float32)
    Sigma_foo = compute_Sigma_tf(d)
    Sigma = (nu / (nu - 2)) * Sigma_foo + (nu/2)*(2*nu - 8)/((nu/2 - 1)**2) * gamma @ tf.transpose(gamma)
    
    all_runs = []
    for run in range(num_runs):
        x_prev = tf.zeros(d, dtype=tf.float32)
        run_states = []
        for t in range(T):
            x_k = sample_skewed_t(x_prev, alpha, Sigma, gamma, nu)
            run_states.append(x_k)
            x_prev = x_k
        all_runs.append(tf.stack(run_states))
    return tf.stack(all_runs)  # [num_runs, T, d]



# b) 
# SCALAR KERNEL Hu's paper

def compute_kernel(x, h):
    """
    x: [Np, d]
    returns:
        K: [Np, Np]
        grad_K: [Np, Np, d]
    """
    x_i = x[:, None, :]     # [1, Np, d]
    x_j = x[None, :, :]     # [1, Np, d]

    diff = x_i - x_j        # [Np, Np, d]
    sqdist = tf.reduce_sum(diff**2, axis=2)  # [Np, Np]

    K = tf.exp(-sqdist / (2.0 * h**2))        # [Np, Np]
    grad_K = -diff / (h**2) * K[:, :, None]   # [Np, Np, d]

    return K, grad_K

def grad_log_posterior(x, x_pred, z, Q, R):
    """
    x, x_pred: [Np, d]
    z: [d]
    """
    Qinv = tf.linalg.inv(Q)
    Rinv = tf.linalg.inv(R)

    term_prior = -tf.matmul(x - x_pred, Qinv)
    term_lik   = -tf.matmul(x - z[None, :], Rinv)

    return term_prior + term_lik   

# Update single step
def hu_particle_flow_update(particles, x_pred, z, Q, R, n_flow_steps=20, step_size=0.1, kernel_bw=1.0):
    """
    particles, x_pred: [Np, d]
    z: [d]
    """
    Np, d = particles.shape
    D = Q  # preconditioner (Hu: covariance-based)

    for _ in range(n_flow_steps):
        # kernel
        K, grad_K = compute_kernel(particles, kernel_bw)
        # grad log posterior
        grad_logp = grad_log_posterior(particles, x_pred, z, Q, R)

        # forces
        f_repelling = tf.reduce_sum(grad_K, axis=1)        # [Np, d]
        f_attract   = tf.matmul(K, grad_logp)              # [Np, d]

        flow = tf.matmul(f_repelling + f_attract, tf.transpose(D))  # [Np, d]

        particles = particles + step_size * flow

    return particles


def hu_pff_filter(T = 5, Np = 20): 
    d = 2

    Q = 0.1 * tf.eye(d, dtype=tf.float32)
    R = 0.2 * tf.eye(d, dtype=tf.float32)

    # True states and observations 
    x_true = [tf.zeros(d, dtype=tf.float32)]
    z_meas = []

    LQ = tf.linalg.cholesky(Q)
    LR = tf.linalg.cholesky(R)

    for _ in range(T):
        process_noise = tf.linalg.matvec(
            LQ, tf.random.normal((d,), dtype=tf.float32)
        )
        x_next = x_true[-1] + process_noise
        x_true.append(x_next)

        meas_noise = tf.linalg.matvec(
            LR, tf.random.normal((d,), dtype=tf.float32)
        )
        z_meas.append(x_next + meas_noise)

    x_true = tf.stack(x_true[1:])   # [T, d]
    z_meas = tf.stack(z_meas)       # [T, d]

    # Particles 
    particles = tf.random.normal((Np, d), dtype=tf.float32)
    estimates = []

    for t in range(T):
        x_pred = particles

        particles = hu_particle_flow_update(
            particles,
            x_pred,
            z_meas[t],
            Q, R,
            n_flow_steps=15,
            step_size=0.05,
            kernel_bw=1.0
        )

        estimates.append(tf.reduce_mean(particles, axis=0))

    return x_true, z_meas, tf.stack(estimates), particles


# ----------------------------
# Matrix-valued kernel functions
# ----------------------------
def matrix_valued_kernel(X, alpha=1.0):
    """
    Compute Gaussian kernel and its gradient
    X: [Np, d]
    Returns:
      K: [Np, Np]
      gradK: [Np, Np, d]
    """
    Np, d = X.shape
    diff = X[:, None, :] - X[None, :, :]   # [Np, Np, d]
    sqdist = tf.reduce_sum(diff**2, axis=2)
    K = tf.exp(-sqdist / alpha)             # [Np, Np]
    gradK = -2.0 / alpha * diff * K[:, :, None]
    return K, gradK

def hu_matrix_kernel_flow(particles, z, Q, R, n_steps=10, eps=0.1):
    """
    Matrix-valued kernel flow update
    particles: [Np, d]
    z:         [d]
    """
    tf.debugging.assert_rank(particles, 2)
    tf.debugging.assert_rank(z, 1)

    Np, d = particles.shape
    R_inv = tf.linalg.inv(R)
    Q_inv = tf.linalg.inv(Q)
    M = Q  # matrix-valued kernel preconditioner

    for _ in range(n_steps):
        # Posterior gradient
        grad_ll = tf.matmul(z[None, :] - particles, R_inv)
        grad_prior = -tf.matmul(particles, Q_inv)
        grad_log_post = grad_ll + grad_prior  # [Np, d]

        # Kernel interactions
        K, gradK = matrix_valued_kernel(particles)
        attract = tf.matmul(K, grad_log_post)  # [Np, d]
        repel = tf.reduce_sum(gradK, axis=1)   # [Np, d]

        flow = tf.matmul(attract + repel, tf.transpose(M))
        particles = particles + eps * flow

        tf.debugging.assert_all_finite(particles, "Particle explosion!")

    return particles

# ----------------------------
# Full Hu matrix-valued PFF
# ----------------------------
def hu_pff_matrix_kernel_example(T=10, Np=100, d=10, alpha=1.0, eps=0.1):
    """
    Example using Sim_HD_LGSSM to simulate the system
    """
    # Simulate high-dimensional LGSSM using the same function of the previous question
    x_true, z_meas = Sim_HD_LGSSM(d=d, T=T, alpha=0.9, sigma_z=0.1, Sigma_tf=0.1 * tf.eye(d, dtype=tf.float32))
    # we need to define the system matrices
    F_hu = 0.9 * tf.eye(d, dtype=tf.float32)   # simple dynamics
    Q_hu = 0.1 * tf.eye(d, dtype=tf.float32)   # process noise
    H_hu = tf.eye(d, dtype=tf.float32)         # observation matrix
    R_hu = 0.1 * tf.eye(d, dtype=tf.float32)   # observation noise

    # Initialisation and allocation space
    particles = tf.random.normal((Np, d), dtype=tf.float32)
    ests_hu = []

    for t in range(T):
        # Predict step: add process noise
        eps_noise = tf.random.normal((Np, d), dtype=tf.float32)
        LQ = tf.linalg.cholesky(Q_hu)
        process_noise = tf.matmul(eps_noise, tf.transpose(LQ))
        particles = tf.matmul(particles, tf.transpose(F_hu)) + process_noise
        # Update with matrix-valued kernel flow
        particles = hu_matrix_kernel_flow(particles, z_meas[t], Q_hu, R_hu, n_steps=10, eps=eps)

        # Estimate mean
        ests_hu.append(tf.reduce_mean(particles, axis=0))

    return x_true, z_meas, tf.stack(ests_hu), particles



## APPLICATION 
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf
import numpy as np

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

mu_ukf_hd = ukf_result_hd["mu_filt"]  # 

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
    Compare particle spread (marginals) for scalar vs matrix-valued kernel.
    
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


# III)
d = 64
Np = 500
#P_pred_foo = Q_tf
P_pred_foo = tf.eye(d) * 0.1
R_tf_foo = tf.eye(d) * sigma_z**2

#
pfpf_edh = particle_flow_pf_new(true_states, measurements, Np = 200, P_pred = P_pred_foo, flow_type="EDH", use_weights=False)
pfpf_edh_w = particle_flow_pf_new(true_states, measurements, Np = 200, P_pred = P_pred_foo, flow_type="EDH", use_weights=True)
pfpf_ledh_w = particle_flow_pf_new(true_states, measurements, Np = 200, P_pred = P_pred_foo, flow_type="LEDH", use_weights=True)