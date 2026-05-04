import tensorflow as tf


########### DESIGN SV MODEL 

### UTILS
def _infer_dim_from_params(phi, sigma_eta, sigma_eps, xi, d=None):
    if d is not None:
        if not isinstance(d, int) or d <= 0:
            raise ValueError("d must be a positive integer")
        return d

    candidates = []

    for obj in [phi, sigma_eta, sigma_eps, xi]:
        t = tf.convert_to_tensor(obj)
        if t.shape.rank == 0:
            continue
        if t.shape.rank == 1:
            candidates.append(int(t.shape[0]))
        elif t.shape.rank == 2:
            if t.shape[0] != t.shape[1]:
                raise ValueError("Matrix parameters must be square")
            candidates.append(int(t.shape[0]))
        else:
            raise ValueError("Parameters must be scalar, vector, or matrix")

    if len(candidates) == 0:
        return 1

    d0 = candidates[0]
    if any(c != d0 for c in candidates):
        raise ValueError("Incompatible dimensions across phi / sigma_eta / sigma_eps / xi")
    return d0


def _make_covariance_and_chol_tf(sigma_like, d, dtype, name):
    """
    Accepts:
      - scalar std
      - vector of stds, length d
      - covariance matrix (d,d)
    Returns:
      Q, chol(Q)
    """
    sigma_like = tf.convert_to_tensor(sigma_like, dtype=dtype)

    if sigma_like.shape.rank == 0:
        tf.debugging.assert_positive(sigma_like, message=f"{name} must be positive")
        Q = tf.eye(d, dtype=dtype) * sigma_like**2
        L = tf.eye(d, dtype=dtype) * sigma_like
        return Q, L

    if sigma_like.shape.rank == 1:
        tf.debugging.assert_equal(
            tf.shape(sigma_like)[0], d,
            message=f"{name} vector must have length d"
        )
        tf.debugging.assert_positive(
            sigma_like,
            message=f"all entries of {name} must be positive"
        )
        Q = tf.linalg.diag(sigma_like**2)
        L = tf.linalg.diag(sigma_like)
        return Q, L

    if sigma_like.shape.rank == 2:
        tf.debugging.assert_equal(
            tf.shape(sigma_like)[0], d,
            message=f"{name} matrix must have shape (d,d)"
        )
        tf.debugging.assert_equal(
            tf.shape(sigma_like)[1], d,
            message=f"{name} matrix must have shape (d,d)"
        )
        L = tf.linalg.cholesky(sigma_like)
        return sigma_like, L

    raise ValueError(f"{name} must be scalar, vector, or matrix")


def _make_scale_vector_tf(xi, d, dtype):
    """
    Scalar -> length-d vector, or validate length-d vector.
    Enforces positivity.
    """
    xi_vec = _as_vector_param_tf(xi, d, dtype, "xi")
    tf.debugging.assert_positive(xi_vec, message="xi must be positive")
    return xi_vec


def _as_vector_param_tf(x, d, dtype, name):
    """
    Convert scalar -> vector of length d
    or validate vector of length d.
    """
    x = tf.convert_to_tensor(x, dtype=dtype)

    if x.shape.rank == 0:
        return tf.fill([d], x)

    if x.shape.rank == 1:
        tf.debugging.assert_equal(
            tf.shape(x)[0], d,
            message=f"{name} must have length d"
        )
        return x

    raise ValueError(f"{name} must be scalar or a rank-1 tensor")


def _as_transition_matrix_tf(phi, d, dtype):
    """
    phi can be:
      - scalar -> phi * I
      - vector -> diag(phi)
      - matrix -> full transition matrix
    """
    phi = tf.convert_to_tensor(phi, dtype=dtype)

    if phi.shape.rank == 0:
        tf.debugging.assert_less(
            tf.abs(phi), tf.cast(1.0, dtype),
            message="scalar phi must satisfy |phi| < 1"
        )
        return phi * tf.eye(d, dtype=dtype)

    if phi.shape.rank == 1:
        tf.debugging.assert_equal(
            tf.shape(phi)[0], d,
            message="phi vector must have length d"
        )
        tf.debugging.assert_less(
            tf.reduce_max(tf.abs(phi)),
            tf.cast(1.0, dtype),
            message="all phi entries must satisfy |phi_i| < 1"
        )
        return tf.linalg.diag(phi)

    if phi.shape.rank == 2:
        tf.debugging.assert_equal(
            tf.shape(phi)[0], d,
            message="phi matrix must have shape (d,d)"
        )
        tf.debugging.assert_equal(
            tf.shape(phi)[1], d,
            message="phi matrix must have shape (d,d)"
        )
        eigvals = tf.linalg.eigvals(tf.cast(phi, tf.complex128))
        tf.debugging.assert_less(
            tf.reduce_max(tf.abs(eigvals)),
            tf.constant(1.0, dtype=tf.float64),
            message="phi matrix must be stable: spectral radius < 1"
        )
        return phi

    raise ValueError("phi must be scalar, vector, or matrix")


def _stationary_covariance_ar1(Phi, Q, dtype):
    """
    Solve vec(P) = (I - Phi⊗Phi)^(-1) vec(Q)
    for the stationary covariance of h_t = Phi h_{t-1} + eta_t.
    """
    d = int(Phi.shape[0])
    kron_term = tf.experimental.numpy.kron(Phi, Phi)
    I = tf.eye(d * d, dtype=dtype)
    vecQ = tf.reshape(Q, (-1, 1))
    vecP = tf.linalg.solve(I - kron_term, vecQ)
    P = tf.reshape(vecP, (d, d))
    P = 0.5 * (P + tf.transpose(P))  # symmetrize
    return P



def _as_noise_chol_tf(sigma_like, d, dtype, name):
    """
    Accept:
      - scalar std
      - vector of stds length d
      - covariance matrix (d,d)

    Return Cholesky factor L such that noise = L z, z~N(0,I).
    Pure TensorFlow, graph-safe.
    """
    sigma_like = tf.convert_to_tensor(sigma_like, dtype=dtype)

    if sigma_like.shape.rank == 0:
        tf.debugging.assert_positive(sigma_like, message=f"{name} must be positive")
        return sigma_like * tf.eye(d, dtype=dtype)

    if sigma_like.shape.rank == 1:
        tf.debugging.assert_equal(
            tf.shape(sigma_like)[0], d,
            message=f"{name} vector must have length d"
        )
        tf.debugging.assert_positive(
            sigma_like,
            message=f"all entries of {name} must be positive"
        )
        return tf.linalg.diag(sigma_like)

    if sigma_like.shape.rank == 2:
        tf.debugging.assert_equal(
            tf.shape(sigma_like)[0], d,
            message=f"{name} matrix must have shape (d,d)"
        )
        tf.debugging.assert_equal(
            tf.shape(sigma_like)[1], d,
            message=f"{name} matrix must have shape (d,d)"
        )
        return tf.linalg.cholesky(sigma_like)

    raise ValueError(f"{name} must be scalar, vector, or matrix")



def SV_model_sim_tf_h(iT, phi, sigma_eta, sigma_eps, xi, seed=123, d=None, dtype=tf.float32):
    """
    Gaussian stochastic volatility simulator

    Supporting:
      - univariate Y shape (T,)
      - multivariate Y shape (T,d) 
    """
    if not isinstance(iT, int) or iT <= 0:
        raise ValueError("iT must be a positive integer")

    if seed is not None:
        tf.random.set_seed(seed)

    d = _infer_dim_from_params(phi, sigma_eta, sigma_eps, xi, d=d)

    Phi = _as_transition_matrix_tf(phi, d, dtype)
    Q_eta, chol_eta = _make_covariance_and_chol_tf(sigma_eta, d, dtype, "sigma_eta")
    _, chol_eps = _make_covariance_and_chol_tf(sigma_eps, d, dtype, "sigma_eps")
    xi_vec = _make_scale_vector_tf(xi, d, dtype)

    z_eta = tf.random.normal((iT, d), dtype=dtype)
    z_eps = tf.random.normal((iT, d), dtype=dtype)

    eta = tf.linalg.matmul(z_eta, tf.transpose(chol_eta))
    eps = tf.linalg.matmul(z_eps, tf.transpose(chol_eps))

    P0 = _stationary_covariance_ar1(Phi, Q_eta, dtype)
    chol_P0 = tf.linalg.cholesky(P0 + tf.cast(1e-12, dtype) * tf.eye(d, dtype=dtype))

    z0 = tf.random.normal((d,), dtype=dtype)
    h0 = tf.linalg.matvec(chol_P0, z0)

    if iT == 1:
        h = h0[None, :]
    else:
        h_rest = tf.scan(
            lambda h_prev, eta_t: tf.linalg.matvec(Phi, h_prev) + eta_t,
            eta[1:],
            initializer=h0
        )
        h = tf.concat([h0[None, :], h_rest], axis=0)

    y = xi_vec[None, :] * eps * tf.exp(h / tf.cast(2.0, dtype))

    if d == 1:
        y = tf.squeeze(y, axis=-1)
        h = tf.squeeze(h, axis=-1)

    tf.debugging.assert_all_finite(h, "latent path h contains NaN/Inf")
    tf.debugging.assert_all_finite(y, "observation path y contains NaN/Inf")

    return {"vY": y, "h": h}


#### BPF SPECIFIC UTILITIES IN ADDITION TO THOSE OF THE SV MODEL
def make_prop_sv(phi, sigma_eta, dtype=tf.float32):
    """
    SV state propagation:
        h_t = Phi h_{t-1} + eta_t

    Works for:
      - univariate: phi scalar, sigma_eta scalar
      - multivariate independent: phi scalar/vector, sigma_eta scalar/vector
      - multivariate correlated state noise: sigma_eta covariance matrix

    Input x must be shape (d,).
    """
    phi = tf.convert_to_tensor(phi, dtype=dtype)
    sigma_eta = tf.convert_to_tensor(sigma_eta, dtype=dtype)

    @tf.function(reduce_retracing=True)
    def prop_sv(x):
        x = tf.convert_to_tensor(x, dtype=dtype)

        tf.debugging.assert_rank(x, 1, message="state x must have shape (d,)")

        d = tf.shape(x)[0]

        Phi = _as_transition_matrix_tf(phi, d, dtype)
        L_eta = _as_noise_chol_tf(sigma_eta, d, dtype, "sigma_eta")

        z = tf.random.normal(tf.shape(x), dtype=dtype)
        eta = tf.linalg.matvec(L_eta, z)

        x_next = tf.linalg.matvec(Phi, x) + eta
        tf.debugging.assert_all_finite(x_next, "prop_sv produced NaN/Inf")

        return x_next

    return prop_sv


def make_loglik_sv(sigma_eps, xi, dtype=tf.float32):
    """
    Gaussian SV observation log-likelihood under conditional independence:
        y_t = xi * exp(h_t / 2) * eps_t

    particles: (Np, d)  
    y:         (d,)
    returns:   (Np,)
    """
    sigma_eps = tf.convert_to_tensor(sigma_eps, dtype=dtype)
    xi = tf.convert_to_tensor(xi, dtype=dtype)

    @tf.function(reduce_retracing=True)
    def loglik_sv(particles, y):
        particles = tf.convert_to_tensor(particles, dtype=dtype)
        y = tf.convert_to_tensor(y, dtype=dtype)

        tf.debugging.assert_rank(particles, 2, message="particles must have shape (Np, d)")
        tf.debugging.assert_rank(y, 1, message="y must have shape (d,)")

        d = tf.shape(particles)[1]
        tf.debugging.assert_equal(
            tf.shape(y)[0], d,
            message="y dimension must match particle dimension"
        )

        sigma_eps_vec = _as_vector_param_tf(sigma_eps, d, dtype, "sigma_eps")
        xi_vec = _as_vector_param_tf(xi, d, dtype, "xi")

        tf.debugging.assert_positive(sigma_eps_vec, message="sigma_eps must be positive")
        tf.debugging.assert_positive(xi_vec, message="xi must be positive")

        # log variance for each particle and dimension:
        # log var = log(xi^2 sigma_eps^2 exp(h)) = 2 log xi + 2 log sigma_eps + h
        log_var = (
            2.0 * tf.math.log(xi_vec)[None, :]
            + 2.0 * tf.math.log(sigma_eps_vec)[None, :]
            + particles
        )

        var = tf.exp(log_var)
        log_2pi = tf.math.log(tf.constant(2.0 * 3.14159265358979323846, dtype=dtype))

        llk = -0.5 * tf.reduce_sum(
            log_2pi + log_var + tf.square(y[None, :]) / var,
            axis=1
        )

        tf.debugging.assert_all_finite(llk, "SV log-likelihood produced NaN/Inf")
        return llk

    return loglik_sv


def multinomial_resampling(particles, weights): 

    particles = tf.convert_to_tensor(particles)
    dtype = particles.dtype
    weights = tf.convert_to_tensor(weights, dtype=dtype)
    
    N = tf.shape(particles)[0]
    idx = tf.random.categorical(tf.expand_dims(tf.math.log(weights), 0), N)[0]
    new_particles = tf.gather(particles, idx)
    new_weights = tf.ones_like(weights)/tf.cast(N, dtype=dtype)

    return new_particles, new_weights

#### Bootstrap Particle Filter
def bpf_generic_resampling(
    Y,
    Np,
    prop_fn,
    log_likelihood_fn,
    resampling_fn=multinomial_resampling,
    resample_threshold=False,
    dtype=tf.float32,
    carry_resampled_weights=False,
    eps=1e-12,
):
    """
    Generic bootstrap particle filter.

    carry_resampled_weights=False:
        use for multinomial / OT / equal-weight output resamplers

    carry_resampled_weights=True:
        use for no-resampling / PFNet soft-resampling
    """
    if not callable(prop_fn):
        raise TypeError("prop_fn must be callable")
    if not callable(log_likelihood_fn):
        raise TypeError("log_likelihood_fn must be callable")
    if not callable(resampling_fn):
        raise TypeError("resampling_fn must be callable")
    if not isinstance(Np, int) or Np <= 1:
        raise ValueError("Np must be an integer > 1")

    Y = tf.cast(tf.convert_to_tensor(Y), dtype)
    if len(Y.shape) != 2:
        raise ValueError("Y must have shape (T, d)")
    if Y.shape[0] < 1:
        raise ValueError("Y must contain at least one time step")

    T, d = Y.shape
    eps = tf.cast(eps, dtype)

    particles = tf.zeros([Np, d], dtype=dtype)
    log_weights = -tf.math.log(tf.cast(Np, dtype)) * tf.ones([Np], dtype=dtype)

    total_loglik = tf.constant(0.0, dtype=dtype)
    ests, ESSs = [], []

    threshold_value = tf.cast(Np / 2.0, dtype)

    for t in range(T):
        y_t = Y[t]

        # Propagation
        particles = tf.vectorized_map(prop_fn, particles)

        # likelihood
        llk = tf.cast(log_likelihood_fn(particles, y_t), dtype)
        tf.debugging.assert_all_finite(llk, "llk has NaN/Inf")

        # generic log-weight recursion
        log_w_unnorm = log_weights + llk
        loglik_t = tf.reduce_logsumexp(log_w_unnorm)
        total_loglik += loglik_t

        # normalize in log-space
        log_weights = log_w_unnorm - loglik_t
        weights = tf.exp(log_weights)

        tf.debugging.assert_all_finite(weights, "weights have NaN/Inf")

        # estimate
        est = tf.reduce_sum(particles * weights[:, None], axis=0)
        ests.append(est)

        # ESS
        ESS = 1.0 / tf.reduce_sum(tf.square(weights))
        ESSs.append(ESS)

        do_resample = ESS < threshold_value if resample_threshold else True

        if do_resample:
            particles, new_weights = resampling_fn(particles, weights)
            new_weights = tf.cast(new_weights, dtype)
            tf.debugging.assert_all_finite(new_weights, "resampled weights have NaN/Inf")

            if carry_resampled_weights:
                new_weights = tf.maximum(new_weights, eps)
                new_weights = new_weights / tf.reduce_sum(new_weights)
                log_weights = tf.math.log(new_weights)
            else:
                # reset to uniform after equal-weight resamplers
                log_weights = -tf.math.log(tf.cast(Np, dtype)) * tf.ones([Np], dtype=dtype)

    return tf.stack(ests), tf.stack(ESSs), total_loglik
