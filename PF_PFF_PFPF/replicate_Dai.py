import tensorflow as tf


def compute_H(x, sensors, R=None):
    """
    Compute the local measurement information matrix H(x)

    Input:
        x       : tf.Tensor (2,)
        sensors : tf.Tensor (Ns, 2)
        R       : tf.Tensor (Ns, Ns)

    Output:
        tf.Tensor (2, 2)

    Notes: 
        H(x) = J(x)^T R^{-1} J(x), where J is the Jacobian of the
        bearing measurement model. This is the local likelihood curvatureinformation term used in the stiffness objective.
    """
    if not isinstance(x, tf.Tensor):
        raise TypeError("x must be a tf.Tensor")
    if not isinstance(sensors, tf.Tensor):
        raise TypeError("sensors must be a tf.Tensor")

    x = tf.convert_to_tensor(x)
    sensors = tf.convert_to_tensor(sensors, dtype=x.dtype)

    if R is None:
        R = globals().get("R", None)

    if R is None:
        R = tf.eye(tf.shape(sensors)[0], dtype=x.dtype)
    else:
        R = tf.convert_to_tensor(R, dtype=x.dtype)

    if x.shape.rank != 1 or x.shape[0] != 2:
        raise ValueError("x must have shape (2,)")

    if sensors.shape.rank != 2 or sensors.shape[1] != 2:
        raise ValueError("sensors must have shape (Ns, 2)")

    Ns = sensors.shape[0]
    if R.shape.rank != 2 or R.shape[0] != Ns or R.shape[1] != Ns:
        raise ValueError(f"R must have shape ({Ns}, {Ns})")

    try:
        tf.linalg.cholesky(R)
    except Exception:
        raise ValueError("R must be positive definite")
    # Jacobian of bearing measurements wrt state
    dx = x[0] - sensors[:, 0]
    dy = x[1] - sensors[:, 1]
    den = dx**2 + dy**2

    if tf.reduce_any(den == 0):
        raise ValueError("Division by zero: sensor exactly at x")
    # Local information/Hessian contribution 
    Jmat = tf.stack([-dy / den, dx / den], axis=1)          # (Ns, 2)
    return tf.transpose(Jmat) @ tf.linalg.inv(R) @ Jmat


# --------------------
# Objective function 
# --------------------
def compute_J(beta, H, Q=None, mu=0.2, h=None, dtype=None):
    """
    Discrete stiffness objective for the beta schedule.

    Input:
        beta : tf.Tensor (N+1,) homotopy path with beta(0)=0, beta(1)=1
        H    : tf.Tensor (2, 2) measurement information matrix
        Q    : tf.Tensor (2, 2) prior covariance term
        mu   : scalar regularization weight

    Output:
        scalar tf.Tensor
    Notes:
        Evaluates the discretized cost used to optimize beta. The objective
        combines a smoothness penalty on beta_dot and a curvature/stiffness
        term based on M(beta) = Q^{-1} + beta H.
    """
    if not isinstance(beta, tf.Tensor):
        raise TypeError("beta must be tf.Tensor")
    if not isinstance(H, tf.Tensor):
        raise TypeError("H must be tf.Tensor")

    if beta.shape.rank != 1:
        raise ValueError("beta must be 1D")
    if beta.shape[0] is not None and beta.shape[0] < 2:
        raise ValueError("beta must have length >= 2")

    if H.shape.rank != 2 or H.shape[0] != 2 or H.shape[1] != 2:
        raise ValueError("H must be (2,2)")

    if dtype is None:
        dtype = beta.dtype

    
    beta = tf.cast(beta, dtype)
    H = tf.cast(H, dtype)

    if Q is None:
        Q = globals().get("Q", None)

    if Q is None:
        Q = tf.eye(2, dtype=dtype)
    else:
        if not isinstance(Q, tf.Tensor):
            raise TypeError("Q must be tf.Tensor")
        Q = tf.cast(Q, dtype)

    if Q.shape.rank != 2 or Q.shape[0] != 2 or Q.shape[1] != 2:
        raise ValueError("Q must be (2,2)")

    if h is None:
        h = globals().get("h", None)

    if h is None:
        h = tf.cast(1.0, dtype) / tf.cast(tf.shape(beta)[0] - 1, dtype)
    else:
        h = tf.cast(h, dtype)

    mu = tf.cast(mu, dtype)

    try:
        Q_inv = tf.linalg.inv(Q)
    except tf.errors.InvalidArgumentError:
        raise ValueError("Q must be invertible")
        
    # Discrete derivative and midpoint values of beta
    beta_dot = (beta[1:] - beta[:-1]) / h
    beta_mid = 0.5 * (beta[1:] + beta[:-1])

    M = Q_inv[None, :, :] + beta_mid[:, None, None] * H[None, :, :]

    # M(beta) on each interval midpoint; in the paper this captures posterior curvature
    try:
        Minv = tf.linalg.inv(M)
    except tf.errors.InvalidArgumentError:
        raise ValueError("Singular matrix in M")

    trace_M = tf.linalg.trace(M)
    trace_Minv = tf.linalg.trace(Minv)

    kappa_vals = mu * (
        trace_M * tf.linalg.trace(Minv @ H @ Minv)
        + tf.linalg.trace(H) * trace_Minv
    )

    integrand = 0.5 * beta_dot**2 + kappa_vals
    return tf.reduce_sum(integrand * h)


def optimize_beta(loss_fct, N, dtype=tf.float64, lr=0.03, num_iters=1200, verbose=False):
    """
    Optimize beta with fixed endpoints beta(0)=0, beta(1)=1.

    Input:
        loss_fct  : callable, objective as a function of beta
        N         : int, number of intervals
        lr        : float, optimizer step size
        num_iters : int, number of optimization iterations

    Output:
        beta_star : tf.Tensor (N+1,) optimized schedule
        J_star    : scalar tf.Tensor

    Notes:
        This gives a numerically optimized homotopy path from prior
        to posterior.
    """
    zero = tf.constant([0.0], dtype=dtype)
    one = tf.constant([1.0], dtype=dtype)

    beta_init = tf.linspace(
        tf.constant(0.0, dtype=dtype),
        tf.constant(1.0, dtype=dtype),
        N + 1,
    )[1:-1]

    beta_var = tf.Variable(beta_init)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for i in range(num_iters):
        with tf.GradientTape() as tape:
            beta_all = tf.concat([zero, beta_var, one], axis=0)
            J = loss_fct(beta_all)

        grads = tape.gradient(J, [beta_var])
        optimizer.apply_gradients(zip(grads, [beta_var]))

        if verbose and i % 300 == 0:
            tf.print("iter", i, "J =", J)

    beta_star = tf.concat([zero, beta_var, one], axis=0)
    J_star = loss_fct(beta_star)

    if verbose:
        tf.print("J star =", J_star)

    return beta_star, J_star


def compute_stiffness_ratio(beta, x, sensors, Q, R):
    beta = tf.convert_to_tensor(beta)
    dtype = beta.dtype

    x = tf.cast(x, dtype)
    sensors = tf.cast(sensors, dtype)
    Q = tf.cast(Q, dtype)
    R = tf.cast(R, dtype)

    H = compute_H(x=x, sensors=sensors, R=R)
    Q_inv = tf.linalg.inv(Q)

    N = tf.shape(beta)[0] - 1
    # dβ/dλ 
    beta_dot = (beta[1:] - beta[:-1]) * tf.cast(N, dtype)

    ratios = []

    for n in range(beta_dot.shape[0]):
        M = Q_inv + beta[n] * H
        J = -0.5 * beta_dot[n] * tf.linalg.solve(M, H)

        eigvals = tf.linalg.eigvals(J)
        eigvals = tf.abs(tf.math.real(eigvals))

        ratio = tf.reduce_max(eigvals) / (tf.reduce_min(eigvals) + tf.cast(1e-12, dtype))
        ratios.append(ratio)

    return tf.stack(ratios)


def compute_derivative_beta(beta):
    beta = tf.convert_to_tensor(beta)
    N = tf.shape(beta)[0] - 1
    return (beta[1:] - beta[:-1]) * tf.cast(N, beta.dtype)


def make_dai_example(dtype=tf.float64, N=50, mu_value=0.2):
    mu = tf.constant(mu_value, dtype)
    h = tf.constant(1.0 / N, dtype)

    sensors = tf.constant(
        [[3.5, 0.0],
         [-3.5, 0.0]],
        dtype=dtype,
    )

    x_true = tf.constant([4.0, 4.0], dtype)
    x_prior = tf.constant([3.0, 5.0], dtype)

    R = tf.constant(
        [[0.04, 0.0],
         [0.0, 0.04]],
        dtype=dtype,
    )

    Q = tf.constant(
        [[4.0, 0.0],
         [0.0, 0.4]],
        dtype=dtype,
    )

    beta_straight = tf.linspace(
        tf.constant(0.0, dtype),
        tf.constant(1.0, dtype),
        N + 1,
    )

    H = compute_H(x=x_true, sensors=sensors, R=R)

    return {
        "dtype": dtype,
        "N": N,
        "mu": mu,
        "h": h,
        "sensors": sensors,
        "x_true": x_true,
        "x_prior": x_prior,
        "R": R,
        "Q": Q,
        "H": H,
        "beta_straight": beta_straight,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cfg = make_dai_example()
    dtype = cfg["dtype"]
    N = cfg["N"]

    loss_J = lambda beta: compute_J(
        beta=beta,
        H=cfg["H"],
        Q=cfg["Q"],
        mu=cfg["mu"],
        h=cfg["h"],
        dtype=dtype,
    )

    J_straight = compute_J(
        beta=cfg["beta_straight"],
        H=cfg["H"],
        Q=cfg["Q"],
        mu=cfg["mu"],
        h=cfg["h"],
        dtype=dtype,
    )

    tf.print("J straight =", J_straight)

    beta_star, J_star = optimize_beta(
        loss_fct=loss_J,
        N=N,
        dtype=dtype,
        verbose=True,
    )

    lam = tf.linspace(
        tf.constant(0.0, dtype),
        tf.constant(1.0, dtype),
        N,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(
        lam.numpy(),
        compute_stiffness_ratio(
            beta=cfg["beta_straight"],
            x=cfg["x_prior"],
            sensors=cfg["sensors"],
            Q=cfg["Q"],
            R=cfg["R"],
        ).numpy(),
        label="β",
    )
    plt.plot(
        lam.numpy(),
        compute_stiffness_ratio(
            beta=beta_star,
            x=cfg["x_prior"],
            sensors=cfg["sensors"],
            Q=cfg["Q"],
            R=cfg["R"],
        ).numpy(),
        label="β*",
    )
    plt.xlabel("λ")
    plt.ylabel("Stiffness ratio")
    plt.grid(True)
    plt.legend()
    plt.savefig("beta_stiffness_ratio.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(cfg["beta_straight"].numpy(), cfg["beta_straight"].numpy(), label="β")
    plt.plot(cfg["beta_straight"].numpy(), beta_star.numpy(), label="β*")
    plt.xlabel("λ")
    plt.ylabel("β(λ)")
    plt.title("Optimal vs Straight β(λ)")
    plt.legend()
    plt.grid(True)
    plt.savefig("beta_vs_straight.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(
        cfg["beta_straight"].numpy(),
        (beta_star - cfg["beta_straight"]).numpy(),
        label="β* - β",
    )
    plt.xlabel("λ")
    plt.ylabel("β*(λ) - λ")
    plt.title("Deviation from Linear Homotopy")
    plt.legend()
    plt.grid(True)
    plt.savefig("beta_deviation.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(lam.numpy(), compute_derivative_beta(beta_star).numpy(), label="β*")
    plt.xlabel("λ")
    plt.ylabel("dβ*/dλ")
    plt.title("Derivative of Optimal β")
    plt.legend()
    plt.grid(True)
    plt.savefig("beta_derivative.png", dpi=300, bbox_inches="tight")
    plt.show()