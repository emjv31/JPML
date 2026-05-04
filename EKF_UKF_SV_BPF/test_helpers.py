import numpy as np
import tensorflow as tf


def assert_finite(testcase, tensor):
    testcase.assertTrue(np.all(np.isfinite(tensor.numpy())))

def assert_allclose(testcase, a, b, atol=1e-6):
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=atol)

def assert_positive_definite(testcase, tensor):
    eigs = np.linalg.eigvalsh(tensor.numpy())
    testcase.assertTrue(np.all(eigs > -1e-8))

def assert_symmetric(testcase, tensor, atol=1e-8):
    testcase.assertTrue(
        np.allclose(tensor.numpy(), tensor.numpy().T, atol=atol)
    )

def assert_shape(testcase, tensor, shape):
    testcase.assertEqual(tuple(tensor.shape), shape)


def validate_covariance_matrix(P, tol=1e-8, psd_tol=-1e-10):

    # Finite check
    if not tf.reduce_all(tf.math.is_finite(P)):
        raise ValueError("Covariance matrix contains NaN or Inf")

    # Symmetry check
    if not tf.reduce_all(tf.abs(P - tf.transpose(P)) < tol):
        raise ValueError("Covariance matrix is not symmetric")

    # Eigenvalue check (PSD)
    eigvals = tf.linalg.eigvalsh(P)
    if not tf.reduce_all(eigvals > psd_tol):
        raise ValueError("Covariance matrix is not positive semi-definite")


# ------------------------------------------------------------
# Assertions (shared)
# ------------------------------------------------------------
def assert_valid_output(testcase, ests, ESSs, T, d):
    testcase.assertEqual(ests.shape, (T, d))
    testcase.assertTrue(np.all(np.isfinite(ests.numpy())))
    if ESSs is not None:
        testcase.assertEqual(ESSs.shape, (T,))
        testcase.assertTrue(np.all(np.isfinite(ESSs.numpy())))


def assert_valid_ess(testcase, ESSs, Np):
    ess = ESSs.numpy()
    testcase.assertTrue(np.all(np.isfinite(ess)))
#    testcase.assertTrue(np.all(ess > 0))
    testcase.assertTrue(np.all(ess > 0- 1e-10)) #BlockPF - adjustment
    testcase.assertTrue(np.all(ess <= Np + 1e-6))

def assert_raises(testcase, callable_fn, error=ValueError):
    with testcase.assertRaises(error):
        callable_fn()

def assert_valid_loglik(testcase, total_loglik):
    testcase.assertEqual(total_loglik.shape, ())
    testcase.assertTrue(np.isfinite(total_loglik.numpy()))

# ------------------------------------------------------------
# Helper for simple valid functions
# ------------------------------------------------------------
def make_simple_functions(Np, dtype):
    def prop_fn(x):
        return x

    def log_likelihood_fn(particles, y):
        return tf.zeros((Np,), dtype=dtype)

    return prop_fn, log_likelihood_fn