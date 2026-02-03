import unittest
import tensorflow as tf
import numpy as np
import math

from filters_utils_q1 import KF_multivariate_robust, Sim_mLGSSM

class TestKFMultivariateRobust(tf.test.TestCase):

    # ---------------------------
    # Structure, shapes, finiteness over multiple dimensions
    # ---------------------------
    def test_structure_and_multiple_dimensions(self):
        # Setup 
        configs = [
            dict(n_x=1, n_y=1, T=5),
            dict(n_x=2, n_y=2, T=10),
            dict(n_x=3, n_y=2, T=7),
        ]
        
        for cfg in configs:
            n_x, n_y, T = cfg["n_x"], cfg["n_y"], cfg["T"]
            F = tf.eye(n_x, dtype=tf.float64)
            H = tf.random.normal([n_y, n_x], dtype=tf.float64)
            Q = tf.eye(n_x, dtype=tf.float64) * 0.1
            R = tf.eye(n_y, dtype=tf.float64) * 0.1
            m0 = tf.zeros([n_x], dtype=tf.float64)
            P0 = tf.eye(n_x, dtype=tf.float64)
            Y = tf.random.normal([n_y, T], dtype=tf.float64)
            out = KF_multivariate_robust(Y, F, H, Q, R, m0, P0)
            # Output variables
            expected = {
                "mu_pred_before", "mu_pred_next", "P_pred", "mu_filt",
                "P_filt", "P_filt_joseph", "v", "F_innov", "K",
                "loglik_vec", "loglik"
            }
            self.assertSetEqual(set(out.keys()), expected)
            # Check shapes
            self.assertEqual(out["mu_filt"].shape, (n_x, T))
            self.assertEqual(out["mu_pred_before"].shape, (n_x, T))
            self.assertEqual(out["P_filt"].shape, (T, n_x, n_x))
            self.assertEqual(out["P_filt_joseph"].shape, (T, n_x, n_x))
            self.assertEqual(out["P_pred"].shape, (T, n_x, n_x))
            self.assertEqual(out["v"].shape, (n_y, T))
            self.assertEqual(out["F_innov"].shape, (T, n_y, n_y))
            self.assertEqual(out["K"].shape, (T, n_x, n_y))
            self.assertEqual(out["loglik_vec"].shape, (T,))
            self.assertEqual(out["loglik"].shape, ())
            # Check finiteness
            for key, val in out.items():
                self.assertTrue(tf.reduce_all(tf.math.is_finite(val)).numpy(),
                                msg=f"{key} contains non-finite values")
            # Check states do not degenerate 
            mu_vals = out["mu_filt"].numpy()
            self.assertFalse(np.all(mu_vals == mu_vals[:, :1]),
                             msg="Filtered states degenerate")
    # -----------------------
    # Shape mismatch should raise
    def test_invalid_shapes_raise(self):
        Y = tf.random.normal([2, 5], dtype=tf.float64)
        F = tf.eye(3, dtype=tf.float64)  # wrong dim
        H = tf.eye(2, dtype=tf.float64)
        Q = tf.eye(3, dtype=tf.float64)
        R = tf.eye(2, dtype=tf.float64)
        m0 = tf.zeros([2], dtype=tf.float64)
        P0 = tf.eye(2, dtype=tf.float64)

        with self.assertRaises(AssertionError):
            KF_multivariate_robust(Y, F, H, Q, R, m0, P0)
    # ---------------------------
    # Non-PSD covariance should raise
    def test_non_psd_covariance_raises(self):

        Y = tf.random.normal([2, 5], dtype=tf.float64)
        F = tf.eye(2, dtype=tf.float64)
        H = tf.eye(2, dtype=tf.float64)
        Q = tf.constant([[1., 2.],
                         [2., -1.]], dtype=tf.float64)  # non-PSD
        R = tf.eye(2, dtype=tf.float64)
        m0 = tf.zeros([2], dtype=tf.float64)
        P0 = tf.eye(2, dtype=tf.float64)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            KF_multivariate_robust(Y, F, H, Q, R, m0, P0)
    # ---------------------------
    # Check error raise when Joseph's covariance matrix is not symmetric or positive semi-definite
    # ---------------------------
    def test_joseph_raises_errors(self):
        
        n_x, n_y, T = 2, 2, 5
        F = tf.eye(n_x, dtype=tf.float64)
        H = tf.eye(n_y, dtype=tf.float64)
        Q = tf.eye(n_x, dtype=tf.float64)
        R = tf.eye(n_y, dtype=tf.float64)
        m0 = tf.zeros(n_x, dtype=tf.float64)
        Y = tf.random.normal([n_y, T], dtype=tf.float64)
        # Setup
        cases = [
            # a. Non-finite covariance (NaN)
            dict(Q=tf.eye(n_x, dtype=tf.float64) * np.nan, P0=tf.eye(n_x, dtype=tf.float64)),
            # b. Non-PSD covariance
            dict(Q=Q, P0=tf.constant([[1., 2.], [2., -5.]], dtype=tf.float64)),
            # c. Asymmetric covariance
            dict(Q=Q, P0=tf.constant([[1., 1.], [0., 1.]], dtype=tf.float64)),
        ]
        for case in cases:
            with self.assertRaises(tf.errors.InvalidArgumentError):
                KF_multivariate_robust(Y, F, H, case["Q"], R, m0, case["P0"])



class TestSimMultiLGSSM(tf.test.TestCase):
    def test_mLGSSM_errors(self):
        n_x, n_y, T = 3, 2, 5
        # Input
        m0 = [0.0] * n_x
        F_mat = tf.eye(n_x, dtype=tf.float64)
        H_mat = tf.eye(n_y, n_x, dtype=tf.float64)
        Q_mat = tf.eye(n_x, dtype=tf.float64)
        R_mat = tf.eye(n_y, dtype=tf.float64)
        P0 = tf.eye(n_x, dtype=tf.float64)
        # Check output (types and shapes)
        X_true, Y_obs = Sim_mLGSSM(T, m0, F_mat, H_mat, Q_mat, R_mat, P0) 
        #self.assertIsInstance(X_true, tf.Tensor)
        #self.assertIsInstance(Y_obs, tf.Tensor)
        self.assertTrue(isinstance(X_true, (tf.Tensor, tf.Variable)))
        self.assertTrue(isinstance(Y_obs, (tf.Tensor, tf.Variable)))
        self.assertEqual(X_true.shape, (n_x, T))
        self.assertEqual(Y_obs.shape, (n_y, T))
        # --- Check raising errors for invalid inputs ---
        cases = [
            # Non-finite Q
            dict(Q_mat=tf.eye(n_x, dtype=tf.float64) * np.nan),
            # Non-PD Q
            dict(Q_mat=tf.constant([[1., 2., 0.], [2., -1., 0.], [0., 0., 1.]], dtype=tf.float64)),
            # F shape not valid
            dict(F_mat=tf.eye(n_x+1, dtype=tf.float64)),
            # H shape not valid
            dict(H_mat=tf.eye(n_y+1, n_x, dtype=tf.float64)),
            # Non-positive T
            dict(T=0),
        ]
        
        for case in cases:
            # vars for the simulation
            var_foo = dict(T=T, m0=m0, F_mat=F_mat, H_mat=H_mat, Q_mat=Q_mat, R_mat=R_mat, P0=P0)
            # replace any argument in cases with var_foo and check error raising
            var_foo.update(case) 
            with self.assertRaises(ValueError):
                Sim_mLGSSM(**var_foo)
