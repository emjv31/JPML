import unittest

import tensorflow as tf 
import math
import numpy as np
import time

from filters_utils_q1b import SV_model_sim_tf_h
from filters_utils_q1b import compute_jacobian_tf, compute_sigma_points_tf, extensionKF_uni_tf_consistent
from filters_utils_q1b import SIR_bootstrap_markov_tf_stat, profile_tf, compute_bias_rmse


class TestSVModelSimTF(tf.test.TestCase):
    # ---------------------------
    # Output structure & non-degeneracy
    # ---------------------------
    def test_structure_and_non_degenerate_output(self):
        """Check that outputs have correct types, shapes, and are non-degenerate."""
        params_list = [
            dict(iT=10, phi=0.95, sigma_eta=0.2, sigma_eps=1.0, xi=1.0),
            dict(iT=15, phi=0.8, sigma_eta=0.1, sigma_eps=0.5, xi=2.0),
            dict(iT=5, phi=0.5, sigma_eta=0.3, sigma_eps=1.5, xi=0.5)
        ]

        for params in params_list:
            out = SV_model_sim_tf_h(**params)
            # Check dictionary
            self.assertIsInstance(out, dict)
            self.assertSetEqual(set(out.keys()), {"vY", "h"})
            # vY and h
            for key in ["vY", "h"]:
                val = out[key]
                self.assertIsInstance(val, tf.Tensor)
                self.assertEqual(val.dtype, tf.float64)
                self.assertEqual(val.shape[0], params["iT"])
                # non-degenerate: finite and not constant
                values = val.numpy()
                self.assertTrue(np.all(np.isfinite(values)), msg=f"{key} contains non-finite values")
                self.assertFalse(np.all(values == values[0]), msg=f"{key} degenerate: all values equal to {values[0]}")
    # ---------------------------
    # Sanity check
    # ---------------------------
    def test_valid_configs(self):
        """Run the simulation on valid configurations"""
        configs = [
            dict(iT=50, phi=0.95, sigma_eta=0.2, sigma_eps=1.0, xi=1.0),
            dict(iT=100, phi=0.8, sigma_eta=0.1, sigma_eps=0.5, xi=2.0),
            dict(iT=200, phi=0.5, sigma_eta=0.3, sigma_eps=1.5, xi=0.7)
        ]
        for cfg in configs:
            out = SV_model_sim_tf_h(**cfg)
            for key in ["vY", "h"]:
                values = out[key].numpy()
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertFalse(np.all(values == values[0]), msg=f"{key} degenerate output")
    # ---------------------------
    # Check invalid input
    # ---------------------------
    def test_invalid_inputs_raise(self):
        """Check that invalid numeric inputs raise errors."""
        bad_calls = [
            dict(iT=-10, phi=0.95, sigma_eta=0.2, sigma_eps=1.0, xi=1.0),
            dict(iT=500, phi=1.2, sigma_eta=0.2, sigma_eps=1.0, xi=1.0),
            dict(iT=500, phi=-2.1, sigma_eta=0.2, sigma_eps=1.0, xi=1.0),
            dict(iT=500, phi=0.95, sigma_eta=-0.2, sigma_eps=1.0, xi=1.0),
            dict(iT=500, phi=0.95, sigma_eta=0.2, sigma_eps=-1.0, xi=1.0),
            dict(iT=0, phi=0.95, sigma_eta=0.2, sigma_eps=1.0, xi=1.0),
        ]
        for kwargs in bad_calls:
            with self.assertRaises(ValueError):
                SV_model_sim_tf_h(**kwargs)

class TestJacobian(tf.test.TestCase):

    def test_scalar_output_and_dtype(self):
        f = lambda x: x**2
        x = tf.constant(2.0)
        J = compute_jacobian_tf(f, x)

        self.assertIsInstance(J, tf.Tensor)
        self.assertEqual(J.shape, ())
        self.assertEqual(J.dtype, tf.float64)

    def test_correct_value(self):
        f = lambda x: x**3
        x = tf.constant(2.0)
        J = compute_jacobian_tf(f, x)

        # Analytical derivative = 3x^2 = 12
        self.assertAllClose(J.numpy(), 12.0, atol=1e-4)

    def test_non_finite_input_raises(self):
        f = lambda x: x**2
        x = tf.constant(float('nan'))

        with self.assertRaises(ValueError):
            compute_jacobian_tf(f, x)

    def test_zero_derivative_raises(self):
        f = lambda x: tf.constant(1.0, dtype=tf.float64)
        x = tf.constant(2.0)

        with self.assertRaises(ValueError):
            compute_jacobian_tf(f, x)

class TestSigmaPointsTF(tf.test.TestCase):

    def test_weighted_mean_recovers_mu(self):
        """Check that the weighted mean of sigma points equals the input mean."""
        mu = 0.0
        P = 1.0
        sp = compute_sigma_points_tf(mu, P)
        X, Wm = sp["X"], sp["Wm"]

        mu_rec = tf.reduce_sum(Wm * X)
        self.assertAllClose(mu_rec, mu, atol=1e-10)

    def test_output_structure_and_types(self):
        """Check dictionary keys, tensor types, and dtypes."""
        sp = compute_sigma_points_tf(mu=0.0, P=1.0)
        self.assertIsInstance(sp, dict)
        self.assertSetEqual(set(sp.keys()), {"X", "Wm", "Wc"})

        X, Wm, Wc = sp["X"], sp["Wm"], sp["Wc"]
        self.assertIsInstance(X, tf.Tensor)
        self.assertIsInstance(Wm, tf.Tensor)
        self.assertIsInstance(Wc, tf.Tensor)

        self.assertEqual(X.dtype, tf.float64)
        self.assertEqual(Wm.dtype, tf.float64)
        self.assertEqual(Wc.dtype, tf.float64)

    def test_output_shapes(self):
        """Check output shapes (assuming univariate sigma points, n=1)."""
        sp = compute_sigma_points_tf(mu=0.0, P=1.0)
        X, Wm, Wc = sp["X"], sp["Wm"], sp["Wc"]

        self.assertEqual(X.shape, (3,))
        self.assertEqual(Wm.shape, (3,))
        self.assertEqual(Wc.shape, (3,))

    def test_all_outputs_finite(self):
        """Ensure all outputs are finite."""
        mu = 1.5
        P = 2.0
        sp = compute_sigma_points_tf(mu, P)
        X, Wm, Wc = sp["X"], sp["Wm"], sp["Wc"]

        # Check all elements are finite using tf.reduce_all
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X)).numpy())
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Wm)).numpy())
        self.assertTrue(tf.reduce_all(tf.math.is_finite(Wc)).numpy())

    def test_sigma_spread_positive(self):
        """Ensure sigma points spread and are not collapsing"""
        mu = tf.constant(1.0, dtype=tf.float64)
        P = tf.constant(2.0, dtype=tf.float64)
        sp = compute_sigma_points_tf(mu, P)
        X = sp["X"]

        spread_plus = X[1] - mu
        spread_minus = mu - X[2]

        self.assertGreater(spread_plus, 0, "Sigma points did not spread above mu")
        self.assertGreater(spread_minus, 0, "Sigma points did not spread below mu")
        self.assertAllClose(spread_plus, spread_minus, atol=1e-12, msg="Sigma points not symmetric")

    def test_weights_sum_and_symmetry(self):
        """Mean weights sum to 1 and off-center weights are symmetric"""
        sp = compute_sigma_points_tf(mu=0.0, P=1.0)
        Wm, Wc = sp["Wm"], sp["Wc"]

        self.assertAllClose(tf.reduce_sum(Wm), 1.0, atol=1e-12)
        self.assertAllClose(Wm[1], Wm[2], atol=1e-12)
        self.assertAllClose(Wc[1], Wc[2], atol=1e-12)

    def test_covariance_reconstruction(self):
        mu = tf.constant(0.5, dtype=tf.float64)
        P = tf.constant(1.5, dtype=tf.float64)
        sp = compute_sigma_points_tf(mu, P)
        X = tf.convert_to_tensor(sp["X"], dtype=tf.float64)
        Wc = tf.convert_to_tensor(sp["Wc"], dtype=tf.float64)
        reconstructed_P = tf.reduce_sum(Wc * (X - mu)**2)

    def test_small_P_degeneracy(self):
        """Very small P should raise a ValueError due to numerical collapse"""
        with self.assertRaises(ValueError):
            compute_sigma_points_tf(mu=1.0, P=1e-30)

    def test_invalid_P_raises(self):
        """Negative covariance should raise ValueError."""
        with self.assertRaises(ValueError):
            compute_sigma_points_tf(mu=0.0, P=-1.0)

    def test_nonfinite_mu_raises(self):
        """Non-finite mean should raise ValueError."""
        with self.assertRaises(ValueError):
            compute_sigma_points_tf(mu=tf.constant(np.nan, dtype=tf.float64), P=1.0)

class TestExtensionKFUniTF(tf.test.TestCase):
    # ---------------------------
    # Output structure and non-degeneracy
    # ---------------------------
    def test_structure_and_non_degenerate_states(self):
        """Check keys, shapes, dtypes, and non-degenerate states for EKF/UKF."""
        y = tf.constant([0.0, 0.2, 0.5, 1.0], dtype=tf.float64)
        # functions
        systems = [
            # linear
            (lambda x: 0.5*x + 0.1, lambda x: 2.0*x + 0.5),
            # non-linear
            (lambda x: 0.1 + 0.5*x + 0.2*tf.sin(x),
             lambda x: 0.5 + 0.5*x + 0.1*tf.cos(x))
        ]
        sigma_eta = sigma_e = tf.constant(0.1, dtype=tf.float64)
        m0 = tf.constant(0.0, dtype=tf.float64)
        P0 = tf.constant(1.0, dtype=tf.float64)

        for f_fun, h_fun in systems:
            for method in ["EKF", "UKF"]:
                out = extensionKF_uni_tf_consistent(
                    y, f_fun, h_fun, sigma_eta, sigma_e, m0, P0, method=method
                )
                # che ck output type
                self.assertIsInstance(out, dict)
                self.assertSetEqual(set(out.keys()), {"mu_filt", "P_filt", "mu_pred", "P_pred", "llk"})
                # variables
                for key in ["mu_filt", "P_filt", "mu_pred", "P_pred"]:
                    self.assertIsInstance(out[key], tf.Variable)
                    self.assertEqual(out[key].dtype, tf.float64)
                    self.assertEqual(out[key].shape, y.shape)
                    # non-degeneracy: finite and non-constant equal values
                    self.assertTrue(tf.reduce_all(tf.math.is_finite(out[key])).numpy())
                    values = out[key].numpy()
                    self.assertFalse(np.all(values == values[0]),
                                    msg=f"{key} degenerate: constant values equal to {values[0]}")
                    #if "P" in key:
                    #    self.assertTrue(tf.reduce_all(out[key] > 0).numpy())
                # likelihood
                self.assertIsInstance(out["llk"], tf.Variable)
                self.assertEqual(out["llk"].dtype, tf.float64)
                self.assertEqual(out["llk"].shape, ())
    # ---------------------------
    # Error raising
    # ---------------------------
    def test_invalid_method_error(self):
        y = [0.0, 1.0]
        f_fun = h_fun = lambda x: x
        sigma_eta = sigma_e = 0.1
        with self.assertRaises(ValueError):
            extensionKF_uni_tf_consistent(y, f_fun, h_fun, sigma_eta, sigma_e, method="m")
            
    def test_multivariate_input_error(self):
        """Check that passing multivariate time-series observations raises ValueError."""
        y_multi = tf.constant([[0.0, 0.1], [0.2, 0.3]], dtype=tf.float64)
        f_fun = lambda x: x
        h_fun = lambda x: x
        sigma_eta = sigma_e = tf.constant(0.1, dtype=tf.float64)
        m0 = tf.constant(0.0, dtype=tf.float64)
        P0 = tf.constant(1.0, dtype=tf.float64)
        
        for method in ["EKF", "UKF"]:
            with self.assertRaises(ValueError):
                extensionKF_uni_tf_consistent(y_multi, f_fun, h_fun, sigma_eta, sigma_e, m0, P0, method=method)                
                
    def test_ukf_degeneracy_error(self):
        """UKF degeneracy gives ValueError"""
        y = tf.constant([0.0, 0.0], dtype=tf.float64)
        # strong non-linearity
        f_fun = lambda x: tf.tanh(x) * 1e-12
        h_fun = lambda x: tf.tanh(x) * 1e-12
        sigma_eta = sigma_e = tf.constant(0.1, dtype=tf.float64)
        with self.assertRaises(ValueError):
            extensionKF_uni_tf_consistent(y, f_fun, h_fun, sigma_eta, sigma_e, method="UKF")
            
    def test_ekf_jacobian_zero_error(self):
        """A zero Jacobian yields EKF degeneracy and a ValueError"""
        y = tf.constant([0.0, 0.0], dtype=tf.float64)
        f_fun = lambda x: tf.sin(x)
        h_fun = lambda x: x**3  # derivative equal to x=0
        sigma_eta = sigma_e = tf.constant(0.1, dtype=tf.float64)
        with self.assertRaises(ValueError):
            extensionKF_uni_tf_consistent(y, f_fun, h_fun, sigma_eta, sigma_e, method="EKF")
            

class TestSIRBootstrapTF(tf.test.TestCase):
    # ---------------------------
    # Output structure & non-degeneracy
    # ---------------------------
    def test_structure_and_non_degenerate_output(self):
        """Check output keys, types, dtypes, shapes, and non-degenerate particle estimates."""
        Y = tf.constant([0.0, 0.1, 0.3, 0.6], dtype=tf.float64)
        configs = [
            {"N": 50, "phi": 0.5, "tau": 0.2, "sigma": 0.3, "squared": False},
            {"N": 100, "phi": 0.9, "tau": 0.1, "sigma": 0.2, "squared": True},
        ]

        for cfg in configs:
            out = SIR_bootstrap_markov_tf_stat(Y=Y, **cfg)
            # Check dictionary
            self.assertIsInstance(out, dict)
            self.assertSetEqual(set(out.keys()), {"part_est", "ESS", "loglikelihood"})
            # part_est and ESS
            for key in ["part_est", "ESS"]:
                self.assertIsInstance(out[key], tf.Variable)
                self.assertEqual(out[key].dtype, tf.float64)
                self.assertEqual(out[key].shape, Y.shape)

                vals = out[key].numpy()
                self.assertTrue(np.all(np.isfinite(vals)), msg=f"{key} contains non-finite values")
                self.assertFalse(np.all(vals == vals[0]),
                                 msg=f"{key} degenerate: all values equal to {vals[0]}")
            # loglikelihood
            self.assertIsInstance(out["loglikelihood"], tf.Tensor)
            self.assertEqual(out["loglikelihood"].dtype, tf.float64)
            self.assertEqual(out["loglikelihood"].shape, ())
    # ---------------------------
    # Check option squared 
    # ---------------------------
    def test_squared_mode_runs(self):
        """Check that squared=True runs correctly and produces finite, non-negative output."""
        Y = tf.constant([0.2, 0.0, -0.1, 0.4], dtype=tf.float64)
        out = SIR_bootstrap_markov_tf_stat(
            N=150,
            phi=0.8,
            tau=0.3,
            sigma=0.2,
            squared=True,
            Y=Y
        )
        vals = out["part_est"].numpy()
        self.assertTrue(np.all(np.isfinite(vals)))
        self.assertTrue(np.all(vals >= 0), msg="Squared estimates should be non-negative")
    # ---------------------------
    # Sanity check: correct behaviour
    # ---------------------------
    def test_sanity_multi(self):
        """Runs without error on different reasonable configurations."""
        Y = tf.constant([0.0, 0.2, 0.4, 0.1, -0.2], dtype=tf.float64)
        configs = [
            dict(N=100, phi=0.7, tau=0.2, sigma=0.3),
            dict(N=200, phi=0.95, tau=0.1, sigma=0.5, squared=True),
            dict(N=50, phi=0.5, tau=0.4, sigma=0.2),
        ]
        for cfg in configs:
            out = SIR_bootstrap_markov_tf_stat(Y=Y, **cfg)
            vals = out["part_est"].numpy()
            self.assertTrue(np.all(np.isfinite(vals)))
            self.assertFalse(np.all(vals == vals[0]), msg="Degenerate output")
    # ---------------------------
    # Error raising tests
    # ---------------------------
    def test_invalid_inputs_raise(self):
        """Function raises ValueError on invalid inputs."""
        Y = tf.constant([0.0, 0.1], dtype=tf.float64)
        bad_calls = [
            # N <= 0
            dict(N=0, phi=0.9, tau=0.2, sigma=0.3, Y=Y),
            # phi outside (-1, 1)
            dict(N=100, phi=-2.1, tau=0.2, sigma=0.3, Y=Y),
            dict(N=100, phi=1.2,  tau=0.2, sigma=0.3, Y=Y),
            # tau or sigma <= 0
            dict(N=100, phi=0.9, tau=-0.1, sigma=0.3, Y=Y),
            dict(N=100, phi=0.9, tau=0.2,  sigma=0.0, Y=Y),
            # Y invalid
            dict(N=100, phi=0.9, tau=0.2, sigma=0.3, Y=[1.0]),
            dict(N=100, phi=0.9, tau=0.2, sigma=0.3, Y=None),
        ]
    
        for kwargs in bad_calls:
            with self.assertRaises(ValueError):
                SIR_bootstrap_markov_tf_stat(**kwargs)
    # ---------------------------
    # Invalid squared argument
    # ---------------------------
    def test_invalid_squared_raises(self):
        Y = tf.constant([0.0, 0.1, 0.2], dtype=tf.float64)
        with self.assertRaises(ValueError):
            SIR_bootstrap_markov_tf_stat(N=100, phi=0.5, tau=0.2, sigma=0.3, squared="yes", Y=Y)
        with self.assertRaises(ValueError):
            SIR_bootstrap_markov_tf_stat(N=100, phi=0.5, tau=0.2, sigma=0.3, squared=None, Y=Y)

    def test_multivariate_Y_raises(self):
        """ Raising Error if Y is not 1D."""
        Y = tf.constant([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float64)

        with self.assertRaises(Exception):
            SIR_bootstrap_markov_tf_stat(
                N=100, phi=0.9, tau=0.2, sigma=0.3, Y=Y
            )

class TestComputeBiasRMSE(tf.test.TestCase):
    def test_output_structure(self):
        """Check bias and RMSE for 1D and 2D arrays are scalar tensors and finite"""
        test_cases = [
            # 1D arrays
            (tf.constant([1.0, 2.0, 3.0], dtype=tf.float64),
            tf.constant([1.1, 1.9, 3.2], dtype=tf.float64)),
            # 2D arrays
            (np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.1, 1.9], [2.8, 4.2]]))
        ]
        for x_true, x_est in test_cases:
            bias, rmse = compute_bias_rmse(x_true, x_est)
            # Check output type and shape
            self.assertIsInstance(bias, tf.Tensor)
            self.assertIsInstance(rmse, tf.Tensor)
            self.assertEqual(bias.shape, ())
            self.assertEqual(rmse.shape, ())
            # Check values are finite
            self.assertTrue(tf.reduce_all(tf.math.is_finite(bias)).numpy())
            self.assertTrue(tf.reduce_all(tf.math.is_finite(rmse)).numpy())
            self.assertGreaterEqual(rmse.numpy(), 0.0)

    def test_type(self):
        """Ensure function handles different data types."""
        x_true = np.array([1, 2, 3], dtype=np.float32)
        x_est = tf.constant([0.9, 2.1, 3.0], dtype=tf.float64)

        bias, rmse = compute_bias_rmse(x_true, x_est)
        self.assertEqual(bias.dtype, tf.float64)
        self.assertEqual(rmse.dtype, tf.float64)
        self.assertTrue(np.all(np.isfinite(bias.numpy())))
        self.assertTrue(np.all(np.isfinite(rmse.numpy())))
