import os
import io
import math
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from contextlib import redirect_stdout
from unittest.mock import patch

from test_helpers import *

from ekf_ukf_diagnostics import (
    compute_jacobian_tf,
    compute_sigma_points_tf,
    EKF_UKF_univariate,
    simulate_quadratic_model,
    make_quadratic_model_components,
    run_heatmap_analysis,
    plot_heatmaps_quadratic,
    print_summary_stats,
    plot_fraction_lollipops,
    find_bad_pair,
    plot_bad_acf,
    plot_bad_pacf,
)




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
        self.assertAllClose(reconstructed_P, P, atol=1e-12)

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



class TestEKFUKFUnivariate(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.dtype = tf.float64

    # --------------------------------------------------------
    # Shared validation helpers for this test file
    # --------------------------------------------------------
    def assert_valid_filter_output(self, out, T):
        self.assertIsInstance(out, dict)

        for key in ["mu_filt", "P_filt", "mu_pred", "P_pred", "llk"]:
            self.assertIn(key, out)

        assert_shape(self, out["mu_filt"], (T,))
        assert_shape(self, out["P_filt"], (T,))
        assert_shape(self, out["mu_pred"], (T,))
        assert_shape(self, out["P_pred"], (T,))

        assert_finite(self, out["mu_filt"])
        assert_finite(self, out["P_filt"])
        assert_finite(self, out["mu_pred"])
        assert_finite(self, out["P_pred"])
        assert_valid_loglik(self, out["llk"])

        self.assertTrue(np.all(out["P_filt"].numpy() > 0.0))
        self.assertTrue(np.all(out["P_pred"].numpy() > 0.0))

    def assert_valid_diagnostics(self, diag, T):
        for key in [
            "Jh", "K", "S", "v", "C_xy", "mu_y",
            "bad_geom_flag", "bad_sigma_flag", "bad_flag"
        ]:
            self.assertIn(key, diag)
            assert_shape(self, diag[key], (T,))

        for key in [
            "frac_bad_geom", "frac_bad_sigma", "frac_bad",
            "n_bad_geom", "n_bad_sigma", "n_bad"
        ]:
            self.assertIn(key, diag)
            self.assertEqual(diag[key].shape, ())

    # --------------------------------------------------------
    # Basic valid behavior
    # --------------------------------------------------------
    def test_basic_run_ekf(self):
        y = tf.constant([0.0, 0.5, -0.2, 0.1], dtype=self.dtype)

        f_fun = lambda x: tf.constant(0.9, dtype=self.dtype) * x
        h_fun = lambda x: tf.constant(1.2, dtype=self.dtype) * x
        jf_fun = lambda x: tf.constant(0.9, dtype=self.dtype)
        jh_fun = lambda x: tf.constant(1.2, dtype=self.dtype)

        out = EKF_UKF_univariate(
            y=y,
            f_fun=f_fun,
            h_fun=h_fun,
            sigma_eta=0.2,
            sigma_e=0.3,
            m0=0.1,
            P0=1.0,
            method="EKF",
            stop_on_failure=False,
            jf_fun=jf_fun,
            jh_fun=jh_fun
        )

        self.assert_valid_filter_output(out, T=4)

    def test_basic_run_ukf(self):
        y = tf.constant([0.0, 0.5, -0.2, 0.1], dtype=self.dtype)

        f_fun = lambda x: tf.constant(0.9, dtype=self.dtype) * x
        h_fun = lambda x: tf.constant(1.2, dtype=self.dtype) * x

        out = EKF_UKF_univariate(
            y=y,
            f_fun=f_fun,
            h_fun=h_fun,
            sigma_eta=0.2,
            sigma_e=0.3,
            m0=0.1,
            P0=1.0,
            method="UKF"
        )

        self.assert_valid_filter_output(out, T=4)

    def test_numpy_input_conversion(self):
        y = np.array([0.0, 1.0, 2.0], dtype=np.float64)

        f_fun = lambda x: tf.constant(0.9, dtype=self.dtype) * x
        h_fun = lambda x: tf.constant(1.2, dtype=self.dtype) * x
        jf_fun = lambda x: tf.constant(0.9, dtype=self.dtype)
        jh_fun = lambda x: tf.constant(1.2, dtype=self.dtype)

        out = EKF_UKF_univariate(
            y=y,
            f_fun=f_fun,
            h_fun=h_fun,
            sigma_eta=0.1,
            sigma_e=0.2,
            m0=1.0,
            method="EKF",
            stop_on_failure=False,
            jf_fun=jf_fun,
            jh_fun=jh_fun
        )

        self.assert_valid_filter_output(out, T=3)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    def test_return_diagnostics_ekf(self):
        y = tf.constant([0.0, 0.5, -0.2], dtype=self.dtype)

        f_fun = lambda x: tf.constant(0.9, dtype=self.dtype) * x
        h_fun = lambda x: tf.constant(1.2, dtype=self.dtype) * x
        jf_fun = lambda x: tf.constant(0.9, dtype=self.dtype)
        jh_fun = lambda x: tf.constant(1.2, dtype=self.dtype)

        out = EKF_UKF_univariate(
            y=y,
            f_fun=f_fun,
            h_fun=h_fun,
            sigma_eta=0.2,
            sigma_e=0.3,
            method="EKF",
            m0=1.0,
            return_diagnostics=True,
            stop_on_failure=False,
            jf_fun=jf_fun,
            jh_fun=jh_fun
        )

        self.assert_valid_filter_output(out, T=3)
        self.assertIn("diagnostics", out)

        diag = out["diagnostics"]
        self.assert_valid_diagnostics(diag, T=3)

        self.assertTrue(np.isnan(diag["C_xy"].numpy()[0]))
        self.assertTrue(np.isnan(diag["mu_y"].numpy()[0]))

    def test_return_diagnostics_ukf(self):
        y = tf.constant([0.0, 0.5, -0.2], dtype=self.dtype)

        f_fun = lambda x: tf.constant(0.9, dtype=self.dtype) * x
        h_fun = lambda x: tf.constant(1.2, dtype=self.dtype) * x

        out = EKF_UKF_univariate(
            y=y,
            f_fun=f_fun,
            h_fun=h_fun,
            sigma_eta=0.2,
            sigma_e=0.3,
            method="UKF",
            return_diagnostics=True
        )

        self.assert_valid_filter_output(out, T=3)
        self.assertIn("diagnostics", out)

        diag = out["diagnostics"]
        self.assert_valid_diagnostics(diag, T=3)

        self.assertTrue(np.all(np.isfinite(diag["mu_y"].numpy()[1:])))
        self.assertTrue(np.all(np.isfinite(diag["C_xy"].numpy()[1:])))

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------
    def test_invalid_method_raises(self):
        y = tf.constant([0.0, 1.0], dtype=self.dtype)
        f_fun = lambda x: x
        h_fun = lambda x: x

        assert_raises(
            self,
            lambda: EKF_UKF_univariate(
                y=y,
                f_fun=f_fun,
                h_fun=h_fun,
                sigma_eta=0.1,
                sigma_e=0.2,
                method="abc"
            ),
            error=ValueError
        )

    def test_invalid_y_shape_or_length_raises(self):
        f_fun = lambda x: x
        h_fun = lambda x: x
        jf_fun = lambda x: tf.constant(1.0, dtype=self.dtype)
        jh_fun = lambda x: tf.constant(1.0, dtype=self.dtype)

        bad_calls = [
            lambda: EKF_UKF_univariate(
                y=tf.constant([[1.0, 2.0]], dtype=self.dtype),
                f_fun=f_fun,
                h_fun=h_fun,
                sigma_eta=0.1,
                sigma_e=0.2,
                method="EKF",
                jf_fun=jf_fun,
                jh_fun=jh_fun
            ),
            lambda: EKF_UKF_univariate(
                y=tf.constant([1.0], dtype=self.dtype),
                f_fun=f_fun,
                h_fun=h_fun,
                sigma_eta=0.1,
                sigma_e=0.2,
                method="EKF",
                jf_fun=jf_fun,
                jh_fun=jh_fun
            ),
        ]

        for fn in bad_calls:
            with self.subTest():
                assert_raises(self, fn, error=ValueError)

    def test_nonfinite_y_raises(self):
        y = tf.constant([0.0, np.nan], dtype=self.dtype)
        f_fun = lambda x: x
        h_fun = lambda x: x
        jf_fun = lambda x: tf.constant(1.0, dtype=self.dtype)
        jh_fun = lambda x: tf.constant(1.0, dtype=self.dtype)

        assert_raises(
            self,
            lambda: EKF_UKF_univariate(
                y=y,
                f_fun=f_fun,
                h_fun=h_fun,
                sigma_eta=0.1,
                sigma_e=0.2,
                method="EKF",
                jf_fun=jf_fun,
                jh_fun=jh_fun
            ),
            error=ValueError
        )

    def test_nonfinite_scalar_inputs_raise(self):
        y = tf.constant([0.0, 1.0], dtype=self.dtype)
        f_fun = lambda x: x
        h_fun = lambda x: x

        bad_calls = [
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, np.nan, 0.2),
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, 0.1, np.nan),
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, 0.1, 0.2, m0=np.nan),
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, 0.1, 0.2, P0=np.nan),
        ]

        for fn in bad_calls:
            with self.subTest():
                assert_raises(self, fn, error=ValueError)

    def test_nonscalar_inputs_raise(self):
        y = tf.constant([0.0, 1.0], dtype=self.dtype)
        f_fun = lambda x: x
        h_fun = lambda x: x

        bad_calls = [
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, 0.1, 0.2, m0=[0.0]),
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, 0.1, 0.2, P0=[1.0]),
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, [0.1], 0.2),
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, 0.1, [0.2]),
        ]

        for fn in bad_calls:
            with self.subTest():
                assert_raises(self, fn, error=ValueError)

    def test_nonpositive_variances_raise(self):
        y = tf.constant([0.0, 1.0], dtype=self.dtype)
        f_fun = lambda x: x
        h_fun = lambda x: x

        bad_calls = [
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, 0.1, 0.2, P0=0.0),
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, 0.1, 0.2, P0=-1.0),
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, 0.0, 0.2),
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, -0.1, 0.2),
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, 0.1, 0.0),
            lambda: EKF_UKF_univariate(y, f_fun, h_fun, 0.1, -0.2),
        ]

        for fn in bad_calls:
            with self.subTest():
                assert_raises(self, fn, error=ValueError)

    def test_noncallable_functions_raise(self):
        y = tf.constant([0.0, 1.0], dtype=self.dtype)
        f_fun = lambda x: x
        h_fun = lambda x: x

        bad_calls = [
            lambda: EKF_UKF_univariate(
                y=y,
                f_fun=None,
                h_fun=h_fun,
                sigma_eta=0.1,
                sigma_e=0.2
            ),
            lambda: EKF_UKF_univariate(
                y=y,
                f_fun=f_fun,
                h_fun=None,
                sigma_eta=0.1,
                sigma_e=0.2
            ),
        ]

        for fn in bad_calls:
            with self.subTest():
                assert_raises(self, fn, error=ValueError)

    # --------------------------------------------------------
    # EKF-specific behavior
    # --------------------------------------------------------
    def test_ekf_bad_geometry_raises_when_stop_on_failure_true(self):
        y = tf.constant([0.0, 1.0, 2.0], dtype=self.dtype)

        f_fun = lambda x: x
        h_fun = lambda x: x
        jf_fun = lambda x: tf.constant(1.0, dtype=self.dtype)
        jh_fun = lambda x: tf.constant(1.0, dtype=self.dtype)

        with self.assertRaisesRegex(ValueError, "EKF bad geometry detected at t=1"):
            EKF_UKF_univariate(
                y=y,
                f_fun=f_fun,
                h_fun=h_fun,
                sigma_eta=0.1,
                sigma_e=0.2,
                m0=0.0,
                P0=1.0,
                method="EKF",
                stop_on_failure=True,
                x_tol=0.05,
                jf_fun=jf_fun,
                jh_fun=jh_fun
            )

    def test_ekf_bad_geometry_recorded_when_stop_on_failure_false(self):
        y = tf.constant([0.0, 1.0, 2.0], dtype=self.dtype)

        f_fun = lambda x: x
        h_fun = lambda x: x
        jf_fun = lambda x: tf.constant(1.0, dtype=self.dtype)
        jh_fun = lambda x: tf.constant(1.0, dtype=self.dtype)

        out = EKF_UKF_univariate(
            y=y,
            f_fun=f_fun,
            h_fun=h_fun,
            sigma_eta=0.1,
            sigma_e=0.2,
            m0=0.0,
            P0=1.0,
            method="EKF",
            stop_on_failure=False,
            return_diagnostics=True,
            x_tol=0.05,
            jf_fun=jf_fun,
            jh_fun=jh_fun
        )

        self.assert_valid_filter_output(out, T=3)

        diag = out["diagnostics"]
        self.assertEqual(int(diag["bad_geom_flag"][1].numpy()), 1)
        self.assertEqual(int(diag["bad_flag"][1].numpy()), 1)
        self.assertGreaterEqual(int(diag["n_bad_geom"].numpy()), 1)

    def test_explicit_jacobians_are_used_in_ekf(self):
        y = tf.constant([0.0, 0.1, 0.2], dtype=self.dtype)
        calls = {"jf": 0, "jh": 0}

        f_fun = lambda x: tf.constant(0.9, dtype=self.dtype) * x
        h_fun = lambda x: tf.constant(1.2, dtype=self.dtype) * x

        def jf_count(x):
            del x
            calls["jf"] += 1
            return tf.constant(0.9, dtype=self.dtype)

        def jh_count(x):
            del x
            calls["jh"] += 1
            return tf.constant(1.2, dtype=self.dtype)

        out = EKF_UKF_univariate(
            y=y,
            f_fun=f_fun,
            h_fun=h_fun,
            sigma_eta=0.1,
            sigma_e=0.2,
            m0=1.0,
            method="EKF",
            stop_on_failure=False,
            jf_fun=jf_count,
            jh_fun=jh_count
        )

        self.assert_valid_filter_output(out, T=3)
        self.assertEqual(calls["jf"], 2)
        self.assertEqual(calls["jh"], 2)

    # --------------------------------------------------------
    # UKF-specific behavior
    # --------------------------------------------------------
    def test_ukf_sigma_degeneracy_raises_when_stop_on_failure_true(self):
        y = tf.constant([0.0, 0.1, 0.2], dtype=self.dtype)

        f_fun = lambda x: x
        h_fun = lambda x: x**2

        with self.assertRaisesRegex(ValueError, "UKF sigma-point degeneracy detected at t=1"):
            EKF_UKF_univariate(
                y=y,
                f_fun=f_fun,
                h_fun=h_fun,
                sigma_eta=0.1,
                sigma_e=0.2,
                m0=0.0,
                P0=1.0,
                method="UKF",
                stop_on_failure=True,
                crosscov_tol=1e-10
            )

    def test_ukf_sigma_degeneracy_recorded_when_stop_on_failure_false(self):
        y = tf.constant([0.0, 0.1, 0.2], dtype=self.dtype)

        f_fun = lambda x: x
        h_fun = lambda x: x**2

        out = EKF_UKF_univariate(
            y=y,
            f_fun=f_fun,
            h_fun=h_fun,
            sigma_eta=0.1,
            sigma_e=0.2,
            m0=0.0,
            P0=1.0,
            method="UKF",
            stop_on_failure=False,
            return_diagnostics=True,
            crosscov_tol=1e-10
        )

        self.assert_valid_filter_output(out, T=3)

        diag = out["diagnostics"]
        self.assertEqual(int(diag["bad_sigma_flag"][1].numpy()), 1)
        self.assertEqual(int(diag["bad_flag"][1].numpy()), 1)
        self.assertGreaterEqual(int(diag["n_bad_sigma"].numpy()), 1)

    def test_ukf_bad_flag_is_union_of_geom_and_sigma_flags(self):
        y = tf.constant([0.0, 0.1, 0.2], dtype=self.dtype)

        f_fun = lambda x: x
        h_fun = lambda x: x**2

        out = EKF_UKF_univariate(
            y=y,
            f_fun=f_fun,
            h_fun=h_fun,
            sigma_eta=0.1,
            sigma_e=0.2,
            m0=0.0,
            P0=1.0,
            method="UKF",
            stop_on_failure=False,
            return_diagnostics=True,
            x_tol=0.05,
            crosscov_tol=1e-10
        )

        diag = out["diagnostics"]

        bad_geom = tf.cast(diag["bad_geom_flag"], tf.int32)
        bad_sigma = tf.cast(diag["bad_sigma_flag"], tf.int32)
        expected_bad = tf.cast(
            tf.logical_or(tf.cast(bad_geom, tf.bool), tf.cast(bad_sigma, tf.bool)),
            tf.int32
        )

        assert_allclose(
            self,
            tf.cast(diag["bad_flag"], self.dtype),
            tf.cast(expected_bad, self.dtype)
        )

    # --------------------------------------------------------
    # Internal failure paths
    # --------------------------------------------------------
    def test_nonfinite_ukf_prediction_raises(self):
        y = tf.constant([0.0, 1.0], dtype=self.dtype)

        def bad_f(x):
            del x
            return tf.constant(np.nan, dtype=self.dtype)

        h_fun = lambda x: tf.constant(1.2, dtype=self.dtype) * x

        with self.assertRaisesRegex(ValueError, "Non-finite prediction at t=1"):
            EKF_UKF_univariate(
                y=y,
                f_fun=bad_f,
                h_fun=h_fun,
                sigma_eta=0.1,
                sigma_e=0.2,
                method="UKF"
            )

    def test_nonfinite_ukf_observation_mapping_raises(self):
        y = tf.constant([0.0, 1.0], dtype=self.dtype)

        f_fun = lambda x: x

        def bad_h(x):
            del x
            return tf.constant(np.nan, dtype=self.dtype)

        with self.assertRaisesRegex(ValueError, "Non-finite observation mapping at t=1"):
            EKF_UKF_univariate(
                y=y,
                f_fun=f_fun,
                h_fun=bad_h,
                sigma_eta=0.1,
                sigma_e=0.2,
                method="UKF"
            )

    def test_invalid_predicted_variance_raises(self):
        y = tf.constant([0.0, 1.0], dtype=self.dtype)

        f_fun = lambda x: tf.constant(0.9, dtype=self.dtype) * x
        h_fun = lambda x: tf.constant(1.2, dtype=self.dtype) * x
        jh_fun = lambda x: tf.constant(1.2, dtype=self.dtype)

        def bad_jf(x):
            del x
            return tf.constant(np.nan, dtype=self.dtype)

        with self.assertRaisesRegex(ValueError, "Invalid P_pred at t=1"):
            EKF_UKF_univariate(
                y=y,
                f_fun=f_fun,
                h_fun=h_fun,
                sigma_eta=0.1,
                sigma_e=0.2,
                method="EKF",
                x_tol=0.0,
                jf_fun=bad_jf,
                jh_fun=jh_fun
            )

    def test_invalid_innovation_variance_raises(self):
        y = tf.constant([0.0, 1.0], dtype=self.dtype)

        f_fun = lambda x: tf.constant(0.9, dtype=self.dtype) * x
        h_fun = lambda x: tf.constant(1.2, dtype=self.dtype) * x
        jf_fun = lambda x: tf.constant(0.9, dtype=self.dtype)

        def bad_jh(x):
            del x
            return tf.constant(np.nan, dtype=self.dtype)

        with self.assertRaisesRegex(ValueError, "Invalid S_t at t=1"):
            EKF_UKF_univariate(
                y=y,
                f_fun=f_fun,
                h_fun=h_fun,
                sigma_eta=0.1,
                sigma_e=0.2,
                method="EKF",
                x_tol=0.0,
                jf_fun=jf_fun,
                jh_fun=bad_jh
            )

# Assumes the following are already imported / defined:
# - simulate_quadratic_model
# - make_quadratic_model_components
# - run_heatmap_analysis
# - EKF_UKF_univariate


class TestSimulateQuadraticModel(unittest.TestCase):
    """Tests for quadratic latent-state simulation output, shape, and reproducibility."""

    def setUp(self):
        self.dtype = tf.float64

    def test_output_structure_and_shape(self):
        """Checks simulator returns dict with finite x and y of shape (T,)."""
        T = 25
        out = simulate_quadratic_model(
            T=T,
            phi=0.9,
            sigma_eta=0.2,
            sigma_e=0.3,
            seed=42
        )

        self.assertIsInstance(out, dict)
        self.assertSetEqual(set(out.keys()), {"x", "y"})

        x, y = out["x"], out["y"]
        self.assertIsInstance(x, tf.Tensor)
        self.assertIsInstance(y, tf.Tensor)

        self.assertEqual(x.dtype, self.dtype)
        self.assertEqual(y.dtype, self.dtype)

        assert_shape(self, x, (T,))
        assert_shape(self, y, (T,))
        assert_finite(self, x)
        assert_finite(self, y)

    def test_same_seed_gives_same_output(self):
        """Checks simulator is reproducible for fixed seed and parameters."""
        kwargs = dict(T=30, phi=0.85, sigma_eta=0.2, sigma_e=0.4, seed=123)

        out1 = simulate_quadratic_model(**kwargs)
        out2 = simulate_quadratic_model(**kwargs)

        assert_allclose(self, out1["x"], out2["x"], atol=1e-12)
        assert_allclose(self, out1["y"], out2["y"], atol=1e-12)

    def test_different_seeds_give_different_output(self):
        """Checks changing the seed changes simulated latent or observed paths."""
        out1 = simulate_quadratic_model(T=30, phi=0.85, sigma_eta=0.2, sigma_e=0.4, seed=123)
        out2 = simulate_quadratic_model(T=30, phi=0.85, sigma_eta=0.2, sigma_e=0.4, seed=456)

        self.assertFalse(np.allclose(out1["x"].numpy(), out2["x"].numpy()))
        self.assertFalse(np.allclose(out1["y"].numpy(), out2["y"].numpy()))

    def test_nondegenerate_output(self):
        """Checks x and y are not constant on a typical valid configuration."""
        out = simulate_quadratic_model(
            T=40,
            phi=0.95,
            sigma_eta=0.2,
            sigma_e=0.3,
            seed=42
        )

        x = out["x"].numpy()
        y = out["y"].numpy()

        self.assertFalse(np.all(x == x[0]), msg="x is degenerate")
        self.assertFalse(np.all(y == y[0]), msg="y is degenerate")

    def test_T_equal_1_runs(self):
        """Checks simulator handles the minimal valid length T=1."""
        out = simulate_quadratic_model(
            T=1,
            phi=0.9,
            sigma_eta=0.2,
            sigma_e=0.3,
            seed=42
        )

        assert_shape(self, out["x"], (1,))
        assert_shape(self, out["y"], (1,))
        assert_finite(self, out["x"])
        assert_finite(self, out["y"])

    def test_invalid_T_raises(self):
        """Checks invalid T values raise an exception."""
        bad_calls = [
            lambda: simulate_quadratic_model(T=0, phi=0.9, sigma_eta=0.2, sigma_e=0.3),
            lambda: simulate_quadratic_model(T=-5, phi=0.9, sigma_eta=0.2, sigma_e=0.3),
            lambda: simulate_quadratic_model(T=2.5, phi=0.9, sigma_eta=0.2, sigma_e=0.3),
            lambda: simulate_quadratic_model(T="10", phi=0.9, sigma_eta=0.2, sigma_e=0.3),
        ]

        for fn in bad_calls:
            with self.subTest():
                with self.assertRaises(Exception):
                    fn()


class TestMakeQuadraticModelComponents(unittest.TestCase):
    """Tests for returned transition, observation, and derivative callables."""

    def setUp(self):
        self.dtype = tf.float64

    def test_output_structure(self):
        """Checks returned object is a dict with the expected callable entries."""
        comps = make_quadratic_model_components(phi=0.7)

        self.assertIsInstance(comps, dict)
        self.assertSetEqual(set(comps.keys()), {"f_fun", "h_fun", "jf_fun", "jh_fun"})

        for key in ["f_fun", "h_fun", "jf_fun", "jh_fun"]:
            self.assertTrue(callable(comps[key]), msg=f"{key} is not callable")

    def test_function_values_match_definition(self):
        """Checks component functions match the quadratic model formulas."""
        phi = 0.8
        x = tf.constant(2.0, dtype=self.dtype)

        comps = make_quadratic_model_components(phi=phi)

        f_val = comps["f_fun"](x)
        h_val = comps["h_fun"](x)
        jf_val = comps["jf_fun"](x)
        jh_val = comps["jh_fun"](x)

        expected_f = tf.constant(phi, dtype=self.dtype) * x
        expected_h = x**2 / 5.0
        expected_jf = tf.constant(phi, dtype=self.dtype)
        expected_jh = 2.0 * x / 5.0

        assert_allclose(self, f_val, expected_f, atol=1e-12)
        assert_allclose(self, h_val, expected_h, atol=1e-12)
        assert_allclose(self, jf_val, expected_jf, atol=1e-12)
        assert_allclose(self, jh_val, expected_jh, atol=1e-12)

    def test_jf_is_constant_in_x(self):
        """Checks jf_fun returns phi regardless of the input value."""
        phi = 0.65
        comps = make_quadratic_model_components(phi=phi)

        x1 = tf.constant(-3.0, dtype=self.dtype)
        x2 = tf.constant(4.5, dtype=self.dtype)

        jf1 = comps["jf_fun"](x1)
        jf2 = comps["jf_fun"](x2)
        expected = tf.constant(phi, dtype=self.dtype)

        assert_allclose(self, jf1, expected, atol=1e-12)
        assert_allclose(self, jf2, expected, atol=1e-12)

    def test_jh_changes_with_x(self):
        """Checks jh_fun depends on x according to 2x/5."""
        comps = make_quadratic_model_components(phi=0.9)

        x1 = tf.constant(1.0, dtype=self.dtype)
        x2 = tf.constant(3.0, dtype=self.dtype)

        jh1 = comps["jh_fun"](x1)
        jh2 = comps["jh_fun"](x2)

        self.assertNotEqual(float(jh1.numpy()), float(jh2.numpy()))
        assert_allclose(self, jh1, tf.constant(2.0 / 5.0, dtype=self.dtype), atol=1e-12)
        assert_allclose(self, jh2, tf.constant(6.0 / 5.0, dtype=self.dtype), atol=1e-12)


class TestRunHeatmapAnalysis(unittest.TestCase):
    """Tests for heatmap analysis table structure, diagnostics store, and consistency."""

    def setUp(self):
        self.dtype = tf.float32

    def test_output_types_and_sizes(self):
        """Checks analysis returns a DataFrame and diagnostics store with expected sizes."""
        phi_grid = [0.7, 0.9]
        sigma_e_grid = [0.2, 0.5]

        df, diagnostics_store = run_heatmap_analysis(
            simulator_fun=simulate_quadratic_model,
            phi_grid=phi_grid,
            sigma_e_grid=sigma_e_grid,
            sigma_eta=0.2,
            T=25,
            m0=0.5,
            P0=1.0,
            seed=42
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(diagnostics_store, dict)

        expected_n = len(phi_grid) * len(sigma_e_grid)
        self.assertEqual(len(df), expected_n)
        self.assertEqual(len(diagnostics_store), expected_n)

    def test_dataframe_columns_exist(self):
        """Checks the returned DataFrame contains the expected analysis columns."""
        df, _ = run_heatmap_analysis(
            simulator_fun=simulate_quadratic_model,
            phi_grid=[0.8],
            sigma_e_grid=[0.2, 0.4],
            sigma_eta=0.2,
            T=20,
            seed=42
        )

        expected_cols = {
            "phi",
            "sigma_e",
            "ekf_frac_bad_geom",
            "ukf_frac_bad_geom",
            "ukf_frac_bad_sigma",
            "ukf_frac_bad_total",
            "ekf_rmse",
            "ukf_rmse",
            "ekf_loglik",
            "ukf_loglik",
        }

        self.assertTrue(expected_cols.issubset(set(df.columns)))

    def test_dataframe_values_are_finite(self):
        """Checks all numeric analysis outputs in the DataFrame are finite."""
        df, _ = run_heatmap_analysis(
            simulator_fun=simulate_quadratic_model,
            phi_grid=[0.75, 0.9],
            sigma_e_grid=[0.2, 0.6],
            sigma_eta=0.2,
            T=20,
            seed=42
        )

        numeric_cols = [
            "phi",
            "sigma_e",
            "ekf_frac_bad_geom",
            "ukf_frac_bad_geom",
            "ukf_frac_bad_sigma",
            "ukf_frac_bad_total",
            "ekf_rmse",
            "ukf_rmse",
            "ekf_loglik",
            "ukf_loglik",
        ]

        self.assertTrue(np.isfinite(df[numeric_cols].to_numpy()).all())

    def test_diagnostics_store_structure(self):
        """Checks each diagnostics entry contains x, y, ekf, and ukf with valid tensors."""
        phi_grid = [0.8]
        sigma_e_grid = [0.3]

        _, diagnostics_store = run_heatmap_analysis(
            simulator_fun=simulate_quadratic_model,
            phi_grid=phi_grid,
            sigma_e_grid=sigma_e_grid,
            sigma_eta=0.2,
            T=15,
            seed=42
        )

        self.assertIn((0.8, 0.3), diagnostics_store)

        entry = diagnostics_store[(0.8, 0.3)]
        self.assertSetEqual(set(entry.keys()), {"x", "y", "ekf", "ukf"})

        x, y = entry["x"], entry["y"]
        assert_shape(self, x, (15,))
        assert_shape(self, y, (15,))
        assert_finite(self, x)
        assert_finite(self, y)

        for method_key in ["ekf", "ukf"]:
            res = entry[method_key]
            self.assertIsInstance(res, dict)
            self.assertIn("mu_filt", res)
            self.assertIn("P_filt", res)
            self.assertIn("mu_pred", res)
            self.assertIn("P_pred", res)
            self.assertIn("llk", res)
            self.assertIn("diagnostics", res)

            assert_shape(self, res["mu_filt"], (15,))
            assert_shape(self, res["P_filt"], (15,))
            assert_shape(self, res["mu_pred"], (15,))
            assert_shape(self, res["P_pred"], (15,))
            assert_finite(self, res["mu_filt"])
            assert_finite(self, res["P_filt"])
            assert_finite(self, res["mu_pred"])
            assert_finite(self, res["P_pred"])

    def test_store_keys_match_grid(self):
        """Checks diagnostics store keys exactly match the parameter grid pairs."""
        phi_grid = [0.6, 0.8]
        sigma_e_grid = [0.1, 0.3, 0.5]

        _, diagnostics_store = run_heatmap_analysis(
            simulator_fun=simulate_quadratic_model,
            phi_grid=phi_grid,
            sigma_e_grid=sigma_e_grid,
            sigma_eta=0.2,
            T=10,
            seed=42
        )

        expected_keys = {(phi, sigma_e) for phi in phi_grid for sigma_e in sigma_e_grid}
        self.assertSetEqual(set(diagnostics_store.keys()), expected_keys)

    def test_dataframe_and_store_have_matching_pairs(self):
        """Checks every row in the DataFrame corresponds to an entry in the diagnostics store."""
        phi_grid = [0.7, 0.9]
        sigma_e_grid = [0.2, 0.4]

        df, diagnostics_store = run_heatmap_analysis(
            simulator_fun=simulate_quadratic_model,
            phi_grid=phi_grid,
            sigma_e_grid=sigma_e_grid,
            sigma_eta=0.2,
            T=12,
            seed=42
        )

        df_pairs = set(zip(df["phi"], df["sigma_e"]))
        store_pairs = set(diagnostics_store.keys())

        self.assertSetEqual(df_pairs, store_pairs)

    def test_fraction_columns_are_in_unit_interval(self):
        """Checks bad-event fractions lie between 0 and 1."""
        df, _ = run_heatmap_analysis(
            simulator_fun=simulate_quadratic_model,
            phi_grid=[0.8, 0.9],
            sigma_e_grid=[0.2, 0.5],
            sigma_eta=0.2,
            T=20,
            seed=42
        )

        frac_cols = [
            "ekf_frac_bad_geom",
            "ukf_frac_bad_geom",
            "ukf_frac_bad_sigma",
            "ukf_frac_bad_total",
        ]

        for col in frac_cols:
            vals = df[col].to_numpy()
            self.assertTrue(np.all(vals >= 0.0))
            self.assertTrue(np.all(vals <= 1.0))

    def test_rmse_columns_are_nonnegative(self):
        """Checks RMSE columns are finite and nonnegative."""
        df, _ = run_heatmap_analysis(
            simulator_fun=simulate_quadratic_model,
            phi_grid=[0.85],
            sigma_e_grid=[0.2, 0.6],
            sigma_eta=0.2,
            T=20,
            seed=42
        )

        self.assertTrue(np.all(df["ekf_rmse"].to_numpy() >= 0.0))
        self.assertTrue(np.all(df["ukf_rmse"].to_numpy() >= 0.0))
        self.assertTrue(np.isfinite(df["ekf_rmse"].to_numpy()).all())
        self.assertTrue(np.isfinite(df["ukf_rmse"].to_numpy()).all())

    def test_reproducibility_with_same_seed(self):
        """Checks heatmap analysis is reproducible with the same seed."""
        kwargs = dict(
            simulator_fun=simulate_quadratic_model,
            phi_grid=[0.8, 0.9],
            sigma_e_grid=[0.2, 0.4],
            sigma_eta=0.2,
            T=20,
            m0=0.5,
            P0=1.0,
            seed=42
        )

        df1, store1 = run_heatmap_analysis(**kwargs)
        df2, store2 = run_heatmap_analysis(**kwargs)

        pd.testing.assert_frame_equal(
            df1.sort_values(["phi", "sigma_e"]).reset_index(drop=True),
            df2.sort_values(["phi", "sigma_e"]).reset_index(drop=True)
        )

        for key in store1:
            assert_allclose(self, store1[key]["x"], store2[key]["x"], atol=1e-12)
            assert_allclose(self, store1[key]["y"], store2[key]["y"], atol=1e-12)
            assert_allclose(self, store1[key]["ekf"]["mu_filt"], store2[key]["ekf"]["mu_filt"], atol=1e-12)
            assert_allclose(self, store1[key]["ukf"]["mu_filt"], store2[key]["ukf"]["mu_filt"], atol=1e-12)



class TestPrintSummaryStats(unittest.TestCase):
    """Tests that summary statistics printing selects and displays the expected columns."""

    def setUp(self):
        self.df = pd.DataFrame([
            {
                "phi": 0.7, "sigma_e": 0.2,
                "ekf_frac_bad_geom": 0.10,
                "ukf_frac_bad_geom": 0.05,
                "ukf_frac_bad_sigma": 0.20,
                "ukf_frac_bad_total": 0.25,
                "ekf_rmse": 0.80,
                "ukf_rmse": 0.70,
                "ekf_loglik": -10.0,
                "ukf_loglik": -9.0,
            },
            {
                "phi": 0.9, "sigma_e": 0.5,
                "ekf_frac_bad_geom": 0.22,
                "ukf_frac_bad_geom": 0.10,
                "ukf_frac_bad_sigma": 0.40,
                "ukf_frac_bad_total": 0.45,
                "ekf_rmse": 1.20,
                "ukf_rmse": 1.05,
                "ekf_loglik": -16.0,
                "ukf_loglik": -15.0,
            },
        ])

    def test_prints_summary_table(self):
        """Checks the summary function prints descriptive statistics for the expected columns."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_summary_stats(self.df)

        output = buf.getvalue()

        self.assertIn("ekf_frac_bad_geom", output)
        self.assertIn("ukf_frac_bad_sigma", output)
        self.assertIn("ekf_rmse", output)
        self.assertIn("ukf_loglik", output)
        self.assertIn("count", output)
        self.assertIn("mean", output)
        self.assertIn("std", output)

    def test_missing_required_column_raises(self):
        """Checks the summary function fails when a required metric column is absent."""
        bad_df = self.df.drop(columns=["ukf_rmse"])

        with self.assertRaises(KeyError):
            print_summary_stats(bad_df)


class TestFindBadPair(unittest.TestCase):
    """Tests pair lookup by approximate floating-point match in the diagnostics store."""

    def setUp(self):
        self.store = {
            (0.7, 0.2): {"dummy": 1},
            (0.7, 0.5): {"dummy": 2},
            (0.9, 0.2): {"dummy": 3},
        }

    def test_exact_match_returns_key(self):
        """Checks an exact pair lookup returns the stored key."""
        key = find_bad_pair(self.store, phi_target=0.7, sigma_e_target=0.2)
        self.assertEqual(key, (0.7, 0.2))

    def test_approximate_match_within_tolerance_returns_key(self):
        """Checks a numerically close pair is matched when inside the tolerance."""
        key = find_bad_pair(
            self.store,
            phi_target=0.70000000001,
            sigma_e_target=0.20000000001,
            tol=1e-8
        )
        self.assertEqual(key, (0.7, 0.2))

    def test_no_match_raises_keyerror(self):
        """Checks lookup raises KeyError when no pair is close enough."""
        with self.assertRaises(KeyError) as cm:
            find_bad_pair(self.store, phi_target=0.8, sigma_e_target=0.3)

        msg = str(cm.exception)
        self.assertIn("No bad pair found", msg)
        self.assertIn("Available pairs", msg)

    def test_tolerance_too_small_raises_keyerror(self):
        """Checks a near pair does not match when tolerance is too strict."""
        with self.assertRaises(KeyError):
            find_bad_pair(
                self.store,
                phi_target=0.7000001,
                sigma_e_target=0.2000001,
                tol=1e-12
            )


class BaseBadCorrPlotMixin:
    """Shared tests for bad-indicator correlation plotting functions."""

    plot_fun = None
    ekf_title = None
    ukf_title = None
    ylabel = None
    used_msg = None

    def setUp(self):
        self.diagnostics_store = {
            (0.7, 0.2): {
                "ekf": {"diagnostics": {"bad_geom_flag": tf.constant([0, 1, 0, 1, 0, 0, 1, 0], dtype=tf.int32)}},
                "ukf": {"diagnostics": {"bad_sigma_flag": tf.constant([0, 0, 1, 1, 0, 1, 0, 0], dtype=tf.int32)}},
            },
            (0.9, 0.5): {
                "ekf": {"diagnostics": {"bad_geom_flag": tf.constant([1, 1, 0, 0, 1, 0, 0, 1], dtype=tf.int32)}},
                "ukf": {"diagnostics": {"bad_sigma_flag": tf.constant([0, 1, 0, 1, 0, 1, 0, 1], dtype=tf.int32)}},
            },
        }

    def tearDown(self):
        plt.close("all")

    def _call_plot(self, **kwargs):
        params = dict(
            diagnostics_store=self.diagnostics_store,
            selected_configs=[(0.7, 0.2)],
            nlags=3,
        )
        params.update(kwargs)
        return self.plot_fun(**params)

    def _main_axes(self, fig):
        return [ax for ax in fig.axes if ax.get_title() != ""]

    def test_plot_runs_and_creates_two_axes(self):
        with patch("matplotlib.pyplot.show"):
            self._call_plot(selected_configs=[(0.7, 0.2), (0.9, 0.5)])
        fig = plt.gcf()
        self.assertEqual(len(self._main_axes(fig)), 2)

    def test_axes_titles_and_labels_are_correct(self):
        with patch("matplotlib.pyplot.show"):
            self._call_plot()
        fig = plt.gcf()
        main_axes = self._main_axes(fig)
        self.assertEqual(main_axes[0].get_title(), self.ekf_title)
        self.assertEqual(main_axes[1].get_title(), self.ukf_title)
        for ax in main_axes:
            self.assertEqual(ax.get_xlabel(), "Lag")
            self.assertEqual(ax.get_ylabel(), self.ylabel)

    def test_suptitle_exists(self):
        with patch("matplotlib.pyplot.show"):
            self._call_plot()
        fig = plt.gcf()
        self.assertIsNotNone(fig._suptitle)
        self.assertTrue(len(fig._suptitle.get_text()) > 0)

    def test_prints_available_and_used_configurations(self):
        buf = io.StringIO()
        with redirect_stdout(buf), patch("matplotlib.pyplot.show"):
            self._call_plot()
        output = buf.getvalue()
        self.assertIn("Available parameter combinations:", output)
        self.assertIn("(0.7, 0.2)", output)
        self.assertIn("(0.9, 0.5)", output)
        self.assertIn(self.used_msg, output)

    def test_prints_bad_counts_and_fractions(self):
        buf = io.StringIO()
        with redirect_stdout(buf), patch("matplotlib.pyplot.show"):
            self._call_plot()
        output = buf.getvalue()
        self.assertIn("EKF: number of bad-geometry time points", output)
        self.assertIn("EKF: fraction in bad geometry", output)
        self.assertIn("UKF: number of sigma-point degeneracy time points", output)
        self.assertIn("UKF: fraction in sigma-point degeneracy", output)

    def test_unavailable_selected_config_is_skipped_if_others_exist(self):
        buf = io.StringIO()
        with redirect_stdout(buf), patch("matplotlib.pyplot.show"):
            self._call_plot(selected_configs=[(0.1, 0.1), (0.7, 0.2)])
        output = buf.getvalue()
        self.assertIn("Skipping (0.1, 0.1): not available in diagnostics_store", output)
        self.assertIn(self.used_msg, output)
        self.assertIn("(0.7, 0.2)", output)

    def test_no_selected_configs_available_raises(self):
        with patch("matplotlib.pyplot.show"):
            with self.assertRaisesRegex(ValueError, "None of the selected configurations are available."):
                self._call_plot(selected_configs=[(0.1, 0.1), (0.2, 0.3)])

    def test_legends_are_present_for_both_axes(self):
        with patch("matplotlib.pyplot.show"):
            self._call_plot(selected_configs=[(0.7, 0.2), (0.9, 0.5)])
        fig = plt.gcf()
        main_axes = self._main_axes(fig)
        self.assertIsNotNone(main_axes[0].get_legend())
        self.assertIsNotNone(main_axes[1].get_legend())

    def test_selected_configs_drive_number_of_legend_entries(self):
        with patch("matplotlib.pyplot.show"):
            self._call_plot(selected_configs=[(0.7, 0.2), (0.9, 0.5)])
        fig = plt.gcf()
        main_axes = self._main_axes(fig)
        for ax in main_axes:
            labels = [txt.get_text() for txt in ax.get_legend().get_texts()]
            self.assertEqual(len(labels), 2)
            self.assertTrue(any("0.7" in label and "0.2" in label for label in labels))
            self.assertTrue(any("0.9" in label and "0.5" in label for label in labels))

    def test_nlags_zero_runs(self):
        with patch("matplotlib.pyplot.show"):
            self._call_plot(nlags=0)
        fig = plt.gcf()
        self.assertEqual(len(self._main_axes(fig)), 2)


class TestPlotBadACF(BaseBadCorrPlotMixin, unittest.TestCase):
    """Tests ACF plotting for EKF and UKF bad-event indicators."""

    plot_fun = staticmethod(plot_bad_acf)
    ekf_title = "ACF of EKF bad-geometry indicator"
    ukf_title = "ACF of UKF sigma-point degeneracy indicator"
    ylabel = "ACF"
    used_msg = "Configurations used in the ACF plot:"


class TestPlotBadPACF(BaseBadCorrPlotMixin, unittest.TestCase):
    """Tests PACF plotting for EKF and UKF bad-event indicators."""

    plot_fun = staticmethod(plot_bad_pacf)
    ekf_title = "PACF of the EKF bad-geometry indicator"
    ukf_title = "PACF of UKF sigma-point degeneracy indicator"
    ylabel = "PACF Value"
    used_msg = "Configurations used in the PACF plot:"

    def test_returns_figure_and_axes(self):
        with patch("matplotlib.pyplot.show"):
            fig, axes = self._call_plot()
        self.assertIsNotNone(fig)
        self.assertIsNotNone(axes)
        self.assertEqual(len(axes), 2)

    def test_save_true_writes_figure(self):
        tmpdir = tempfile.mkdtemp()
        try:
            with patch("matplotlib.pyplot.show"):
                fig, axes = self._call_plot(
                    save=True,
                    save_dir=tmpdir,
                    filename="test_bad_pacf.png",
                    dpi=150,
                )
            save_path = os.path.join(tmpdir, "test_bad_pacf.png")
            self.assertTrue(os.path.exists(save_path))
            self.assertTrue(os.path.getsize(save_path) > 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_save_true_prints_save_path(self):
        tmpdir = tempfile.mkdtemp()
        try:
            buf = io.StringIO()
            with redirect_stdout(buf), patch("matplotlib.pyplot.show"):
                self._call_plot(
                    save=True,
                    save_dir=tmpdir,
                    filename="test_bad_pacf.png",
                )
            output = buf.getvalue()
            self.assertIn("Saved figure to:", output)
            self.assertIn("test_bad_pacf.png", output)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_close_after_save_closes_figure(self):
        tmpdir = tempfile.mkdtemp()
        try:
            with patch("matplotlib.pyplot.show"):
                fig, axes = self._call_plot(
                    save=True,
                    save_dir=tmpdir,
                    filename="test_bad_pacf.png",
                    close_after_save=True,
                )
            self.assertFalse(plt.fignum_exists(fig.number))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_close_after_save_false_keeps_figure_open(self):
        tmpdir = tempfile.mkdtemp()
        try:
            with patch("matplotlib.pyplot.show"):
                fig, axes = self._call_plot(
                    save=True,
                    save_dir=tmpdir,
                    filename="test_bad_pacf.png",
                    close_after_save=False,
                )
            self.assertTrue(plt.fignum_exists(fig.number))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            plt.close("all")



class TestQuadraticHeatmapAnalysisIntegration(unittest.TestCase):
    """Small real integration test for the EKF/UKF graphical analysis pipeline."""

    @classmethod
    def setUpClass(cls):
        # Keep this small so the integration test stays fast
        cls.phi_grid = [0.3, 0.9]
        cls.sigma_e_grid = [0.05, 0.5]

        cls.df_heat, cls.diagnostics_store = run_heatmap_analysis(
            simulator_fun=simulate_quadratic_model,
            phi_grid=cls.phi_grid,
            sigma_e_grid=cls.sigma_e_grid,
            sigma_eta=0.2,
            T=30,
            m0=0.5,
            P0=1.0,
            x_tol=0.05,
            crosscov_tol=1e-10,
            seed=42,
        )

    @classmethod
    def tearDownClass(cls):
        plt.close("all")

    def test_dataframe_and_store_have_expected_size(self):
        """Checks the real pipeline returns one result per grid pair."""
        expected_n = len(self.phi_grid) * len(self.sigma_e_grid)

        self.assertIsInstance(self.df_heat, pd.DataFrame)
        self.assertEqual(len(self.df_heat), expected_n)
        self.assertEqual(len(self.diagnostics_store), expected_n)

    def test_dataframe_has_expected_columns_and_finite_values(self):
        """Checks the summary DataFrame has the expected columns and finite numeric values."""
        expected_cols = {
            "phi",
            "sigma_e",
            "ekf_frac_bad_geom",
            "ukf_frac_bad_geom",
            "ukf_frac_bad_sigma",
            "ukf_frac_bad_total",
            "ekf_rmse",
            "ukf_rmse",
            "ekf_loglik",
            "ukf_loglik",
        }

        self.assertTrue(expected_cols.issubset(set(self.df_heat.columns)))

        numeric = self.df_heat[list(expected_cols)].to_numpy()
        self.assertTrue(np.all(np.isfinite(numeric)))

    def test_diagnostics_store_has_expected_structure(self):
        """Checks each diagnostics entry contains simulation output and both filters."""
        for key, out in self.diagnostics_store.items():
            with self.subTest(cfg=key):
                self.assertIn("x", out)
                self.assertIn("y", out)
                self.assertIn("ekf", out)
                self.assertIn("ukf", out)

                x = out["x"]
                y = out["y"]
                ekf = out["ekf"]
                ukf = out["ukf"]

                self.assertIsInstance(x, tf.Tensor)
                self.assertIsInstance(y, tf.Tensor)
                self.assertTrue(np.all(np.isfinite(x.numpy())))
                self.assertTrue(np.all(np.isfinite(y.numpy())))

                for filt in [ekf, ukf]:
                    self.assertIn("mu_filt", filt)
                    self.assertIn("P_filt", filt)
                    self.assertIn("llk", filt)
                    self.assertIn("diagnostics", filt)

                    self.assertTrue(np.all(np.isfinite(filt["mu_filt"].numpy())))
                    self.assertTrue(np.all(np.isfinite(filt["P_filt"].numpy())))
                    self.assertTrue(np.isfinite(filt["llk"].numpy()))

    def test_find_bad_pair_returns_existing_key(self):
        """Checks exact existing parameter pairs are recovered from diagnostics_store."""
        key = find_bad_pair(self.diagnostics_store, 0.3, 0.05)
        self.assertEqual(key, (0.3, 0.05))

    def test_find_bad_pair_missing_raises(self):
        """Checks missing parameter pairs raise KeyError."""
        with self.assertRaises(KeyError):
            find_bad_pair(self.diagnostics_store, 999.0, 999.0)


class TestQuadraticPlotsIntegration(unittest.TestCase):
    """Runs plotting functions on real small analysis outputs."""

    @classmethod
    def setUpClass(cls):
        cls.df_heat, cls.diagnostics_store = run_heatmap_analysis(
            simulator_fun=simulate_quadratic_model,
            phi_grid=[0.3, 0.9],
            sigma_e_grid=[0.05, 0.5],
            sigma_eta=0.2,
            T=30,
            m0=0.5,
            P0=1.0,
            x_tol=0.05,
            crosscov_tol=1e-10,
            seed=42,
        )

    def tearDown(self):
        plt.close("all")

    def test_summary_and_dataframe_plots_run(self):
        """Checks summary/heatmap plots run on real pipeline output."""
        print_summary_stats(self.df_heat)
        plot_fraction_lollipops(self.df_heat)
        plot_heatmaps_quadratic(self.df_heat)

    def test_acf_plot_runs_on_real_selected_configs(self):
        """Checks ACF plot runs on real diagnostics output."""
        selected = [(0.3, 0.05), (0.9, 0.05), (0.9, 0.5)]
        plot_bad_acf(self.diagnostics_store, selected_configs=selected, nlags=10)

    def test_pacf_plot_runs_on_real_selected_configs(self):
        """Checks PACF plot runs on real diagnostics output."""
        selected = [(0.3, 0.05), (0.9, 0.05), (0.9, 0.5)]
        fig, axes = plot_bad_pacf(
            self.diagnostics_store,
            selected_configs=selected,
            nlags=10,
            save=False,
            close_after_save=False,
        )

        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 2)


if __name__ == "__main__":
    unittest.main()