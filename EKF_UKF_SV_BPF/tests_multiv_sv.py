import unittest
import io
import os
import math
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from unittest.mock import patch, DEFAULT
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings("ignore")

from test_helpers import *

from multiv_sv_bpf_core import (
    _infer_dim_from_params,
    _make_covariance_and_chol_tf,
    _make_scale_vector_tf,
    _as_vector_param_tf,
    _as_transition_matrix_tf,
    _stationary_covariance_ar1,
    _as_noise_chol_tf,
    SV_model_sim_tf_h,
    make_prop_sv,
    make_loglik_sv,
    multinomial_resampling,
    bpf_generic_resampling,
)

from multiv_ekf_ukf_core import (
    unscented_sigma_points_batch,
    unscented_sigma_points,
    unscented_transform,
    ukf_predict,
    ukf_update_check,
    _filter_core,
    make_ukf_kernels,
    make_ekf_kernels,
)

from sv_experiments import (
    compute_bias_rmse,
    benchmark_cpu,
    run_kf_experiments_multivariate,
    _run_bpf_mc_for_fixed_N,
    compare_methods_one_config,
    plot_benchmark_grouped,
    plot_ESS_heatmap_over_d,
    plot_ESS_over_time_bpf,
    plot_metric_over_time_algorithms,
    plot_rmse_over_dimension,
    build_ESS_summary_dataframe,
)



##### UNIT TESTS UTILS

class TestInferDimFromParamsTF(unittest.TestCase):
    """Tests dimension inference from scalar, vector, and matrix parameters."""

    def test_explicit_d_is_returned(self):
        """Checks explicit valid d overrides parameter inspection."""
        out = _infer_dim_from_params(
            phi=0.9,
            sigma_eta=0.2,
            sigma_eps=0.3,
            xi=1.0,
            d=4
        )
        self.assertEqual(out, 4)

    def test_invalid_explicit_d_raises(self):
        """Checks invalid explicit d raises ValueError."""
        for d in [0, -1, 2.5, "3"]:
            with self.subTest(d=d):
                with self.assertRaises(ValueError):
                    _infer_dim_from_params(0.9, 0.2, 0.3, 1.0, d=d)

    def test_all_scalars_default_to_one(self):
        """Checks all-scalar inputs imply univariate dimension 1."""
        out = _infer_dim_from_params(0.9, 0.2, 0.3, 1.0)
        self.assertEqual(out, 1)

    def test_infers_dimension_from_vector(self):
        """Checks a vector parameter determines the inferred dimension."""
        out = _infer_dim_from_params(
            phi=[0.8, 0.7],
            sigma_eta=0.2,
            sigma_eps=0.3,
            xi=1.0
        )
        self.assertEqual(out, 2)

    def test_infers_dimension_from_square_matrix(self):
        """Checks a square matrix parameter determines the inferred dimension."""
        out = _infer_dim_from_params(
            phi=[[0.8, 0.0], [0.0, 0.7]],
            sigma_eta=0.2,
            sigma_eps=0.3,
            xi=1.0
        )
        self.assertEqual(out, 2)

    def test_non_square_matrix_raises(self):
        """Checks non-square matrix parameters raise ValueError."""
        with self.assertRaises(ValueError):
            _infer_dim_from_params(
                phi=[[0.8, 0.0, 0.1], [0.0, 0.7, 0.2]],
                sigma_eta=0.2,
                sigma_eps=0.3,
                xi=1.0
            )

    def test_rank_greater_than_two_raises(self):
        """Checks rank-3 or higher parameters raise ValueError."""
        with self.assertRaises(ValueError):
            _infer_dim_from_params(
                phi=tf.ones((2, 2, 2)),
                sigma_eta=0.2,
                sigma_eps=0.3,
                xi=1.0
            )

    def test_incompatible_dimensions_raise(self):
        """Checks incompatible inferred dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            _infer_dim_from_params(
                phi=[0.8, 0.7],
                sigma_eta=[0.2, 0.3, 0.4],
                sigma_eps=0.3,
                xi=1.0
            )


class TestAsVectorParamTF(unittest.TestCase):
    """Tests vector parameter expansion and validation."""

    def setUp(self):
        self.dtype = tf.float64

    def test_scalar_expands_to_length_d_vector(self):
        """Checks a scalar is expanded to a constant vector of length d."""
        out = _as_vector_param_tf(2.0, d=3, dtype=self.dtype, name="x")
        expected = tf.constant([2.0, 2.0, 2.0], dtype=self.dtype)

        assert_shape(self, out, (3,))
        assert_allclose(self, out, expected, atol=1e-12)

    def test_vector_is_returned_unchanged(self):
        """Checks a valid length-d vector is returned unchanged."""
        x = tf.constant([1.0, 2.0], dtype=self.dtype)
        out = _as_vector_param_tf(x, d=2, dtype=self.dtype, name="x")

        assert_shape(self, out, (2,))
        assert_allclose(self, out, x, atol=1e-12)

    def test_wrong_length_vector_raises(self):
        """Checks a vector with the wrong length raises InvalidArgumentError."""
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _as_vector_param_tf([1.0, 2.0], d=3, dtype=self.dtype, name="x")

    def test_rank_greater_than_one_raises(self):
        """Checks only scalar or rank-1 inputs are accepted."""
        with self.assertRaises(ValueError):
            _as_vector_param_tf([[1.0, 2.0], [3.0, 4.0]], d=2, dtype=self.dtype, name="x")


class TestMakeScaleVectorTF(unittest.TestCase):
    """Tests positive scale-vector construction from scalar or vector xi."""

    def setUp(self):
        self.dtype = tf.float64

    def test_scalar_xi_returns_constant_vector(self):
        """Checks scalar xi is expanded to a positive length-d vector."""
        out = _make_scale_vector_tf(2.0, d=3, dtype=self.dtype)
        expected = tf.constant([2.0, 2.0, 2.0], dtype=self.dtype)

        assert_shape(self, out, (3,))
        assert_allclose(self, out, expected, atol=1e-12)

    def test_vector_xi_returns_same_vector(self):
        """Checks a valid xi vector is returned unchanged."""
        xi = tf.constant([1.0, 2.0], dtype=self.dtype)
        out = _make_scale_vector_tf(xi, d=2, dtype=self.dtype)

        assert_shape(self, out, (2,))
        assert_allclose(self, out, xi, atol=1e-12)

    def test_scalar_xi_nonpositive_raises(self):
        """Checks scalar xi must be strictly positive."""
        for val in [0.0, -1.0]:
            with self.subTest(val=val):
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    _make_scale_vector_tf(val, d=2, dtype=self.dtype)

    def test_vector_xi_nonpositive_entry_raises(self):
        """Checks all xi vector entries must be strictly positive."""
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _make_scale_vector_tf([1.0, 0.0], d=2, dtype=self.dtype)

    def test_vector_xi_wrong_length_raises(self):
        """Checks xi vector must have length d."""
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _make_scale_vector_tf([1.0, 2.0], d=3, dtype=self.dtype)

    def test_rank_greater_than_one_raises(self):
        """Checks xi must be scalar or vector only."""
        with self.assertRaises(ValueError):
            _make_scale_vector_tf([[1.0, 2.0], [3.0, 4.0]], d=2, dtype=self.dtype)


class TestAsTransitionMatrixTF(unittest.TestCase):
    """Tests transition matrix construction from scalar, vector, or matrix phi."""

    def setUp(self):
        self.dtype = tf.float64

    def test_scalar_phi_returns_scaled_identity(self):
        """Checks scalar phi produces phi times the identity matrix."""
        out = _as_transition_matrix_tf(0.8, d=3, dtype=self.dtype)
        expected = tf.eye(3, dtype=self.dtype) * tf.constant(0.8, dtype=self.dtype)

        assert_shape(self, out, (3, 3))
        assert_allclose(self, out, expected, atol=1e-12)

    def test_scalar_phi_with_abs_ge_one_raises(self):
        """Checks unstable scalar phi raises InvalidArgumentError."""
        for phi in [1.0, -1.0, 1.2, -1.2]:
            with self.subTest(phi=phi):
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    _as_transition_matrix_tf(phi, d=2, dtype=self.dtype)

    def test_vector_phi_returns_diagonal_matrix(self):
        """Checks vector phi produces a diagonal transition matrix."""
        phi = tf.constant([0.8, 0.6], dtype=self.dtype)
        out = _as_transition_matrix_tf(phi, d=2, dtype=self.dtype)
        expected = tf.linalg.diag(phi)

        assert_shape(self, out, (2, 2))
        assert_allclose(self, out, expected, atol=1e-12)

    def test_vector_phi_wrong_length_raises(self):
        """Checks phi vector must have length d."""
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _as_transition_matrix_tf([0.8, 0.6], d=3, dtype=self.dtype)

    def test_vector_phi_with_unstable_entry_raises(self):
        """Checks all phi vector entries must satisfy |phi_i| < 1."""
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _as_transition_matrix_tf([0.8, 1.0], d=2, dtype=self.dtype)

    def test_matrix_phi_returns_matrix_when_stable(self):
        """Checks a stable phi matrix is returned unchanged."""
        phi = tf.constant([[0.5, 0.1], [0.0, 0.6]], dtype=self.dtype)
        out = _as_transition_matrix_tf(phi, d=2, dtype=self.dtype)

        assert_shape(self, out, (2, 2))
        assert_allclose(self, out, phi, atol=1e-12)

    def test_matrix_phi_wrong_shape_raises(self):
        """Checks phi matrix must have shape (d,d)."""
        phi = tf.constant([[0.5, 0.1, 0.2], [0.0, 0.6, 0.1]], dtype=self.dtype)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _as_transition_matrix_tf(phi, d=2, dtype=self.dtype)

    def test_matrix_phi_unstable_raises(self):
        """Checks a phi matrix with spectral radius >= 1 raises InvalidArgumentError."""
        phi = tf.constant([[1.1, 0.0], [0.0, 0.5]], dtype=self.dtype)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _as_transition_matrix_tf(phi, d=2, dtype=self.dtype)

    def test_rank_greater_than_two_raises(self):
        """Checks phi must be scalar, vector, or matrix only."""
        with self.assertRaises(ValueError):
            _as_transition_matrix_tf(tf.ones((2, 2, 2), dtype=self.dtype), d=2, dtype=self.dtype)


class TestMakeCovarianceAndCholTF(unittest.TestCase):
    """Tests covariance/cholesky construction from scalar, vector, or matrix inputs."""

    def setUp(self):
        self.dtype = tf.float64

    def test_scalar_sigma_returns_isotropic_covariance_and_chol(self):
        """Checks scalar std produces isotropic covariance and matching cholesky."""
        Q, chol = _make_covariance_and_chol_tf(2.0, d=3, dtype=self.dtype, name="sigma_eta")

        expected_Q = tf.eye(3, dtype=self.dtype) * 4.0
        expected_chol = tf.eye(3, dtype=self.dtype) * 2.0

        assert_shape(self, Q, (3, 3))
        assert_shape(self, chol, (3, 3))
        assert_allclose(self, Q, expected_Q, atol=1e-12)
        assert_allclose(self, chol, expected_chol, atol=1e-12)
        assert_symmetric(self, Q)
        assert_positive_definite(self, Q)

    def test_scalar_sigma_nonpositive_raises(self):
        """Checks scalar std must be positive."""
        for val in [0.0, -1.0]:
            with self.subTest(val=val):
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    _make_covariance_and_chol_tf(val, d=2, dtype=self.dtype, name="sigma_eta")

    def test_vector_sigma_returns_diagonal_covariance_and_chol(self):
        """Checks vector stds produce diagonal covariance and diagonal cholesky."""
        sigma = tf.constant([2.0, 3.0], dtype=self.dtype)
        Q, chol = _make_covariance_and_chol_tf(sigma, d=2, dtype=self.dtype, name="sigma_eps")

        expected_Q = tf.linalg.diag(tf.constant([4.0, 9.0], dtype=self.dtype))
        expected_chol = tf.linalg.diag(sigma)

        assert_shape(self, Q, (2, 2))
        assert_shape(self, chol, (2, 2))
        assert_allclose(self, Q, expected_Q, atol=1e-12)
        assert_allclose(self, chol, expected_chol, atol=1e-12)
        assert_symmetric(self, Q)
        assert_positive_definite(self, Q)

    def test_vector_sigma_wrong_length_raises(self):
        """Checks sigma vector must have length d."""
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _make_covariance_and_chol_tf([1.0, 2.0], d=3, dtype=self.dtype, name="sigma_eta")

    def test_vector_sigma_nonpositive_entry_raises(self):
        """Checks all vector std entries must be positive."""
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _make_covariance_and_chol_tf([1.0, 0.0], d=2, dtype=self.dtype, name="sigma_eta")

    def test_matrix_sigma_returns_matrix_and_cholesky(self):
        """Checks covariance matrix input is returned with its cholesky factor."""
        Sigma = tf.constant([[2.0, 0.5], [0.5, 1.5]], dtype=self.dtype)
        Q, chol = _make_covariance_and_chol_tf(Sigma, d=2, dtype=self.dtype, name="sigma_eps")

        assert_shape(self, Q, (2, 2))
        assert_shape(self, chol, (2, 2))
        assert_allclose(self, Q, Sigma, atol=1e-12)
        assert_symmetric(self, Q)
        assert_positive_definite(self, Q)
        validate_covariance_matrix(Q)

    def test_matrix_sigma_wrong_shape_raises(self):
        """Checks covariance matrix input must have shape (d,d)."""
        Sigma = tf.ones((2, 3), dtype=self.dtype)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _make_covariance_and_chol_tf(Sigma, d=2, dtype=self.dtype, name="sigma_eta")

    def test_non_psd_matrix_sigma_fails(self):
        """Checks non-positive-definite covariance input fails or yields non-finite cholesky."""
        Sigma = tf.constant([[1.0, 2.0], [2.0, 1.0]], dtype=self.dtype)

        try:
            _, chol = _make_covariance_and_chol_tf(Sigma, d=2, dtype=self.dtype, name="sigma_eta")
            self.assertFalse(np.all(np.isfinite(chol.numpy())))
        except tf.errors.InvalidArgumentError:
            pass

    def test_rank_greater_than_two_raises(self):
        """Checks sigma input with rank greater than 2 raises ValueError."""
        with self.assertRaises(ValueError):
            _make_covariance_and_chol_tf(tf.ones((2, 2, 2), dtype=self.dtype), d=2, dtype=self.dtype, name="sigma_eta")


class TestAsNoiseCholTF(unittest.TestCase):
    """Tests direct Cholesky-factor construction from scalar, vector, or covariance inputs."""

    def setUp(self):
        self.dtype = tf.float64

    def test_scalar_sigma_returns_scaled_identity(self):
        """Checks scalar std produces a scaled identity cholesky factor."""
        L = _as_noise_chol_tf(2.0, d=3, dtype=self.dtype, name="sigma_eta")
        expected = tf.eye(3, dtype=self.dtype) * 2.0

        assert_shape(self, L, (3, 3))
        assert_allclose(self, L, expected, atol=1e-12)

    def test_scalar_sigma_nonpositive_raises(self):
        """Checks scalar std must be positive."""
        for val in [0.0, -1.0]:
            with self.subTest(val=val):
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    _as_noise_chol_tf(val, d=2, dtype=self.dtype, name="sigma_eta")

    def test_vector_sigma_returns_diagonal_cholesky(self):
        """Checks vector stds produce a diagonal cholesky factor."""
        sigma = tf.constant([2.0, 3.0], dtype=self.dtype)
        L = _as_noise_chol_tf(sigma, d=2, dtype=self.dtype, name="sigma_eps")
        expected = tf.linalg.diag(sigma)

        assert_shape(self, L, (2, 2))
        assert_allclose(self, L, expected, atol=1e-12)

    def test_vector_sigma_wrong_length_raises(self):
        """Checks sigma vector must have length d."""
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _as_noise_chol_tf([1.0, 2.0], d=3, dtype=self.dtype, name="sigma_eta")

    def test_vector_sigma_nonpositive_entry_raises(self):
        """Checks all vector std entries must be positive."""
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _as_noise_chol_tf([1.0, 0.0], d=2, dtype=self.dtype, name="sigma_eta")

    def test_matrix_sigma_returns_cholesky(self):
        """Checks covariance matrix input returns its cholesky factor."""
        Sigma = tf.constant([[2.0, 0.5], [0.5, 1.5]], dtype=self.dtype)
        L = _as_noise_chol_tf(Sigma, d=2, dtype=self.dtype, name="sigma_eps")

        reconstructed = L @ tf.transpose(L)
        assert_shape(self, L, (2, 2))
        assert_allclose(self, reconstructed, Sigma, atol=1e-10)

    def test_matrix_sigma_wrong_shape_raises(self):
        """Checks covariance matrix input must have shape (d,d)."""
        Sigma = tf.ones((2, 3), dtype=self.dtype)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            _as_noise_chol_tf(Sigma, d=2, dtype=self.dtype, name="sigma_eta")

    def test_non_psd_matrix_sigma_fails(self):
        """Checks non-positive-definite covariance input fails or yields non-finite cholesky."""
        Sigma = tf.constant([[1.0, 2.0], [2.0, 1.0]], dtype=self.dtype)

        try:
            L = _as_noise_chol_tf(Sigma, d=2, dtype=self.dtype, name="sigma_eta")
            self.assertFalse(np.all(np.isfinite(L.numpy())))
        except tf.errors.InvalidArgumentError:
            pass

    def test_rank_greater_than_two_raises(self):
        """Checks sigma input with rank greater than 2 raises ValueError."""
        with self.assertRaises(ValueError):
            _as_noise_chol_tf(tf.ones((2, 2, 2), dtype=self.dtype), d=2, dtype=self.dtype, name="sigma_eta")


class TestStationaryCovarianceAR1TF(unittest.TestCase):
    """Tests stationary covariance computation for AR(1) state dynamics."""

    def setUp(self):
        self.dtype = tf.float64

    def test_univariate_matches_closed_form(self):
        """Checks the univariate stationary covariance matches Q / (1 - phi^2)."""
        Phi = tf.constant([[0.8]], dtype=self.dtype)
        Q = tf.constant([[4.0]], dtype=self.dtype)

        P = _stationary_covariance_ar1(Phi, Q, self.dtype)
        expected = tf.constant([[4.0 / (1.0 - 0.8**2)]], dtype=self.dtype)

        assert_shape(self, P, (1, 1))
        assert_allclose(self, P, expected, atol=1e-10)
        assert_symmetric(self, P)
        assert_positive_definite(self, P)

    def test_diagonal_case_matches_closed_form(self):
        """Checks diagonal Phi gives diagonal stationary covariance with closed-form entries."""
        Phi = tf.constant([[0.5, 0.0], [0.0, 0.2]], dtype=self.dtype)
        Q = tf.constant([[4.0, 0.0], [0.0, 9.0]], dtype=self.dtype)

        P = _stationary_covariance_ar1(Phi, Q, self.dtype)
        expected = tf.constant([
            [4.0 / (1.0 - 0.5**2), 0.0],
            [0.0, 9.0 / (1.0 - 0.2**2)]
        ], dtype=self.dtype)

        assert_shape(self, P, (2, 2))
        assert_allclose(self, P, expected, atol=1e-10)
        assert_symmetric(self, P)
        assert_positive_definite(self, P)
        validate_covariance_matrix(P)

    def test_output_is_symmetric_and_finite(self):
        """Checks the returned stationary covariance is finite and symmetric."""
        Phi = tf.constant([[0.5, 0.1], [0.0, 0.6]], dtype=self.dtype)
        Q = tf.constant([[2.0, 0.3], [0.3, 1.5]], dtype=self.dtype)

        P = _stationary_covariance_ar1(Phi, Q, self.dtype)

        assert_shape(self, P, (2, 2))
        assert_finite(self, P)
        assert_symmetric(self, P)
        validate_covariance_matrix(P)

    def test_solution_satisfies_discrete_lyapunov_equation(self):
        """Checks P solves P = Phi P Phi' + Q."""
        Phi = tf.constant([[0.4, 0.1], [0.0, 0.5]], dtype=self.dtype)
        Q = tf.constant([[1.0, 0.2], [0.2, 0.8]], dtype=self.dtype)

        P = _stationary_covariance_ar1(Phi, Q, self.dtype)
        rhs = Phi @ P @ tf.transpose(Phi) + Q

        assert_allclose(self, P, rhs, atol=1e-8)

    def test_unstable_system_may_fail_solve(self):
        """Checks singular I - Phi⊗Phi can fail for unit-root systems."""
        Phi = tf.constant([[1.0]], dtype=self.dtype)
        Q = tf.constant([[1.0]], dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            _stationary_covariance_ar1(Phi, Q, self.dtype)


####### TEST SV MODEL
#############

class TestSVModelSimTFH(tf.test.TestCase):
    """Tests output structure, reproducibility, and input validation for the Gaussian SV simulator."""

    # --------------------------------------------------------
    # Basic output structure
    # --------------------------------------------------------
    def test_univariate_structure_and_finite_output(self):
        """Checks univariate outputs have correct shape, dtype, and finite values."""
        configs = [
            dict(iT=10, phi=0.95, sigma_eta=0.2, sigma_eps=1.0, xi=1.0, dtype=tf.float32),
            dict(iT=15, phi=0.80, sigma_eta=0.1, sigma_eps=0.5, xi=2.0, dtype=tf.float64),
            dict(iT=5,  phi=0.50, sigma_eta=0.3, sigma_eps=1.5, xi=0.5, dtype=tf.float32),
        ]

        for cfg in configs:
            out = SV_model_sim_tf_h(**cfg)

            self.assertIsInstance(out, dict)
            self.assertSetEqual(set(out.keys()), {"vY", "h"})

            vY, h = out["vY"], out["h"]

            self.assertIsInstance(vY, tf.Tensor)
            self.assertIsInstance(h, tf.Tensor)
            self.assertEqual(vY.dtype, cfg["dtype"])
            self.assertEqual(h.dtype, cfg["dtype"])

            assert_shape(self, vY, (cfg["iT"],))
            assert_shape(self, h, (cfg["iT"],))
            assert_finite(self, vY)
            assert_finite(self, h)

            self.assertFalse(np.all(vY.numpy() == vY.numpy()[0]), msg="vY is degenerate")
            self.assertFalse(np.all(h.numpy() == h.numpy()[0]), msg="h is degenerate")

    def test_multivariate_structure_and_finite_output(self):
        """Checks multivariate outputs have correct shape, dtype, and finite values."""
        configs = [
            dict(iT=12, phi=0.95, sigma_eta=0.2, sigma_eps=1.0, xi=1.0, d=3, dtype=tf.float32),
            dict(iT=20, phi=0.80, sigma_eta=0.1, sigma_eps=0.5, xi=2.0, d=2, dtype=tf.float64),
            dict(iT=8,  phi=0.60, sigma_eta=0.3, sigma_eps=1.2, xi=0.7, d=4, dtype=tf.float32),
        ]

        for cfg in configs:
            out = SV_model_sim_tf_h(**cfg)

            self.assertIsInstance(out, dict)
            self.assertSetEqual(set(out.keys()), {"vY", "h"})

            vY, h = out["vY"], out["h"]

            self.assertIsInstance(vY, tf.Tensor)
            self.assertIsInstance(h, tf.Tensor)
            self.assertEqual(vY.dtype, cfg["dtype"])
            self.assertEqual(h.dtype, cfg["dtype"])

            assert_shape(self, vY, (cfg["iT"], cfg["d"]))
            assert_shape(self, h, (cfg["iT"], cfg["d"]))
            assert_finite(self, vY)
            assert_finite(self, h)

            self.assertFalse(np.all(vY.numpy() == vY.numpy()[0, 0]), msg="vY is degenerate")
            self.assertFalse(np.all(h.numpy() == h.numpy()[0, 0]), msg="h is degenerate")

    # --------------------------------------------------------
    # Edge cases
    # --------------------------------------------------------
    def test_iT_equal_1_univariate(self):
        """Checks the iT=1 branch returns univariate outputs with shape (1,)."""
        out = SV_model_sim_tf_h(
            iT=1,
            phi=0.95,
            sigma_eta=0.2,
            sigma_eps=1.0,
            xi=1.0,
            dtype=tf.float64
        )

        assert_shape(self, out["vY"], (1,))
        assert_shape(self, out["h"], (1,))
        assert_finite(self, out["vY"])
        assert_finite(self, out["h"])

    def test_iT_equal_1_multivariate(self):
        """Checks the iT=1 branch returns multivariate outputs with shape (1,d)."""
        out = SV_model_sim_tf_h(
            iT=1,
            phi=0.95,
            sigma_eta=0.2,
            sigma_eps=1.0,
            xi=1.0,
            d=3,
            dtype=tf.float64
        )

        assert_shape(self, out["vY"], (1, 3))
        assert_shape(self, out["h"], (1, 3))
        assert_finite(self, out["vY"])
        assert_finite(self, out["h"])

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    def test_same_seed_gives_same_output(self):
        """Checks identical seeds and parameters reproduce the same simulated paths."""
        kwargs = dict(
            iT=20,
            phi=0.9,
            sigma_eta=0.2,
            sigma_eps=1.0,
            xi=1.5,
            d=2,
            seed=123,
            dtype=tf.float64
        )

        out1 = SV_model_sim_tf_h(**kwargs)
        out2 = SV_model_sim_tf_h(**kwargs)

        assert_allclose(self, out1["vY"], out2["vY"], atol=1e-12)
        assert_allclose(self, out1["h"], out2["h"], atol=1e-12)

    def test_different_seeds_give_different_output(self):
        """Checks different seeds produce different simulated paths."""
        kwargs1 = dict(
            iT=20,
            phi=0.9,
            sigma_eta=0.2,
            sigma_eps=1.0,
            xi=1.5,
            d=2,
            seed=123,
            dtype=tf.float64
        )
        kwargs2 = dict(
            iT=20,
            phi=0.9,
            sigma_eta=0.2,
            sigma_eps=1.0,
            xi=1.5,
            d=2,
            seed=456,
            dtype=tf.float64
        )

        out1 = SV_model_sim_tf_h(**kwargs1)
        out2 = SV_model_sim_tf_h(**kwargs2)

        self.assertFalse(np.allclose(out1["vY"].numpy(), out2["vY"].numpy()))
        self.assertFalse(np.allclose(out1["h"].numpy(), out2["h"].numpy()))

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------
    def test_valid_configs_run(self):
        """Checks several valid configurations run and return finite outputs."""
        configs = [
            dict(iT=30, phi=0.95, sigma_eta=0.2, sigma_eps=1.0, xi=1.0, dtype=tf.float64),
            dict(iT=50, phi=0.80, sigma_eta=0.1, sigma_eps=0.5, xi=2.0, d=2, dtype=tf.float32),
            dict(iT=40, phi=0.60, sigma_eta=0.3, sigma_eps=1.5, xi=0.7, d=3, dtype=tf.float64),
        ]

        for cfg in configs:
            out = SV_model_sim_tf_h(**cfg)
            assert_finite(self, out["vY"])
            assert_finite(self, out["h"])

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------
    def test_invalid_iT_raises(self):
        """Checks invalid values of iT raise ValueError."""
        bad_calls = [
            dict(iT=0, phi=0.95, sigma_eta=0.2, sigma_eps=1.0, xi=1.0),
            dict(iT=-10, phi=0.95, sigma_eta=0.2, sigma_eps=1.0, xi=1.0),
            dict(iT=2.5, phi=0.95, sigma_eta=0.2, sigma_eps=1.0, xi=1.0),
            dict(iT="10", phi=0.95, sigma_eta=0.2, sigma_eps=1.0, xi=1.0),
        ]

        for kwargs in bad_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SV_model_sim_tf_h(**kwargs)

def test_invalid_numeric_parameters_raise(self):
    """Checks invalid model parameters raise ValueError."""
    bad_calls = [
        dict(iT=10, phi=1.2,  sigma_eta=0.2,  sigma_eps=1.0,  xi=1.0),
        dict(iT=10, phi=-2.0, sigma_eta=0.2,  sigma_eps=1.0,  xi=1.0),
        dict(iT=10, phi=0.95, sigma_eta=-0.2, sigma_eps=1.0,  xi=1.0),
        dict(iT=10, phi=0.95, sigma_eta=0.2,  sigma_eps=-1.0, xi=1.0),
        dict(iT=10, phi=0.95, sigma_eta=0.2,  sigma_eps=1.0,  xi=0.0),
        dict(iT=10, phi=0.95, sigma_eta=0.2,  sigma_eps=1.0,  xi=-1.0),
    ]

    for kwargs in bad_calls:
        with self.subTest(kwargs=kwargs):
            with self.assertRaises(ValueError):
                SV_model_sim_tf_h(**kwargs)


class TestMakePropSV(unittest.TestCase):
    """Tests SV propagation closure for shape, dtype, and validation behavior."""

    @classmethod
    def setUpClass(cls):
        tf.random.set_seed(0)
        cls.dtype = tf.float32

        cls.prop_uni = staticmethod(make_prop_sv(phi=0.9, sigma_eta=0.2, dtype=cls.dtype))
        cls.prop_mv_scalar = staticmethod(make_prop_sv(phi=0.8, sigma_eta=0.3, dtype=cls.dtype))
        cls.prop_mv_vec = staticmethod(
            make_prop_sv(
                phi=tf.constant([0.8, 0.6], dtype=cls.dtype),
                sigma_eta=tf.constant([0.2, 0.4], dtype=cls.dtype),
                dtype=cls.dtype
            )
        )
        cls.prop_mv_cov = staticmethod(
            make_prop_sv(
                phi=0.7,
                sigma_eta=tf.constant([[0.25, 0.05], [0.05, 0.16]], dtype=cls.dtype),
                dtype=cls.dtype
            )
        )

    def test_returns_callable(self):
        """Checks the factory returns a callable propagation function."""
        self.assertTrue(callable(type(self).prop_uni))

    def test_univariate_output_shape_and_dtype(self):
        """Checks univariate propagation returns a finite vector of shape (1,)."""
        x = tf.constant([0.5], dtype=self.dtype)
        out = type(self).prop_uni(x)

        assert_shape(self, out, (1,))
        assert_finite(self, out)
        self.assertEqual(out.dtype, self.dtype)

    def test_multivariate_scalar_phi_scalar_sigma(self):
        """Checks multivariate propagation works with scalar phi and scalar sigma_eta."""
        x = tf.constant([0.1, -0.2, 0.3], dtype=self.dtype)
        out = type(self).prop_mv_scalar(x)

        assert_shape(self, out, (3,))
        assert_finite(self, out)

    def test_multivariate_vector_phi_vector_sigma(self):
        """Checks multivariate propagation works with vector phi and vector sigma_eta."""
        x = tf.constant([1.0, -1.0], dtype=self.dtype)
        out = type(self).prop_mv_vec(x)

        assert_shape(self, out, (2,))
        assert_finite(self, out)

    def test_multivariate_matrix_sigma(self):
        """Checks multivariate propagation works with correlated state noise."""
        x = tf.constant([0.2, -0.1], dtype=self.dtype)
        out = type(self).prop_mv_cov(x)

        assert_shape(self, out, (2,))
        assert_finite(self, out)

    def test_output_is_not_equal_to_input_when_noise_present(self):
        """Checks propagation changes the state when nonzero noise is present."""
        prop = make_prop_sv(phi=0.5, sigma_eta=0.2, dtype=self.dtype)
        x = tf.constant([1.0], dtype=self.dtype)
        out = prop(x)

        self.assertFalse(np.allclose(out.numpy(), x.numpy()))

    def test_rank_two_input_raises(self):
        """Checks state input must have rank 1."""
        prop = make_prop_sv(phi=0.9, sigma_eta=0.2, dtype=self.dtype)

        with self.assertRaises(ValueError):
            prop(tf.constant([[1.0, 2.0]], dtype=self.dtype))

    def test_invalid_phi_vector_length_raises(self):
        """Checks phi vector length mismatch causes an error."""
        prop = make_prop_sv(
            phi=tf.constant([0.8, 0.6], dtype=self.dtype),
            sigma_eta=0.2,
            dtype=self.dtype
        )
        x = tf.constant([1.0, 2.0, 3.0], dtype=self.dtype)

        with self.assertRaises(ValueError):
            prop(x)

    def test_invalid_sigma_vector_length_raises(self):
        """Checks sigma_eta vector length mismatch causes an error."""
        prop = make_prop_sv(
            phi=0.8,
            sigma_eta=tf.constant([0.2, 0.3], dtype=self.dtype),
            dtype=self.dtype
        )
        x = tf.constant([1.0, 2.0, 3.0], dtype=self.dtype)

        with self.assertRaises(ValueError):
            prop(x)

    def test_nonpositive_sigma_scalar_raises(self):
        """Checks scalar sigma_eta must be positive."""
        prop = make_prop_sv(phi=0.8, sigma_eta=0.0, dtype=self.dtype)
        x = tf.constant([1.0], dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            prop(x)

    def test_nonpositive_sigma_vector_entry_raises(self):
        """Checks all sigma_eta vector entries must be positive."""
        prop = make_prop_sv(
            phi=0.8,
            sigma_eta=tf.constant([0.2, 0.0], dtype=self.dtype),
            dtype=self.dtype
        )
        x = tf.constant([1.0, 2.0], dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            prop(x)

    def test_unstable_scalar_phi_raises(self):
        """Checks scalar phi must satisfy |phi| < 1."""
        prop = make_prop_sv(phi=1.0, sigma_eta=0.2, dtype=self.dtype)
        x = tf.constant([1.0], dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            prop(x)

    def test_unstable_vector_phi_raises(self):
        """Checks all phi vector entries must satisfy |phi_i| < 1."""
        prop = make_prop_sv(
            phi=tf.constant([0.8, 1.0], dtype=self.dtype),
            sigma_eta=0.2,
            dtype=self.dtype
        )
        x = tf.constant([1.0, 2.0], dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            prop(x)

    def test_unstable_matrix_phi_raises(self):
        """Checks matrix phi must be stable."""
        phi = tf.constant([[1.1, 0.0], [0.0, 0.5]], dtype=self.dtype)
        prop = make_prop_sv(phi=phi, sigma_eta=0.2, dtype=self.dtype)
        x = tf.constant([1.0, 2.0], dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            prop(x)


class TestMakeLoglikSV(unittest.TestCase):
    """Tests SV Gaussian log-likelihood closure for correctness and validation."""

    @classmethod
    def setUpClass(cls):
        tf.random.set_seed(0)
        cls.dtype = tf.float32

        cls.loglik_uni = staticmethod(make_loglik_sv(sigma_eps=0.5, xi=1.0, dtype=cls.dtype))
        cls.loglik_mv = staticmethod(
            make_loglik_sv(
                sigma_eps=tf.constant([0.5, 0.8], dtype=cls.dtype),
                xi=tf.constant([1.0, 1.2], dtype=cls.dtype),
                dtype=cls.dtype
            )
        )
        cls.loglik_broadcast = staticmethod(make_loglik_sv(sigma_eps=0.5, xi=1.5, dtype=cls.dtype))

    def test_returns_callable(self):
        """Checks the factory returns a callable log-likelihood function."""
        self.assertTrue(callable(type(self).loglik_uni))

    def test_univariate_output_shape_and_dtype(self):
        """Checks univariate log-likelihood returns shape (Np,) with finite values."""
        particles = tf.constant([[0.0], [0.2], [-0.1]], dtype=self.dtype)
        y = tf.constant([1.0], dtype=self.dtype)

        out = type(self).loglik_uni(particles, y)

        assert_shape(self, out, (3,))
        assert_finite(self, out)
        self.assertEqual(out.dtype, self.dtype)

    def test_multivariate_output_shape_and_dtype(self):
        """Checks multivariate log-likelihood returns one value per particle."""
        particles = tf.constant(
            [[0.0, 0.1],
             [0.2, -0.3],
             [-0.1, 0.4]],
            dtype=self.dtype
        )
        y = tf.constant([1.0, -0.5], dtype=self.dtype)

        out = type(self).loglik_mv(particles, y)

        assert_shape(self, out, (3,))
        assert_finite(self, out)
        self.assertEqual(out.dtype, self.dtype)

    def test_known_univariate_value_matches_manual_formula(self):
        """Checks the univariate log-likelihood matches the manual Gaussian formula."""
        sigma_eps = 0.5
        xi = 2.0
        loglik = make_loglik_sv(sigma_eps=sigma_eps, xi=xi, dtype=self.dtype)

        particles = tf.constant([[0.0]], dtype=self.dtype)
        y = tf.constant([1.5], dtype=self.dtype)

        out = loglik(particles, y)

        var = (xi ** 2) * (sigma_eps ** 2) * np.exp(0.0)
        expected = -0.5 * (np.log(2.0 * np.pi) + np.log(var) + (1.5 ** 2) / var)
        self.assertAlmostEqual(float(out.numpy()[0]), float(expected), places=5)

    def test_scalar_parameters_broadcast_across_dimensions(self):
        """Checks scalar sigma_eps and xi broadcast correctly in multivariate settings."""
        particles = tf.constant(
            [[0.0, 0.1],
             [0.2, -0.2]],
            dtype=self.dtype
        )
        y = tf.constant([1.0, -1.0], dtype=self.dtype)

        out = type(self).loglik_broadcast(particles, y)

        assert_shape(self, out, (2,))
        assert_finite(self, out)

    def test_particles_rank_not_two_raises(self):
        """Checks particles input must have rank 2."""
        particles = tf.constant([0.0, 0.1], dtype=self.dtype)
        y = tf.constant([1.0], dtype=self.dtype)
    
        with self.assertRaises(tf.errors.InvalidArgumentError):
            type(self).loglik_uni(particles, y)


    def test_y_rank_not_one_raises(self):
        """Checks y input must have rank 1."""
        particles = tf.constant([[0.0], [0.1]], dtype=self.dtype)
        y = tf.constant([[1.0]], dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            type(self).loglik_uni(particles, y)

    def test_y_dimension_mismatch_raises(self):
        """Checks y dimension must match the particle dimension."""
        particles = tf.constant([[0.0, 0.1], [0.2, -0.1]], dtype=self.dtype)
        y = tf.constant([1.0], dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            type(self).loglik_mv(particles, y)

    def test_sigma_eps_nonpositive_scalar_raises(self):
        """Checks scalar sigma_eps must be strictly positive."""
        loglik = make_loglik_sv(sigma_eps=0.0, xi=1.0, dtype=self.dtype)
        particles = tf.constant([[0.0]], dtype=self.dtype)
        y = tf.constant([1.0], dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            loglik(particles, y)

    def test_sigma_eps_nonpositive_vector_entry_raises(self):
        """Checks all sigma_eps vector entries must be strictly positive."""
        loglik = make_loglik_sv(
            sigma_eps=tf.constant([0.5, 0.0], dtype=self.dtype),
            xi=1.0,
            dtype=self.dtype
        )
        particles = tf.constant([[0.0, 0.1]], dtype=self.dtype)
        y = tf.constant([1.0, -1.0], dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            loglik(particles, y)

    def test_xi_nonpositive_scalar_raises(self):
        """Checks scalar xi must be strictly positive."""
        loglik = make_loglik_sv(sigma_eps=0.5, xi=0.0, dtype=self.dtype)
        particles = tf.constant([[0.0]], dtype=self.dtype)
        y = tf.constant([1.0], dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            loglik(particles, y)

    def test_xi_nonpositive_vector_entry_raises(self):
        """Checks all xi vector entries must be strictly positive."""
        loglik = make_loglik_sv(
            sigma_eps=0.5,
            xi=tf.constant([1.0, 0.0], dtype=self.dtype),
            dtype=self.dtype
        )
        particles = tf.constant([[0.0, 0.1]], dtype=self.dtype)
        y = tf.constant([1.0, -1.0], dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            loglik(particles, y)

    def test_sigma_eps_wrong_length_raises(self):
        """Checks sigma_eps vector length mismatch causes an error."""
        loglik = make_loglik_sv(
            sigma_eps=tf.constant([0.5, 0.8], dtype=self.dtype),
            xi=1.0,
            dtype=self.dtype
        )
        particles = tf.constant([[0.0, 0.1, -0.2]], dtype=self.dtype)
        y = tf.constant([1.0, -1.0, 0.5], dtype=self.dtype)

        with self.assertRaises(ValueError):
            loglik(particles, y)

    def test_xi_wrong_length_raises(self):
        """Checks xi vector length mismatch causes an error."""
        loglik = make_loglik_sv(
            sigma_eps=0.5,
            xi=tf.constant([1.0, 1.2], dtype=self.dtype),
            dtype=self.dtype
        )
        particles = tf.constant([[0.0, 0.1, -0.2]], dtype=self.dtype)
        y = tf.constant([1.0, -1.0, 0.5], dtype=self.dtype)

        with self.assertRaises(ValueError):
            loglik(particles, y)

    def test_extreme_particles_still_return_finite_when_valid(self):
        """Checks moderately large but valid particles still give finite output."""
        particles = tf.constant([[10.0], [-10.0], [5.0]], dtype=self.dtype)
        y = tf.constant([1.0], dtype=self.dtype)

        out = type(self).loglik_uni(particles, y)

        assert_shape(self, out, (3,))
        assert_finite(self, out)

class BaseResamplingTest(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.dtype = tf.float64
        self.Np = 10
        self.d = 3

    def resampling_fn(self):
        raise NotImplementedError

    def default_kwargs(self):
        return {}

    # --------------------------------------------------------
    # Shared wrapper
    # --------------------------------------------------------
    def run_resampling(self, particles, weights, **kwargs):
        fn = self.resampling_fn()
        final_kwargs = self.default_kwargs().copy()
        final_kwargs.update(kwargs)
        return fn(particles, weights, **final_kwargs)

    # --------------------------------------------------------
    # Shared tests: true for all resamplers
    # --------------------------------------------------------
    def test_output_shapes(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.ones(self.Np, dtype=self.dtype) / self.Np

        new_particles, new_weights = self.run_resampling(particles, weights)

        assert_shape(self, new_particles, (self.Np, self.d))
        assert_shape(self, new_weights, (self.Np,))

    def test_outputs_are_finite(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.ones(self.Np, dtype=self.dtype) / self.Np

        new_particles, new_weights = self.run_resampling(particles, weights)

        assert_finite(self, new_particles)
        assert_finite(self, new_weights)

    def test_weights_normalization(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.ones(self.Np, dtype=self.dtype) / self.Np

        _, new_weights = self.run_resampling(particles, weights)

        np.testing.assert_allclose(
            tf.reduce_sum(new_weights).numpy(),
            1.0,
            atol=1e-6
        )

    def test_particle_shape_preserved(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.linspace(
            tf.constant(0.1, dtype=self.dtype),
            tf.constant(1.0, dtype=self.dtype),
            self.Np
        )
        weights = weights / tf.reduce_sum(weights)

        new_particles, _ = self.run_resampling(particles, weights)

        assert_shape(self, new_particles, (self.Np, self.d))

    def test_weight_shape_preserved(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.linspace(
            tf.constant(0.1, dtype=self.dtype),
            tf.constant(1.0, dtype=self.dtype),
            self.Np
        )
        weights = weights / tf.reduce_sum(weights)

        _, new_weights = self.run_resampling(particles, weights)

        assert_shape(self, new_weights, (self.Np,))


# ============================================================
# Multinomial
# ============================================================

class TestMultinomialResampling(BaseResamplingTest):

    def resampling_fn(self):
        return multinomial_resampling

    def test_uniform_weights_after_resampling(self):
        particles = tf.constant([[1.0], [2.0], [3.0], [4.0]], dtype=self.dtype)
        weights = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=self.dtype)

        _, new_weights = self.run_resampling(particles, weights)
        expected = tf.ones_like(new_weights) / tf.cast(tf.shape(new_weights)[0], self.dtype)

        assert_allclose(self, new_weights, expected, atol=1e-6)

class TestBootstrapParticleFilterResampling(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.dtype = tf.float64
        self.Np = 20

    # --------------------------------------------------------
    # Simple shared mocks
    # --------------------------------------------------------
    def identity_prop_fn(self, x):
        return x

    def zero_loglik_fn(self, particles, y):
        return tf.zeros((self.Np,), dtype=self.dtype)

    # --------------------------------------------------------
    # Basic valid behavior
    # --------------------------------------------------------
    def test_basic_run(self):
        Y = tf.constant([[1.0], [2.0], [3.0]], dtype=self.dtype)
        prop_fn, log_likelihood_fn = make_simple_functions(self.Np, self.dtype)

        ests, ESSs, total_loglik = bpf_generic_resampling(
            Y, self.Np, prop_fn, log_likelihood_fn, dtype=self.dtype
        )

        assert_valid_output(self, ests, ESSs, T=3, d=1)
        assert_valid_ess(self, ESSs, self.Np)
        assert_valid_loglik(self, total_loglik)

    def test_numpy_input_conversion(self):
        Y = np.ones((5, 2))
        prop_fn, log_likelihood_fn = make_simple_functions(self.Np, self.dtype)

        ests, ESSs, total_loglik = bpf_generic_resampling(
            Y, self.Np, prop_fn, log_likelihood_fn, dtype=self.dtype
        )

        assert_valid_output(self, ests, ESSs, T=5, d=2)
        assert_valid_ess(self, ESSs, self.Np)
        assert_valid_loglik(self, total_loglik)

    def test_zero_loglik_gives_zero_estimates_and_zero_loglik(self):
        T, d = 4, 2
        Y = tf.ones((T, d), dtype=self.dtype)

        ests, ESSs, total_loglik = bpf_generic_resampling(
            Y,
            self.Np,
            self.identity_prop_fn,
            self.zero_loglik_fn,
            dtype=self.dtype
        )

        assert_valid_output(self, ests, ESSs, T=T, d=d)
        assert_valid_ess(self, ESSs, self.Np)
        assert_valid_loglik(self, total_loglik)

        np.testing.assert_allclose(ests.numpy(), np.zeros((T, d)), atol=1e-12)
        self.assertAlmostEqual(total_loglik.numpy(), 0.0, places=10)

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------
    def test_invalid_inputs_raise(self):
        Y_good = tf.ones((3, 1), dtype=self.dtype)
        prop_fn, log_likelihood_fn = make_simple_functions(self.Np, self.dtype)

        bad_calls = [
            (lambda: bpf_generic_resampling(Y_good, 1, prop_fn, log_likelihood_fn), ValueError),
            (lambda: bpf_generic_resampling(Y_good, "20", prop_fn, log_likelihood_fn), ValueError),
            (lambda: bpf_generic_resampling(tf.ones((3,), dtype=self.dtype), self.Np, prop_fn, log_likelihood_fn), ValueError),
            (lambda: bpf_generic_resampling(tf.ones((0, 2), dtype=self.dtype), self.Np, prop_fn, log_likelihood_fn), ValueError),
            (lambda: bpf_generic_resampling(Y_good, self.Np, None, log_likelihood_fn), TypeError),
            (lambda: bpf_generic_resampling(Y_good, self.Np, prop_fn, None), TypeError),
            (lambda: bpf_generic_resampling(Y_good, self.Np, prop_fn, log_likelihood_fn, resampling_fn=None), TypeError),
        ]

        for fn, err in bad_calls:
            with self.subTest(error=err):
                assert_raises(self, fn, error=err)

    # --------------------------------------------------------
    # Likelihood sanity
    # --------------------------------------------------------
    def test_constant_loglik_matches_total_loglik(self):
        T = 4
        c = tf.constant(-2.5, dtype=self.dtype)
        Y = tf.ones((T, 1), dtype=self.dtype)

        def constant_loglik_fn(particles, y):
            return tf.ones((self.Np,), dtype=self.dtype) * c

        ests, ESSs, total_loglik = bpf_generic_resampling(
            Y,
            self.Np,
            self.identity_prop_fn,
            constant_loglik_fn,
            dtype=self.dtype,
            carry_resampled_weights=False
        )

        assert_valid_output(self, ests, ESSs, T=T, d=1)
        assert_valid_ess(self, ESSs, self.Np)
        assert_valid_loglik(self, total_loglik)
        self.assertAlmostEqual(total_loglik.numpy(), T * c.numpy(), places=10)

    def test_nan_loglik_raises_invalid_argument(self):
        Y = tf.ones((2, 1), dtype=self.dtype)

        def nan_loglik_fn(particles, y):
            vals = tf.zeros((self.Np,), dtype=self.dtype)
            vals = tf.tensor_scatter_nd_update(
                vals,
                indices=[[0]],
                updates=[tf.constant(np.nan, dtype=np.float64)]
            )
            return vals

        with self.assertRaises(tf.errors.InvalidArgumentError):
            bpf_generic_resampling(
                Y, self.Np, self.identity_prop_fn, nan_loglik_fn, dtype=self.dtype
            )

    # --------------------------------------------------------
    # Generic-resampling branches
    # --------------------------------------------------------
    def test_carry_resampled_weights_branch_runs(self):
        Y = tf.constant([[1.0], [2.0]], dtype=self.dtype)

        def log_likelihood_fn(particles, y):
            return tf.linspace(
                tf.constant(-1.0, dtype=self.dtype),
                tf.constant(0.0, dtype=self.dtype),
                self.Np
            )

        def identity_resampling(particles, weights):
            return particles, weights

        ests, ESSs, total_loglik = bpf_generic_resampling(
            Y,
            self.Np,
            self.identity_prop_fn,
            log_likelihood_fn,
            resampling_fn=identity_resampling,
            carry_resampled_weights=True,
            dtype=self.dtype
        )

        assert_valid_output(self, ests, ESSs, T=2, d=1)
        assert_valid_ess(self, ESSs, self.Np)
        assert_valid_loglik(self, total_loglik)

    def test_resample_threshold_branch_runs(self):
        Y = tf.constant([[1.0], [2.0], [3.0]], dtype=self.dtype)

        def peaked_loglik_fn(particles, y):
            vals = tf.fill((self.Np,), tf.constant(-100.0, dtype=self.dtype))
            vals = tf.tensor_scatter_nd_update(
                vals,
                indices=[[0]],
                updates=[tf.constant(0.0, dtype=self.dtype)]
            )
            return vals

        ests, ESSs, total_loglik = bpf_generic_resampling(
            Y,
            self.Np,
            self.identity_prop_fn,
            peaked_loglik_fn,
            resample_threshold=True,
            dtype=self.dtype
        )

        assert_valid_output(self, ests, ESSs, T=3, d=1)
        assert_valid_ess(self, ESSs, self.Np)
        assert_valid_loglik(self, total_loglik)

    def test_nan_resampled_weights_raise_invalid_argument(self):
        Y = tf.ones((2, 1), dtype=self.dtype)

        def bad_resampling_fn(particles, weights):
            new_weights = tf.fill((self.Np,), tf.constant(np.nan, dtype=np.float64))
            return particles, new_weights

        with self.assertRaises(tf.errors.InvalidArgumentError):
            bpf_generic_resampling(
                Y,
                self.Np,
                self.identity_prop_fn,
                self.zero_loglik_fn,
                resampling_fn=bad_resampling_fn,
                carry_resampled_weights=True,
                dtype=self.dtype
            )



################
########## UKF/EKF
class TestUKFUpdate(unittest.TestCase):

    # --------------------------------------------------------
    # Test shape and dtype
    # --------------------------------------------------------
    def test_shapes_and_dtypes(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)
        y = tf.constant([0.5], dtype=tf.float64)

        def h(x, t):
            return tf.reduce_sum(x, keepdims=True)

        R = tf.eye(1, dtype=tf.float64)

        x_filt, P_filt, v, S, K = ukf_update_check(x, P, y, h, R)

        self.assertEqual(x_filt.shape, (2,))
        self.assertEqual(P_filt.shape, (2, 2))
        self.assertEqual(v.shape, (1,))
        self.assertEqual(S.shape, (1, 1))
        self.assertEqual(K.shape, (2, 1))

    # --------------------------------------------------------
    # Test finiteness
    # --------------------------------------------------------
    def test_outputs_are_finite(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)
        y = tf.constant([0.5], dtype=tf.float64)

        def h(x, t):
            return tf.reduce_sum(x, keepdims=True)

        R = tf.eye(1, dtype=tf.float64)

        x_filt, P_filt, v, S, K = ukf_update_check(x, P, y, h, R)

        assert_finite(self, x_filt)
        assert_finite(self, P_filt)
        assert_finite(self, v)
        assert_finite(self, S)
        assert_finite(self, K)

    # --------------------------------------------------------
    # Test covariance properties
    # --------------------------------------------------------
    def test_covariance_properties(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)
        y = tf.constant([0.5], dtype=tf.float64)

        def h(x, t):
            return tf.reduce_sum(x, keepdims=True)

        R = tf.eye(1, dtype=tf.float64)

        _, P_filt, _, S, _ = ukf_update_check(x, P, y, h, R)

        assert_symmetric(self, P_filt)
        assert_positive_definite(self, P_filt)

        assert_symmetric(self, S)
        assert_positive_definite(self, S)

    # --------------------------------------------------------
    # Deterministic behavior
    # --------------------------------------------------------
    def test_deterministic(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)
        y = tf.constant([0.5], dtype=tf.float64)

        def h(x, t):
            return tf.reduce_sum(x, keepdims=True)

        R = tf.eye(1, dtype=tf.float64)

        out1 = ukf_update_check(x, P, y, h, R)
        out2 = ukf_update_check(x, P, y, h, R)

        for a, b in zip(out1, out2):
            assert_allclose(self, a, b)

    # --------------------------------------------------------
    # Error: non-finite input
    # --------------------------------------------------------
    def test_non_finite_input_raises(self):

        x = tf.constant([float('nan'), 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)
        y = tf.constant([0.5], dtype=tf.float64)

        def h(x, t):
            return tf.reduce_sum(x, keepdims=True)

        R = tf.eye(1, dtype=tf.float64)

#        with self.assertRaises(Exception):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            ukf_update_check(x, P, y, h, R)

    # --------------------------------------------------------
    # Error: non-positive definite covariance
    # --------------------------------------------------------
    def test_non_positive_definite_raises(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)

        P = tf.constant([[1.0, 2.0],
                         [2.0, 1.0]], dtype=tf.float64)

        y = tf.constant([0.5], dtype=tf.float64)

        def h(x, t):
            return tf.reduce_sum(x, keepdims=True)

        R = tf.eye(1, dtype=tf.float64)

        #with self.assertRaises(Exception):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            ukf_update_check(x, P, y, h, R)

    # --------------------------------------------------------
    # Numerical stability
    # --------------------------------------------------------
    def test_covariance_stability(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)
        y = tf.constant([0.5], dtype=tf.float64)

        def h(x, t):
            return tf.reduce_sum(x, keepdims=True)

        R = tf.eye(1, dtype=tf.float64)

        _, P_filt, _, _, _ = ukf_update_check(x, P, y, h, R)

        diag = tf.linalg.diag_part(P_filt)

        self.assertTrue(np.all(diag.numpy() > 0))
        self.assertTrue(np.all(tf.linalg.eigvalsh(P_filt).numpy() > -1e-8))

        assert_finite(self, P_filt)


    def test_nonlinear_measurement(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)
        y = tf.constant([5.0], dtype=tf.float64)

        # nonlinear function
        def h(x, t):
            return tf.stack([x[0]**2 + tf.sin(x[1])])

        R = tf.eye(1, dtype=tf.float64) * 0.1

        x_filt, P_filt, _, _, _ = ukf_update_check(x, P, y, h, R)

        # Basic sanity checks
        assert_finite(self, x_filt)
        assert_finite(self, P_filt)

        # State should move (filter should actually update)
        self.assertFalse(np.allclose(x_filt.numpy(), x.numpy()))

    def test_state_update_effect(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)

        y = tf.constant([10.0], dtype=tf.float64)  # far from prediction

        def h(x, t):
            return tf.stack([tf.reduce_sum(x)])
    
        R = tf.eye(1, dtype=tf.float64) * 0.01

        x_filt, _, _, _, _ = ukf_update_check(x, P, y, h, R)

        # should move towards measurement
        self.assertFalse(np.allclose(x_filt.numpy(), x.numpy()))



class TestSigmaPointsBatch(unittest.TestCase):

    def setUp(self):
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0

    # --------------------------------------------------------
    # Shape test
    # --------------------------------------------------------
    def test_shapes(self):

        x = tf.random.normal((3, 4), dtype=tf.float64)
        P = tf.eye(4, dtype=tf.float64)

        sigma, Wm, Wc = unscented_sigma_points_batch(
            x, P, self.alpha, self.beta, self.kappa
        )

        self.assertEqual(sigma.shape, (3, 2 * 4 + 1, 4))
        self.assertEqual(Wm.shape, (2 * 4 + 1,))
        self.assertEqual(Wc.shape, (2 * 4 + 1,))

    # --------------------------------------------------------
    # Weights sanity
    # --------------------------------------------------------
    def test_weights(self):

        x = tf.zeros((1, 3), dtype=tf.float64)
        P = tf.eye(3, dtype=tf.float64)

        _, Wm, Wc = unscented_sigma_points_batch(
            x, P, self.alpha, self.beta, self.kappa
        )

        self.assertTrue(np.isclose(tf.reduce_sum(Wm).numpy(), 1.0))
        self.assertEqual(Wm.shape, Wc.shape)

    # --------------------------------------------------------
    # Mean reconstruction
    # --------------------------------------------------------
    def test_mean_reconstruction(self):

        x = tf.constant([[1.0, -2.0, 3.0]], dtype=tf.float64)
        P = tf.eye(3, dtype=tf.float64)

        sigma, Wm, _ = unscented_sigma_points_batch(
            x, P, self.alpha, self.beta, self.kappa
        )

        mean = tf.reduce_sum(sigma * Wm[None, :, None], axis=1)

        assert_allclose(self, mean, x, atol=1e-6)

    # --------------------------------------------------------
    # Sigma symmetry
    # --------------------------------------------------------
    def test_sigma_symmetry(self):

        x = tf.zeros((1, 2), dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)

        sigma, _, _ = unscented_sigma_points_batch(
            x, P, self.alpha, self.beta, self.kappa
        )

        center = sigma[:, 0, :]
        plus   = sigma[:, 1:3, :]
        minus  = sigma[:, 3:5, :]

        assert_allclose(
            self,
            plus - center[:, None, :],
            -(minus - center[:, None, :]),
            atol=1e-6
        )

    # --------------------------------------------------------
    # Particle shift invariance
    # --------------------------------------------------------
    def test_particle_shift_invariance(self):

        x1 = tf.constant([[0.0, 0.0]], dtype=tf.float64)
        x2 = tf.constant([[10.0, 10.0]], dtype=tf.float64)

        P = tf.eye(2, dtype=tf.float64)

        sigma1, _, _ = unscented_sigma_points_batch(x1, P, self.alpha, self.beta, self.kappa)
        sigma2, _, _ = unscented_sigma_points_batch(x2, P, self.alpha, self.beta, self.kappa)

        shift = x2 - x1  # (1, d)

        assert_allclose(
            self,
            sigma2,
            sigma1 + shift[:, None, :],
            atol=1e-6
        )

    # --------------------------------------------------------
    # Finite outputs
    # --------------------------------------------------------
    def test_finite(self):

        x = tf.random.normal((5, 6), dtype=tf.float64)
        P = tf.eye(6, dtype=tf.float64)

        sigma, Wm, Wc = unscented_sigma_points_batch(
            x, P, self.alpha, self.beta, self.kappa
        )

        assert_finite(self, sigma)
        assert_finite(self, Wm)
        assert_finite(self, Wc)


class TestUnscentedSigmaPoints(unittest.TestCase):

    def test_shapes(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)

        sigma, Wm, Wc = unscented_sigma_points(x, P, 1e-3, 2.0, 0.0)

        self.assertEqual(sigma.shape, (5, 2))
        self.assertEqual(Wm.shape, (5,))
        self.assertEqual(Wc.shape, (5,))

class TestUnscentedTransform(unittest.TestCase):

    def setUp(self):
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------
    def test_shapes(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)

        sigma, Wm, Wc = unscented_sigma_points(
            x, P, self.alpha, self.beta, self.kappa
        )

        mean, cov, diff = unscented_transform(sigma, Wm, Wc)

        self.assertEqual(sigma.shape, (5, 2))
        self.assertEqual(Wm.shape, (5,))
        self.assertEqual(Wc.shape, (5,))
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(cov.shape, (2, 2))
        self.assertEqual(diff.shape, (5, 2))

    # --------------------------------------------------------
    # Finite outputs
    # --------------------------------------------------------
    def test_finite_outputs(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)

        sigma, Wm, Wc = unscented_sigma_points(
            x, P, self.alpha, self.beta, self.kappa
        )

        mean, cov, diff = unscented_transform(sigma, Wm, Wc)

        assert_finite(self, sigma)
        assert_finite(self, Wm)
        assert_finite(self, Wc)
        assert_finite(self, mean)
        assert_finite(self, cov)
        assert_finite(self, diff)

    # --------------------------------------------------------
    # Mean reconstruction
    # --------------------------------------------------------
    def test_mean_reconstruction(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)

        sigma, Wm, Wc = unscented_sigma_points(
            x, P, self.alpha, self.beta, self.kappa
        )

        mean, _, _ = unscented_transform(sigma, Wm, Wc)

        assert_allclose(self, mean, x, atol=1e-6)

    # --------------------------------------------------------
    # Covariance reconstruction
    # --------------------------------------------------------
    def test_covariance_reconstruction(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.constant([[1.0, 0.2],
                         [0.2, 2.0]], dtype=tf.float64)

        sigma, Wm, Wc = unscented_sigma_points(
            x, P, self.alpha, self.beta, self.kappa
        )

        _, cov, _ = unscented_transform(sigma, Wm, Wc)

        assert_symmetric(self, cov)
        assert_positive_definite(self, cov)

        assert_allclose(self, cov, P, atol=1e-5)

    # --------------------------------------------------------
    # Deterministic behavior
    # --------------------------------------------------------
    def test_deterministic(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)

        sigma1, Wm1, Wc1 = unscented_sigma_points(
            x, P, self.alpha, self.beta, self.kappa
        )
        mean1, cov1, _ = unscented_transform(sigma1, Wm1, Wc1)

        sigma2, Wm2, Wc2 = unscented_sigma_points(
            x, P, self.alpha, self.beta, self.kappa
        )
        mean2, cov2, _ = unscented_transform(sigma2, Wm2, Wc2)

        assert_allclose(self, sigma1, sigma2)
        assert_allclose(self, mean1, mean2)
        assert_allclose(self, cov1, cov2)

    # --------------------------------------------------------
    # Noise covariance handling
    # --------------------------------------------------------
    def test_noise_addition(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)
        R = 0.1 * tf.eye(2, dtype=tf.float64)

        sigma, Wm, Wc = unscented_sigma_points(
            x, P, self.alpha, self.beta, self.kappa
        )

        _, cov_no_noise, _ = unscented_transform(sigma, Wm, Wc)
        _, cov_with_noise, _ = unscented_transform(sigma, Wm, Wc, R)

        assert_allclose(self, cov_with_noise, cov_no_noise + R)

    # --------------------------------------------------------
    # Stability sanity
    # --------------------------------------------------------
    def test_covariance_reasonable(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)

        sigma, Wm, Wc = unscented_sigma_points(
            x, P, self.alpha, self.beta, self.kappa
        )

        _, cov, _ = unscented_transform(sigma, Wm, Wc)

        assert_finite(self, cov)
        self.assertLess(tf.norm(cov).numpy(), 1e6)


class TestUKFPredict(unittest.TestCase):

    def setUp(self):
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0

    def test_shapes_and_finite(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)
        Q = 0.01 * tf.eye(2, dtype=tf.float64)

        def f(x, t):
            return x  # identity dynamics (simple sanity test)

        x_pred, P_pred = ukf_predict(
            x, P, f, Q, self.alpha, self.beta, self.kappa, t=0
        )

        # --- Shape checks ---
        self.assertEqual(x_pred.shape, (2,))
        self.assertEqual(P_pred.shape, (2, 2))

        # --- Finite checks ---
        self.assertTrue(np.all(np.isfinite(x_pred.numpy())))
        self.assertTrue(np.all(np.isfinite(P_pred.numpy())))

    ## INPUT CHECK
#    def test_nan_input_raises(self):
#        x = tf.constant([float('nan'), 2.0], dtype=tf.float64)
#        P = tf.eye(2, dtype=tf.float64)

#        def f(x, t):
#            return x

##        with self.assertRaises(Exception):
#        with self.assertRaises(tf.errors.InvalidArgumentError):
#            ukf_predict(x, P, f, Q=0.01*tf.eye(2), alpha=self.alpha, beta=self.beta, kappa=self.kappa, t=0)

    def test_deterministic(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)
        Q = 0.01 * tf.eye(2, dtype=tf.float64)

        def f(x, t):
            return tf.stack([x[0] + 1.0, x[1] + 2.0])

        out1 = ukf_predict(x, P, f, Q, self.alpha, self.beta, self.kappa, t=0)
        out2 = ukf_predict(x, P, f, Q, self.alpha, self.beta, self.kappa, t=0)

#        self.assertAllClose(out1[0], out2[0])
#        self.assertAllClose(out1[1], out2[1])
        assert_allclose(self, out1[0], out2[0])
        assert_allclose(self, out1[1], out2[1])

    def test_nonlinear_dynamics(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)
        Q = 0.01 * tf.eye(2, dtype=tf.float64)

        def f(x, t):
            return tf.stack([
                x[0] ** 2,
                tf.sin(x[1])
            ])

        x_pred, P_pred = ukf_predict(
            x, P, f, Q, self.alpha, self.beta, self.kappa, t=0
        )

        assert_finite(self, x_pred)
        assert_finite(self, P_pred)

        # Ensure something actually changed
        self.assertFalse(np.allclose(x_pred.numpy(), x.numpy())) # atol=1e-3

    def test_covariance_structure(self):

        x = tf.constant([1.0, 2.0], dtype=tf.float64)
        P = tf.eye(2, dtype=tf.float64)
        Q = 0.1 * tf.eye(2, dtype=tf.float64)

        def f(x, t):
            return x  # identity dynamics

        _, P_pred = ukf_predict(
            x, P, f, Q, self.alpha, self.beta, self.kappa, t=0
        )

        # Symmetry
        self.assertTrue(np.allclose(P_pred.numpy(), P_pred.numpy().T, atol=1e-8))

        # Positive definiteness (eigenvalues > 0)
        eigs = np.linalg.eigvalsh(P_pred.numpy())
        self.assertTrue(np.all(eigs > -1e-8))

        # Covariance should not shrink unrealistically
#        self.assertGreater(np.linalg.norm(P_pred.numpy()), 0.0) # too weak
        self.assertGreater(np.trace(P_pred.numpy()), 0.0)


class TestUKFKernels(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.Q = 0.1 * tf.eye(2, dtype=self.dtype)
        self.R = tf.eye(1, dtype=self.dtype)

    def test_ukf_kernels_basic(self):

        def f(x, t):
            return tf.stack([x[0] + x[1], x[1]])

        def h(x, t):
            return tf.stack([x[0]])

        predict_fn, update_fn = make_ukf_kernels(
            f, h, self.Q,
            alpha=1e-3, beta=2.0, kappa=0.0
        )

        x = tf.constant([1.0, 2.0], dtype=self.dtype)
        P = tf.eye(2, dtype=self.dtype)
        y = tf.constant([1.5], dtype=self.dtype)

        # --- predict ---
        x_pred, P_pred = predict_fn(x, P, t=0)

        self.assertEqual(x_pred.shape, (2,))
        self.assertEqual(P_pred.shape, (2, 2))

        # --- update ---
        x_filt, P_filt, v, S = update_fn["step"](x_pred, P_pred, y, self.R, t=0)

        self.assertEqual(x_filt.shape, (2,))
        self.assertEqual(P_filt.shape, (2, 2))

        # --- finite ---
        self.assertTrue(np.all(np.isfinite(x_filt.numpy())))
        self.assertTrue(np.all(np.isfinite(P_filt.numpy())))

class TestEKFKernels(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.Q = 0.1 * tf.eye(2, dtype=self.dtype)
        self.R = tf.eye(1, dtype=self.dtype)

    def test_ekf_kernels_basic(self):

        def f(x, t):
            return tf.stack([x[0] + x[1], x[1]])

        def h(x, t):
            return tf.stack([x[0]])

        def F_jac(x, t):
            return tf.constant([[1.0, 1.0],
                                [0.0, 1.0]], dtype=self.dtype)

        def H_jac(x, t):
            return tf.constant([[1.0, 0.0]], dtype=self.dtype)

        predict_fn, update_fn = make_ekf_kernels(
            f, h, F_jac, H_jac, self.Q
        )

        x = tf.constant([1.0, 2.0], dtype=self.dtype)
        P = tf.eye(2, dtype=self.dtype)
        y = tf.constant([1.5], dtype=self.dtype)

        # --- predict ---
        x_pred, P_pred = predict_fn(x, P, t=0)

        self.assertEqual(x_pred.shape, (2,))
        self.assertEqual(P_pred.shape, (2, 2))

        # --- update ---
        x_filt, P_filt, v, S = update_fn["step"](x_pred, P_pred, y, self.R, t=0)

        self.assertEqual(x_filt.shape, (2,))
        self.assertEqual(P_filt.shape, (2, 2))

        # --- finite ---
        self.assertTrue(np.all(np.isfinite(x_filt.numpy())))
        self.assertTrue(np.all(np.isfinite(P_filt.numpy())))


class TestFilterCore(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64

    def test_filter_core_basic(self):

        # --- simple data ---
        Y = tf.constant([[1.0, 2.0, 3.0]], dtype=self.dtype)  # (1, T=3)

        m0 = tf.constant([0.0], dtype=self.dtype)
        P0 = tf.eye(1, dtype=self.dtype)
        R = tf.eye(1, dtype=self.dtype)

        # --- minimal mock predict ---
        def predict_fn(x, P, t):
            return x, P

        # --- minimal mock update ---
        def update_step(x_pred, P_pred, y, R_mat, t):
            v = y - x_pred
            return x_pred, P_pred, v, R_mat

        def h(x, t):
            return tf.reduce_sum(x, keepdims=True)

        update_fn = {
            "step": update_step,
            "h": h
        }

        # --- call function ---
        result = _filter_core(
            Y=Y,
            predict_fn=predict_fn,
            update_fn=update_fn,
            R_mat=R,
            m0=m0,
            P0=P0,
            measurement_type="gaussian",
            dtype=self.dtype
        )

        # --- structure checks ---
        self.assertIn("mu_filt", result)
        self.assertIn("P_filt", result)
        self.assertIn("P_pred", result)
        self.assertIn("loglik", result)

        # --- shape checks ---
        self.assertEqual(result["mu_filt"].shape, (1, 3))
        self.assertEqual(result["P_filt"].shape, (3, 1, 1))
        self.assertEqual(result["P_pred"].shape, (3, 1, 1))

        # --- finite checks ---
        self.assertTrue(np.all(np.isfinite(result["mu_filt"].numpy())))
        self.assertTrue(np.all(np.isfinite(result["P_filt"].numpy())))
        self.assertTrue(np.all(np.isfinite(result["P_pred"].numpy())))
        self.assertTrue(np.isfinite(result["loglik"].numpy()))

    # --------------------------------------------------------
    # Input validation test (optional but important)
    # --------------------------------------------------------
    def test_non_finite_input_raises(self):

        Y = tf.constant([[np.nan, 2.0]], dtype=self.dtype)

        m0 = tf.constant([0.0], dtype=self.dtype)
        P0 = tf.eye(1, dtype=self.dtype)
        R = tf.eye(1, dtype=self.dtype)

        def predict_fn(x, P, t):
            return x, P

        def update_step(x_pred, P_pred, y, R_mat, t):
            return x_pred, P_pred, y - x_pred, R_mat

        def h(x, t):
            return tf.reduce_sum(x, keepdims=True)

        update_fn = {
            "step": update_step,
            "h": h
        }

        with self.assertRaises(tf.errors.InvalidArgumentError):
            _filter_core(
                Y=Y,
                predict_fn=predict_fn,
                update_fn=update_fn,
                R_mat=R,
                m0=m0,
                P0=P0,
                measurement_type="gaussian",
                dtype=self.dtype
            )



class TestRunKfExperimentsMultivariate(unittest.TestCase):
    def setUp(self):
        self.y = tf.ones((3, 2), dtype=tf.float64)
        self.h = tf.zeros((3, 2), dtype=tf.float64)

        self.fake_filter_out = {
            "mu_filt": tf.zeros((2, 3), dtype=tf.float64),
            "P_filt": tf.zeros((3, 2, 2), dtype=tf.float64),
            "P_pred": tf.zeros((3, 2, 2), dtype=tf.float64),
            "loglik": tf.constant(-1.0, dtype=tf.float64),
        }

        self.fake_bias = tf.zeros((3, 2), dtype=tf.float64)
        self.fake_rmse = tf.ones((3, 2), dtype=tf.float64)

    def _patch_core(self):
        return patch.multiple(
            __name__,
            make_ekf_kernels=DEFAULT,
            make_ukf_kernels=DEFAULT,
            _filter_core=DEFAULT,
            compute_bias_rmse=DEFAULT,
        )

    def _configure_mocks(self, mocks):
        mocks["make_ekf_kernels"].return_value = ("ekf_predict", "ekf_update")
        mocks["make_ukf_kernels"].return_value = ("ukf_predict", "ukf_update")
        mocks["_filter_core"].return_value = self.fake_filter_out
        mocks["compute_bias_rmse"].return_value = (self.fake_bias, self.fake_rmse)

    def test_default_methods_output_structure(self):
        with self._patch_core() as mocks:
            self._configure_mocks(mocks)

            out = run_kf_experiments_multivariate(
                y_tf=self.y,
                h_tf=self.h,
                phi=0.9,
                sigma_eps=0.5,
                sigma_eta=1.0,
                xi=1.0,
            )

        expected = {"EKF_misspec", "UKF_misspec", "EKF_correct", "UKF_correct"}
        self.assertSetEqual(set(out.keys()), expected)

        for method in expected:
            self.assertSetEqual(
                set(out[method].keys()),
                {"bias", "rmse", "mu_filt", "P_filt", "P_pred", "loglik"},
            )

        self.assertEqual(mocks["_filter_core"].call_count, 4)

    def test_methods_argument_runs_only_requested_methods(self):
        with self._patch_core() as mocks:
            self._configure_mocks(mocks)

            out = run_kf_experiments_multivariate(
                y_tf=self.y,
                h_tf=self.h,
                phi=0.9,
                sigma_eps=0.5,
                sigma_eta=1.0,
                xi=1.0,
                methods=["EKF_misspec", "UKF_correct"],
            )

        self.assertSetEqual(set(out.keys()), {"EKF_misspec", "UKF_correct"})
        self.assertEqual(mocks["_filter_core"].call_count, 2)
        self.assertEqual(mocks["make_ekf_kernels"].call_count, 1)
        self.assertEqual(mocks["make_ukf_kernels"].call_count, 1)

    def test_rejects_non_2d_inputs(self):
        with self.assertRaisesRegex(ValueError, "y_tf must be 2D"):
            run_kf_experiments_multivariate(
                y_tf=tf.ones((3,), dtype=tf.float64),
                h_tf=self.h,
                phi=0.9,
                sigma_eps=0.5,
                sigma_eta=1.0,
                xi=1.0,
            )

        with self.assertRaisesRegex(ValueError, "h_tf must be 2D"):
            run_kf_experiments_multivariate(
                y_tf=self.y,
                h_tf=tf.ones((3,), dtype=tf.float64),
                phi=0.9,
                sigma_eps=0.5,
                sigma_eta=1.0,
                xi=1.0,
            )


class TestComputeBiasRMSE(unittest.TestCase):
    """Tests bias/RMSE computation for univariate and multivariate Monte Carlo estimates."""

    def setUp(self):
        self.dtype = tf.float64

    def test_univariate_shapes_and_values(self):
        """Checks univariate bias and RMSE match manual values."""
        h_true = tf.constant([1.0, 2.0], dtype=self.dtype)
        est_mc = tf.constant([
            [1.0, 3.0],
            [3.0, 1.0],
        ], dtype=self.dtype)  # (R,T) = (2,2)

        bias, rmse = compute_bias_rmse(h_true, est_mc)

        expected_bias = tf.constant([[1.0], [0.0]], dtype=self.dtype)
        expected_rmse = tf.constant([[np.sqrt(2.0)], [1.0]], dtype=self.dtype)

        assert_shape(self, bias, (2, 1))
        assert_shape(self, rmse, (2, 1))
        assert_allclose(self, bias, expected_bias, atol=1e-12)
        assert_allclose(self, rmse, expected_rmse, atol=1e-12)

    def test_multivariate_shapes_and_values(self):
        """Checks multivariate bias and RMSE match manual values."""
        h_true = tf.constant([
            [1.0, 2.0],
            [0.0, 1.0],
        ], dtype=self.dtype)  # (T,d) = (2,2)

        est_mc = tf.constant([
            [[1.0, 2.0], [0.0, 1.0]],
            [[3.0, 4.0], [2.0, 3.0]],
        ], dtype=self.dtype)  # (R,T,d) = (2,2,2)

        bias, rmse = compute_bias_rmse(h_true, est_mc)

        expected_bias = tf.constant([
            [1.0, 1.0],
            [1.0, 1.0],
        ], dtype=self.dtype)

        expected_rmse = tf.constant([
            [np.sqrt(2.0), np.sqrt(2.0)],
            [np.sqrt(2.0), np.sqrt(2.0)],
        ], dtype=self.dtype)

        assert_shape(self, bias, (2, 2))
        assert_shape(self, rmse, (2, 2))
        assert_allclose(self, bias, expected_bias, atol=1e-12)
        assert_allclose(self, rmse, expected_rmse, atol=1e-12)

    def test_input_rank_expansion(self):
        """Checks rank-1 h_true and rank-2 est_mc are expanded consistently."""
        h_true = tf.constant([0.0, 0.0, 0.0], dtype=self.dtype)
        est_mc = tf.constant([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ], dtype=self.dtype)

        bias, rmse = compute_bias_rmse(h_true, est_mc)

        expected = tf.constant([[1.0], [2.0], [3.0]], dtype=self.dtype)

        assert_shape(self, bias, (3, 1))
        assert_shape(self, rmse, (3, 1))
        assert_allclose(self, bias, expected, atol=1e-12)
        assert_allclose(self, rmse, expected, atol=1e-12)

    def test_outputs_are_finite(self):
        """Checks outputs are finite for valid finite inputs."""
        h_true = tf.constant([1.0, 2.0, 3.0], dtype=self.dtype)
        est_mc = tf.constant([
            [0.5, 2.5, 3.5],
            [1.5, 1.5, 2.5],
            [1.0, 2.0, 3.0],
        ], dtype=self.dtype)

        bias, rmse = compute_bias_rmse(h_true, est_mc)

        assert_finite(self, bias)
        assert_finite(self, rmse)


# ============================================================
# Tests for _run_bpf_mc_for_fixed_N
# ============================================================
class TestRunBPFMCForFixedN(unittest.TestCase):
    """Tests Monte Carlo aggregation for one fixed particle count."""

    @classmethod
    def setUpClass(cls):
        tf.config.run_functions_eagerly(True)

    @classmethod
    def tearDownClass(cls):
        tf.config.run_functions_eagerly(False)

    def setUp(self):
        self.dtype = tf.float32
        self.T = 4
        self.d = 2
        self.R = 3
        self.N = 10

        self.y_tf = tf.constant([
            [1.0, 2.0],
            [1.5, 2.5],
            [2.0, 3.0],
            [2.5, 3.5],
        ], dtype=self.dtype)

        self.h_tf = tf.constant([
            [0.0, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
        ], dtype=tf.float64)

        self.prop_fn = lambda x: x
        self.log_likelihood_fn = lambda particles, y: tf.zeros((tf.shape(particles)[0],), dtype=self.dtype)

    @patch(__name__ + ".bpf_generic_resampling")
    def test_output_structure_and_shapes_multivariate(self, mock_bpf):
        """Checks aggregated outputs have the expected keys, shapes, and finite values."""
        ests = tf.constant([
            [0.0, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
            [0.6, 0.7],
        ], dtype=self.dtype)  # (T,d)

        ESSs = tf.constant([5.0, 6.0, 7.0, 8.0], dtype=self.dtype)  # (T,)
        total_loglik = tf.constant(-3.5, dtype=self.dtype)

        mock_bpf.return_value = (ests, ESSs, total_loglik)

        out = _run_bpf_mc_for_fixed_N(
            y_tf=self.y_tf,
            h_tf=self.h_tf,
            N=self.N,
            R=self.R,
            prop_fn=self.prop_fn,
            log_likelihood_fn=self.log_likelihood_fn,
            degeneracy_threshold=0.2,
            resampling_fn=lambda p, w: (p, w),
            resample_threshold=False,
            carry_resampled_weights=False,
            dtype=self.dtype,
        )

        expected_keys = {
            "ESS_mean_t", "ESS_min_t", "ESS_norm_mean_t", "ESS_norm_min_t",
            "ESS_mc", "ESS_norm_mc", "part_est_mc", "loglikelihood_mc",
            "bias", "rmse", "ess_mean", "ess_min", "ess_frac_below",
            "ess_n_below", "loglik_mean"
        }
        self.assertSetEqual(set(out.keys()), expected_keys)

        assert_shape(self, out["part_est_mc"], (self.R, self.T, self.d))
        assert_shape(self, out["ESS_mc"], (self.R, self.T))
        assert_shape(self, out["ESS_norm_mc"], (self.R, self.T))
        assert_shape(self, out["loglikelihood_mc"], (self.R,))
        assert_shape(self, out["ESS_mean_t"], (self.T,))
        assert_shape(self, out["ESS_min_t"], (self.T,))
        assert_shape(self, out["ESS_norm_mean_t"], (self.T,))
        assert_shape(self, out["ESS_norm_min_t"], (self.T,))
        assert_shape(self, out["bias"], (self.T, self.d))
        assert_shape(self, out["rmse"], (self.T, self.d))

        for key in ["part_est_mc", "ESS_mc", "ESS_norm_mc", "loglikelihood_mc",
                    "ESS_mean_t", "ESS_min_t", "ESS_norm_mean_t", "ESS_norm_min_t",
                    "bias", "rmse", "ess_mean", "ess_min", "ess_frac_below",
                    "ess_n_below", "loglik_mean"]:
            assert_finite(self, tf.cast(out[key], tf.float64))

        self.assertEqual(mock_bpf.call_count, self.R)

    @patch(__name__ + ".bpf_generic_resampling")
    def test_univariate_h_is_expanded(self, mock_bpf):
        """Checks rank-1 h_true is expanded internally and produces (T,1) bias/RMSE."""
        y_tf = tf.constant([[1.0], [2.0], [3.0]], dtype=self.dtype)
        h_tf = tf.constant([0.0, 0.5, 1.0], dtype=tf.float64)

        ests = tf.constant([[0.0], [0.5], [1.0]], dtype=self.dtype)  # (T,1)
        ESSs = tf.constant([4.0, 5.0, 6.0], dtype=self.dtype)
        total_loglik = tf.constant(-1.0, dtype=self.dtype)

        mock_bpf.return_value = (ests, ESSs, total_loglik)

        out = _run_bpf_mc_for_fixed_N(
            y_tf=y_tf,
            h_tf=h_tf,
            N=8,
            R=2,
            prop_fn=self.prop_fn,
            log_likelihood_fn=self.log_likelihood_fn,
            degeneracy_threshold=0.2,
            resampling_fn=lambda p, w: (p, w),
            resample_threshold=False,
            carry_resampled_weights=False,
            dtype=self.dtype,
        )

        assert_shape(self, out["bias"], (3, 1))
        assert_shape(self, out["rmse"], (3, 1))

    @patch(__name__ + ".bpf_generic_resampling")
    def test_ess_normalization_and_summary_stats(self, mock_bpf):
        """Checks ESS normalization and scalar summaries are computed correctly."""
        ests = tf.zeros((self.T, self.d), dtype=self.dtype)
        ESSs = tf.constant([5.0, 6.0, 7.0, 8.0], dtype=self.dtype)
        total_loglik = tf.constant(-2.0, dtype=self.dtype)

        mock_bpf.return_value = (ests, ESSs, total_loglik)

        out = _run_bpf_mc_for_fixed_N(
            y_tf=self.y_tf,
            h_tf=self.h_tf,
            N=10,
            R=2,
            prop_fn=self.prop_fn,
            log_likelihood_fn=self.log_likelihood_fn,
            degeneracy_threshold=0.7,
            resampling_fn=lambda p, w: (p, w),
            resample_threshold=False,
            carry_resampled_weights=False,
            dtype=self.dtype,
        )

        expected_ESS_norm_mc = tf.constant([
            [0.5, 0.6, 0.7, 0.8],
            [0.5, 0.6, 0.7, 0.8],
        ], dtype=tf.float64)

        assert_allclose(self, out["ESS_norm_mc"], expected_ESS_norm_mc, atol=1e-12)
        self.assertAlmostEqual(float(out["ess_mean"].numpy()), 0.65, places=12)
        self.assertAlmostEqual(float(out["ess_min"].numpy()), 0.5, places=12)
        self.assertAlmostEqual(float(out["ess_frac_below"].numpy()), 0.5, places=12)
        self.assertEqual(int(out["ess_n_below"].numpy()), 4)
        self.assertAlmostEqual(float(out["loglik_mean"].numpy()), -2.0, places=12)

    @patch(__name__ + ".bpf_generic_resampling")
    def test_bias_and_rmse_are_zero_when_estimates_match_truth(self, mock_bpf):
        """Checks bias and RMSE are numerically zero when every replication matches the true state."""
        ests = tf.cast(self.h_tf, self.dtype)
        ESSs = tf.constant([5.0, 6.0, 7.0, 8.0], dtype=self.dtype)
        total_loglik = tf.constant(-2.0, dtype=self.dtype)

        mock_bpf.return_value = (ests, ESSs, total_loglik)

        out = _run_bpf_mc_for_fixed_N(
            y_tf=self.y_tf,
            h_tf=self.h_tf,
            N=self.N,
            R=3,
            prop_fn=self.prop_fn,
            log_likelihood_fn=self.log_likelihood_fn,
            degeneracy_threshold=0.2,
            resampling_fn=lambda p, w: (p, w),
            resample_threshold=False,
            carry_resampled_weights=False,
            dtype=self.dtype,
        )

        assert_allclose(self, out["bias"], tf.zeros_like(out["bias"]), atol=1e-7)
        assert_allclose(self, out["rmse"], tf.zeros_like(out["rmse"]), atol=1e-7)

    @patch(__name__ + ".bpf_generic_resampling")
    def test_arguments_are_forwarded_to_bpf(self, mock_bpf):
        """Checks wrapper forwards arguments to the inner BPF."""
        ests = tf.zeros((self.T, self.d), dtype=self.dtype)
        ESSs = tf.ones((self.T,), dtype=self.dtype)
        total_loglik = tf.constant(0.0, dtype=self.dtype)
        mock_bpf.return_value = (ests, ESSs, total_loglik)

        resampling_fn = lambda p, w: (p, w)

        _run_bpf_mc_for_fixed_N(
            y_tf=self.y_tf,
            h_tf=self.h_tf,
            N=15,
            R=2,
            prop_fn=self.prop_fn,
            log_likelihood_fn=self.log_likelihood_fn,
            degeneracy_threshold=0.2,
            resampling_fn=resampling_fn,
            resample_threshold=True,
            carry_resampled_weights=True,
            dtype=self.dtype,
        )

        self.assertEqual(mock_bpf.call_count, 2)
        kwargs = mock_bpf.call_args.kwargs

        self.assertEqual(kwargs["Np"], 15)
        self.assertTrue(kwargs["resample_threshold"])
        self.assertTrue(kwargs["carry_resampled_weights"])
        self.assertIs(kwargs["resampling_fn"], resampling_fn)
        self.assertIs(kwargs["prop_fn"], self.prop_fn)
        self.assertIs(kwargs["log_likelihood_fn"], self.log_likelihood_fn)



class TestBenchmarkCPU(unittest.TestCase):
    """Checks benchmark_cpu returns the wrapped output and basic timing/memory stats."""

    def test_basic_run(self):
        def f(x):
            return x + 1

        out, stats = benchmark_cpu(f, 3, sample_interval=0.001)

        self.assertEqual(out, 4)
        self.assertIsInstance(stats, dict)

        for key in [
            "runtime_seconds",
            "rss_before_mb",
            "rss_after_mb",
            "rss_peak_mb",
            "rss_delta_mb",
            "rss_peak_increase_mb",
        ]:
            self.assertIn(key, stats)
            self.assertTrue(np.isfinite(stats[key]))


class TestCompareMethodsOneConfig(unittest.TestCase):
    """Checks the one-configuration comparison pipeline builds the expected output."""

    def setUp(self):
        self.y = tf.constant([1.0, 2.0, 3.0], dtype=tf.float64)
        self.h = tf.constant([0.1, 0.2, 0.3], dtype=tf.float64)

        self.fake_bpf_out = {
            "ESS_mean_t": tf.constant([5.0, 6.0, 7.0], dtype=tf.float64),
            "ESS_min_t": tf.constant([4.0, 5.0, 6.0], dtype=tf.float64),
            "ESS_norm_mean_t": tf.constant([0.5, 0.6, 0.7], dtype=tf.float64),
            "ESS_norm_min_t": tf.constant([0.4, 0.5, 0.6], dtype=tf.float64),
            "ESS_mc": tf.constant([[5.0, 6.0, 7.0]], dtype=tf.float64),
            "ESS_norm_mc": tf.constant([[0.5, 0.6, 0.7]], dtype=tf.float64),
            "part_est_mc": tf.constant([[[0.1], [0.2], [0.3]]], dtype=tf.float64),
            "loglikelihood_mc": tf.constant([-1.0], dtype=tf.float64),
            "bias": tf.constant([[0.0], [0.0], [0.0]], dtype=tf.float64),
            "rmse": tf.constant([[0.1], [0.2], [0.3]], dtype=tf.float64),
            "ess_mean": tf.constant(0.6, dtype=tf.float64),
            "ess_min": tf.constant(0.4, dtype=tf.float64),
            "ess_frac_below": tf.constant(0.0, dtype=tf.float64),
            "ess_n_below": tf.constant(0, dtype=tf.int32),
            "loglik_mean": tf.constant(-1.0, dtype=tf.float64),
        }

        self.fake_stats = {
            "runtime_seconds": 0.01,
            "rss_before_mb": 100.0,
            "rss_after_mb": 101.0,
            "rss_peak_mb": 102.0,
            "rss_delta_mb": 1.0,
            "rss_peak_increase_mb": 2.0,
        }

    @patch(__name__ + ".benchmark_cpu")
    @patch(__name__ + ".make_loglik_sv")
    @patch(__name__ + ".make_prop_sv")
    @patch(__name__ + ".SV_model_sim_tf_h")
    def test_basic_output_structure(self, mock_sim, mock_make_prop, mock_make_loglik, mock_bench):
        """Checks the pipeline returns the expected top-level structure and expanded simulated shapes."""
        mock_sim.return_value = {"vY": self.y, "h": self.h}
        mock_make_prop.return_value = object()
        mock_make_loglik.return_value = object()

        methods = ["EKF_misspec", "UKF_misspec", "EKF_correct", "UKF_correct"]

        mock_bench.side_effect = (
            [(self.fake_bpf_out, self.fake_stats) for _ in (5, 10)] +
            [({m: {"ok": True}}, self.fake_stats) for m in methods]
        )

        out = compare_methods_one_config(
            d=1,
            phi=0.9,
            sigma_eps=0.5,
            Np=(5, 10),
            T=3,
            N_MC=1,
        )

        self.assertSetEqual(set(out.keys()), {"sim", "ESS", "metrics", "KF", "benchmark"})
        self.assertSetEqual(set(out["ESS"].keys()), {5, 10})
        self.assertSetEqual(set(out["metrics"].keys()), {5, 10})
        self.assertSetEqual(set(out["KF"].keys()), set(methods))
        self.assertSetEqual(set(out["benchmark"].keys()), {"BPF", "KF"})

        assert_shape(self, out["sim"]["y_tf"], (3, 1))
        assert_shape(self, out["sim"]["h_tf"], (3, 1))

    @patch(__name__ + ".benchmark_cpu")
    @patch(__name__ + ".make_loglik_sv")
    @patch(__name__ + ".make_prop_sv")
    @patch(__name__ + ".SV_model_sim_tf_h")
    def test_benchmark_call_count(self, mock_sim, mock_make_prop, mock_make_loglik, mock_bench):
        """Checks benchmark_cpu is called once per BPF particle size and once per KF method."""
        mock_sim.return_value = {"vY": self.y, "h": self.h}
        mock_make_prop.return_value = object()
        mock_make_loglik.return_value = object()

        methods = ["EKF_misspec", "UKF_misspec", "EKF_correct", "UKF_correct"]

        mock_bench.side_effect = (
            [(self.fake_bpf_out, self.fake_stats) for _ in (5, 10, 20)] +
            [({m: {"ok": True}}, self.fake_stats) for m in methods]
        )

        compare_methods_one_config(
            d=1,
            phi=0.9,
            sigma_eps=0.5,
            Np=(5, 10, 20),
            T=3,
            N_MC=1,
        )

        self.assertEqual(mock_bench.call_count, 7)

    @patch(__name__ + ".benchmark_cpu")
    @patch(__name__ + ".make_loglik_sv")
    @patch(__name__ + ".make_prop_sv")
    @patch(__name__ + ".SV_model_sim_tf_h")
    def test_simulator_called_once(self, mock_sim, mock_make_prop, mock_make_loglik, mock_bench):
        """Checks the simulator is called once with dtype float64."""
        mock_sim.return_value = {"vY": self.y, "h": self.h}
        mock_make_prop.return_value = object()
        mock_make_loglik.return_value = object()

        methods = ["EKF_misspec", "UKF_misspec", "EKF_correct", "UKF_correct"]
        mock_bench.side_effect = (
            [(self.fake_bpf_out, self.fake_stats)] +
            [({m: {"ok": True}}, self.fake_stats) for m in methods]
        )

        compare_methods_one_config(
            d=2,
            phi=0.8,
            sigma_eps=0.3,
            sigma_eta=1.1,
            xi=1.2,
            Np=(5,),
            T=3,
            N_MC=1,
            seed=999,
        )

        mock_sim.assert_called_once_with(
            iT=3,
            phi=0.8,
            sigma_eta=1.1,
            sigma_eps=0.3,
            xi=1.2,
            seed=999,
            d=2,
            dtype=tf.float64,
        )


class TestConfigIntegration(unittest.TestCase):
    """Checks the full config loop produces valid results_all."""

    def test_small_config_loop(self):
        results_all = {}

        for d in [1, 2]:
            for phi in [0.5]:
                for sigma_eps in [1.0]:
                    results_all[(d, phi, sigma_eps)] = compare_methods_one_config(
                        d=d,
                        phi=phi,
                        sigma_eps=sigma_eps,
                        sigma_eta=1.0,
                        xi=1.0,
                        Np=(5,),
                        T=5,
                        N_MC=1,
                        dtype_bpf=tf.float32,
                        dtype_kf=tf.float64,
                    )

        # ---- check keys ----
        expected_keys = {(1, 0.5, 1.0), (2, 0.5, 1.0)}
        self.assertSetEqual(set(results_all.keys()), expected_keys)

        # ---- check structure ----
        for cfg, out in results_all.items():
            with self.subTest(cfg=cfg):

                self.assertSetEqual(
                    set(out.keys()),
                    {"sim", "ESS", "metrics", "KF", "benchmark"}
                )

                # check BPF exists
                self.assertIn(5, out["ESS"])
                self.assertIn(5, out["metrics"])

                # check shapes are plot-safe
                self.assertEqual(out["sim"]["y_tf"].shape.rank, 2)
                self.assertEqual(out["sim"]["h_tf"].shape.rank, 2)

                # check no NaNs
                self.assertTrue(np.all(np.isfinite(out["sim"]["y_tf"].numpy())))
                self.assertTrue(np.all(np.isfinite(out["sim"]["h_tf"].numpy())))
                self.assertTrue(np.all(np.isfinite(out["metrics"][5]["rmse"].numpy())))


class TestResultsAllIntegration(unittest.TestCase):
    """Builds a tiny real results_all object and checks it is plot-ready."""

    def test_small_results_all(self):
        results_all = {}

        for d in [1, 2]:
            for phi in [0.5]:
                for sigma_eps in [1.0]:
                    results_all[(d, phi, sigma_eps)] = compare_methods_one_config(
                        d=d,
                        phi=phi,
                        sigma_eps=sigma_eps,
                        sigma_eta=1.0,
                        xi=1.0,
                        Np=(5,),
                        T=5,
                        N_MC=1,
                        dtype_bpf=tf.float32,
                        dtype_kf=tf.float64,
                    )

        expected_keys = {(1, 0.5, 1.0), (2, 0.5, 1.0)}
        self.assertSetEqual(set(results_all.keys()), expected_keys)

        for cfg, out in results_all.items():
            with self.subTest(cfg=cfg):
                self.assertSetEqual(set(out.keys()), {"sim", "ESS", "metrics", "KF", "benchmark"})
                self.assertIn(5, out["ESS"])
                self.assertIn(5, out["metrics"])

                self.assertTrue(np.all(np.isfinite(out["sim"]["y_tf"].numpy())))
                self.assertTrue(np.all(np.isfinite(out["sim"]["h_tf"].numpy())))
                self.assertTrue(np.all(np.isfinite(out["ESS"][5]["ESS_norm_mean_t"].numpy())))
                self.assertTrue(np.all(np.isfinite(out["metrics"][5]["rmse"].numpy())))



class TestPlotFunctionsUnit(unittest.TestCase):

    def setUp(self):
        # tiny fake results_all compatible with all plots
        self.results_all = {
            (1, 0.5, 1.0): {
                "benchmark": {
                    "BPF": {5: {"rss_peak_mb": 100.0, "runtime_seconds": 0.1}},
                    "KF": {
                        "EKF_misspec": {"rss_peak_mb": 90.0, "runtime_seconds": 0.05},
                        "UKF_misspec": {"rss_peak_mb": 95.0, "runtime_seconds": 0.06},
                        "EKF_correct": {"rss_peak_mb": 85.0, "runtime_seconds": 0.04},
                        "UKF_correct": {"rss_peak_mb": 88.0, "runtime_seconds": 0.045},
                    },
                },
                "ESS": {
                    5: {
#                        "ESS_norm_mean_t": tf.ones(3, tf.float64),
#                        "ESS_norm_min_t": tf.ones(3, tf.float64),
                        "ESS_mean_t": tf.ones(3, tf.float64),
                        "ESS_min_t": tf.ones(3, tf.float64),
                    }
                },
                "metrics": {
                    5: {"rmse": tf.ones((3, 1), tf.float64)}
                },
                "KF": {
                    k: {"rmse": tf.ones((3, 1), tf.float64)}
                    for k in ["EKF_misspec", "UKF_misspec", "EKF_correct", "UKF_correct"]
                },
            }
        }

    def test_plot_benchmark_grouped_runs(self):
        plot_benchmark_grouped(self.results_all)  # just runs

    def test_plot_ess_heatmap_runs(self):
        plot_ESS_heatmap_over_d(self.results_all, N_list=[5], phi=0.5, sigma_eps=1.0)

    def test_plot_ess_heatmap_bad_config_raises(self):
        with self.assertRaises(ValueError):
            plot_ESS_heatmap_over_d(self.results_all, N_list=[5], phi=99, sigma_eps=99)

    def test_plot_metric_over_time_runs(self):
        plot_metric_over_time_algorithms(
            self.results_all,
            phi_fixed=0.5,
            sigma_eps_fixed=1.0,
            N_bpf_list=[5],
        )

    def test_plot_metric_bad_reduce_mode(self):
        with self.assertRaises(ValueError):
            plot_metric_over_time_algorithms(
                self.results_all,
                phi_fixed=0.5,
                sigma_eps_fixed=1.0,
                N_bpf_list=[5],
                reduce_mode="bad",
            )

    def test_plot_rmse_over_dimension_runs(self):
        plot_rmse_over_dimension(
            self.results_all,
            phi_fixed=0.5,
            sigma_eps_fixed=1.0,
            N_list=[5],
        )

    def test_plot_rmse_bad_config(self):
        with self.assertRaises(ValueError):
            plot_rmse_over_dimension(
                self.results_all,
                phi_fixed=99,
                sigma_eps_fixed=99,
                N_list=[5],
            )



class TestPlotFunctionsIntegration(unittest.TestCase):

    def test_plots_run_on_real_small_results(self):
        results_all = {}

        for d in [1]:
            for phi in [0.5]:
                for sigma_eps in [1.0]:
                    results_all[(d, phi, sigma_eps)] = compare_methods_one_config(
                        d=d, phi=phi, sigma_eps=sigma_eps, sigma_eta=1.0, xi=1.0, Np=(5,), T=5, N_MC=1,
                        dtype_bpf=tf.float32,
                        dtype_kf=tf.float64,
                    )

        # just ensure no crash
        plot_benchmark_grouped(results_all)
        plot_ESS_heatmap_over_d(results_all, N_list=[5], phi=0.5, sigma_eps=1.0)
        plot_metric_over_time_algorithms(results_all, 0.5, 1.0, N_bpf_list=[5])
        plot_rmse_over_dimension(results_all, 0.5, 1.0, N_list=[5])



BaseResamplingTest.__unittest_skip__ = True
BaseResamplingTest.__unittest_skip_why__ = "Abstract base class"

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)