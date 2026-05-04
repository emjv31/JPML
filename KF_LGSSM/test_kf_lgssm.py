"""Tests for kf_lgssm.py."""

import os
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from kf_lgssm import (
    KF_multivariate_robust,
    Sim_mLGSSM,
    build_global_metrics_dataframe,
    compute_kf_metrics,
    print_reconstruction_summary,
    reconstruct_and_check,
    run_experiments_KF,
)


TEST_OUTPUT_DIR = os.environ.get("KF_TEST_OUTPUT_DIR", "test_kf_lgssm_outputs")
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


class TestKFMultivariateRobust(unittest.TestCase):
    def test_structure_and_multiple_dimensions(self):
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

            expected = {
                "mu_pred_before", "mu_pred_next", "P_pred", "mu_filt",
                "P_filt", "P_filt_joseph", "v", "F_innov", "K",
                "loglik_vec", "loglik",
            }
            self.assertSetEqual(set(out.keys()), expected)
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

            for key, val in out.items():
                self.assertTrue(tf.reduce_all(tf.math.is_finite(val)).numpy(), msg=f"{key} contains non-finite values")

    def test_invalid_shapes_raise(self):
        Y = tf.random.normal([2, 5], dtype=tf.float64)
        F = tf.eye(3, dtype=tf.float64)
        H = tf.eye(2, dtype=tf.float64)
        Q = tf.eye(3, dtype=tf.float64)
        R = tf.eye(2, dtype=tf.float64)
        m0 = tf.zeros([2], dtype=tf.float64)
        P0 = tf.eye(2, dtype=tf.float64)

        with self.assertRaises(AssertionError):
            KF_multivariate_robust(Y, F, H, Q, R, m0, P0)

    def test_non_psd_covariance_raises(self):
        Y = tf.random.normal([2, 5], dtype=tf.float64)
        F = tf.eye(2, dtype=tf.float64)
        H = tf.eye(2, dtype=tf.float64)
        Q = tf.constant([[1.0, 2.0], [2.0, -1.0]], dtype=tf.float64)
        R = tf.eye(2, dtype=tf.float64)
        m0 = tf.zeros([2], dtype=tf.float64)
        P0 = tf.eye(2, dtype=tf.float64)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            KF_multivariate_robust(Y, F, H, Q, R, m0, P0)

    def test_joseph_raises_errors(self):
        n_x, n_y, T = 2, 2, 5
        F = tf.eye(n_x, dtype=tf.float64)
        H = tf.eye(n_y, dtype=tf.float64)
        Q = tf.eye(n_x, dtype=tf.float64)
        R = tf.eye(n_y, dtype=tf.float64)
        m0 = tf.zeros(n_x, dtype=tf.float64)
        Y = tf.random.normal([n_y, T], dtype=tf.float64)

        cases = [
            dict(Q=tf.eye(n_x, dtype=tf.float64) * np.nan, P0=tf.eye(n_x, dtype=tf.float64)),
            dict(Q=Q, P0=tf.constant([[1.0, 2.0], [2.0, -5.0]], dtype=tf.float64)),
            dict(Q=Q, P0=tf.constant([[1.0, 1.0], [0.0, 1.0]], dtype=tf.float64)),
        ]
        for case in cases:
            with self.assertRaises(tf.errors.InvalidArgumentError):
                KF_multivariate_robust(Y, F, H, case["Q"], R, m0, case["P0"])


class TestSimMultiVLGSSM(unittest.TestCase):
    def test_mLGSSM_errors(self):
        n_x, n_y, T = 3, 2, 5
        m0 = [0.0] * n_x
        F_mat = tf.eye(n_x, dtype=tf.float64)
        H_mat = tf.eye(n_y, n_x, dtype=tf.float64)
        Q_mat = tf.eye(n_x, dtype=tf.float64)
        R_mat = tf.eye(n_y, dtype=tf.float64)
        P0 = tf.eye(n_x, dtype=tf.float64)

        X_true, Y_obs = Sim_mLGSSM(T, m0, F_mat, H_mat, Q_mat, R_mat, P0)
        self.assertTrue(isinstance(X_true, (tf.Tensor, tf.Variable)))
        self.assertTrue(isinstance(Y_obs, (tf.Tensor, tf.Variable)))
        self.assertEqual(X_true.shape, (n_x, T))
        self.assertEqual(Y_obs.shape, (n_y, T))

        cases = [
            dict(Q_mat=tf.eye(n_x, dtype=tf.float64) * np.nan),
            dict(Q_mat=tf.constant([[1.0, 2.0, 0.0], [2.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=tf.float64)),
            dict(F_mat=tf.eye(n_x + 1, dtype=tf.float64)),
            dict(H_mat=tf.eye(n_y + 1, n_x, dtype=tf.float64)),
            dict(T=0),
        ]

        for case in cases:
            args = dict(T=T, m0=m0, F_mat=F_mat, H_mat=H_mat, Q_mat=Q_mat, R_mat=R_mat, P0=P0)
            args.update(case)
            with self.assertRaises(Exception):
                Sim_mLGSSM(**args)


class TestReconstructError(unittest.TestCase):
    def setUp(self):
        self.T = 4
        self.n_x = 2
        self.n_y = 2
        self.mu_pred = tf.ones((self.n_x, self.T), dtype=tf.float64)
        self.mu_filt = tf.ones((self.n_x, self.T), dtype=tf.float64)
        self.v = tf.ones((self.n_y, self.T), dtype=tf.float64)
        self.K = tf.ones((self.T, self.n_x, self.n_y), dtype=tf.float64)
        self.P_pred = tf.ones((self.T, self.n_x, self.n_x), dtype=tf.float64)
        self.P_filt = tf.ones((self.T, self.n_x, self.n_x), dtype=tf.float64)
        self.H_mat = tf.ones((self.n_y, self.n_x), dtype=tf.float64)
        self.R_mat = tf.eye(self.n_y, dtype=tf.float64)
        self.d = self.n_x

    def test_output_structure(self):
        out = reconstruct_and_check(self.mu_pred, self.K, self.v, self.mu_filt, self.P_pred, self.P_filt, self.H_mat, self.R_mat, self.d)
        self.assertEqual(set(out.keys()), {"mu_rec", "P_joseph", "rec_mu", "rec_mu_by_state", "rec_P", "rec_P_diag"})
        self.assertEqual(out["mu_rec"].shape, (self.n_x, self.T))
        self.assertEqual(out["P_joseph"].shape, (self.T, self.n_x, self.n_x))
        self.assertEqual(out["rec_mu"].shape, (self.T,))
        self.assertEqual(out["rec_mu_by_state"].shape, (self.n_x, self.T))
        self.assertEqual(out["rec_P"].shape, (self.T,))
        self.assertEqual(out["rec_P_diag"].shape, (self.T, self.n_x))

    def test_finite_outputs(self):
        out = reconstruct_and_check(self.mu_pred, self.K, self.v, self.mu_filt, self.P_pred, self.P_filt, self.H_mat, self.R_mat, self.d)
        for key in out:
            self.assertTrue(tf.reduce_all(tf.math.is_finite(out[key])))

    def test_detect_rank_errors(self):
        with self.assertRaises(Exception):
            reconstruct_and_check(self.mu_pred, tf.ones((self.T, self.n_x)), self.v, self.mu_filt, self.P_pred, self.P_filt, self.H_mat, self.R_mat, self.d)


class TestComputeKFMetrics(unittest.TestCase):
    def setUp(self):
        self.n_x = 2
        self.n_y = 2
        self.T = 4
        self.X_true = tf.constant([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]], dtype=tf.float64)
        self.mu_filt = tf.constant([[1.1, 1.9, 3.2, 3.8], [2.1, 2.9, 4.1, 4.9]], dtype=tf.float64)
        self.P_filt_joseph = tf.stack([tf.eye(self.n_x, dtype=tf.float64) * 0.25 for _ in range(self.T)])
        self.v = tf.ones((self.n_y, self.T), dtype=tf.float64)
        self.F_innov = tf.stack([tf.eye(self.n_y, dtype=tf.float64) for _ in range(self.T)])
        self.loglik_vec = tf.constant([-1.0, -2.0, -3.0, -4.0], dtype=tf.float64)
        self.kf_out = {
            "mu_filt": self.mu_filt,
            "P_filt_joseph": self.P_filt_joseph,
            "v": self.v,
            "F_innov": self.F_innov,
            "loglik_vec": self.loglik_vec,
            "loglik": tf.reduce_sum(self.loglik_vec),
        }

    def test_output_structure_and_shapes(self):
        out = compute_kf_metrics(self.X_true, self.kf_out)
        expected_keys = {
            "bias_state", "bias_time", "rmse_state", "rmse_time", "coverage95_state",
            "avg_ci_width_state", "nis", "nees", "nis_mean", "nees_mean", "loglik_total", "loglik_cum",
        }
        self.assertEqual(set(out.keys()), expected_keys)
        self.assertEqual(out["bias_state"].shape, (self.n_x,))
        self.assertEqual(out["bias_time"].shape, (self.T,))
        self.assertEqual(out["rmse_state"].shape, (self.n_x,))
        self.assertEqual(out["rmse_time"].shape, (self.T,))
        self.assertEqual(out["coverage95_state"].shape, (self.n_x,))
        self.assertEqual(out["avg_ci_width_state"].shape, (self.n_x,))
        self.assertEqual(out["nis"].shape, (self.T,))
        self.assertEqual(out["nees"].shape, (self.T,))
        self.assertEqual(out["nis_mean"].shape, ())
        self.assertEqual(out["nees_mean"].shape, ())
        self.assertEqual(out["loglik_total"].shape, ())
        self.assertEqual(out["loglik_cum"].shape, (self.T,))

    def test_outputs_are_finite(self):
        out = compute_kf_metrics(self.X_true, self.kf_out)
        for key, val in out.items():
            self.assertTrue(tf.reduce_all(tf.math.is_finite(val)).numpy(), msg=f"{key} contains non-finite values")

    def test_bias_and_rmse_values(self):
        out = compute_kf_metrics(self.X_true, self.kf_out)
        err = self.mu_filt - self.X_true
        np.testing.assert_allclose(out["bias_state"].numpy(), tf.reduce_mean(err, axis=1).numpy(), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(out["bias_time"].numpy(), tf.reduce_mean(err, axis=0).numpy(), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(out["rmse_state"].numpy(), tf.sqrt(tf.reduce_mean(tf.square(err), axis=1)).numpy(), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(out["rmse_time"].numpy(), tf.sqrt(tf.reduce_mean(tf.square(err), axis=0)).numpy(), rtol=1e-10, atol=1e-10)

    def test_coverage_and_ci_width(self):
        out = compute_kf_metrics(self.X_true, self.kf_out)
        np.testing.assert_allclose(out["avg_ci_width_state"].numpy(), np.array([1.96, 1.96]), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(out["coverage95_state"].numpy(), np.array([1.0, 1.0]), rtol=1e-10, atol=1e-10)

    def test_nis_nees_and_loglik(self):
        out = compute_kf_metrics(self.X_true, self.kf_out)
        np.testing.assert_allclose(out["nis"].numpy(), np.array([2.0, 2.0, 2.0, 2.0]), rtol=1e-10, atol=1e-10)
        self.assertAlmostEqual(out["nis_mean"].numpy(), 2.0, places=10)
        np.testing.assert_allclose(out["loglik_cum"].numpy(), np.array([-1.0, -3.0, -6.0, -10.0]), rtol=1e-10, atol=1e-10)
        self.assertAlmostEqual(out["loglik_total"].numpy(), -10.0, places=10)


class TestBuildGlobalMetricsDataFrame(unittest.TestCase):
    def test_dataframe_structure_and_values(self):
        fake_metrics = {
            "bias_state": tf.constant([1.0, 3.0], dtype=tf.float64),
            "rmse_state": tf.constant([2.0, 4.0], dtype=tf.float64),
            "coverage95_state": tf.constant([0.9, 1.0], dtype=tf.float64),
            "avg_ci_width_state": tf.constant([1.0, 1.0], dtype=tf.float64),
            "nis_mean": tf.constant(2.0, dtype=tf.float64),
            "nees_mean": tf.constant(3.0, dtype=tf.float64),
            "loglik_total": tf.constant(-10.0, dtype=tf.float64),
        }
        df = build_global_metrics_dataframe([{"label": "test_exp", "metrics": fake_metrics}])
        self.assertIsInstance(df, pd.DataFrame)
        expected_cols = {"experiment", "bias_global", "rmse_global", "coverage_95_global", "ci_width_global", "nis_mean", "nees_mean", "loglik"}
        self.assertEqual(set(df.columns), expected_cols)
        self.assertEqual(df.loc[0, "experiment"], "test_exp")
        self.assertAlmostEqual(df.loc[0, "bias_global"], 2.0)
        self.assertAlmostEqual(df.loc[0, "rmse_global"], 3.0)
        self.assertAlmostEqual(df.loc[0, "coverage_95_global"], 0.95)
        self.assertAlmostEqual(df.loc[0, "ci_width_global"], 1.0)
        self.assertAlmostEqual(df.loc[0, "nis_mean"], 2.0)
        self.assertAlmostEqual(df.loc[0, "nees_mean"], 3.0)
        self.assertAlmostEqual(df.loc[0, "loglik"], -10.0)


class TestRunExperimentsUnit(unittest.TestCase):
    def setUp(self):
        self.T = 5
        self.n_x = 2
        self.n_y = 2
        self.m0 = tf.zeros((self.n_x,), dtype=tf.float64)
        self.F_mat = tf.eye(self.n_x, dtype=tf.float64)
        self.H_mat = tf.eye(self.n_y, self.n_x, dtype=tf.float64)
        self.Q_mat = 0.1 * tf.eye(self.n_x, dtype=tf.float64)
        self.R_mat = 0.1 * tf.eye(self.n_y, dtype=tf.float64)
        self.P0 = tf.eye(self.n_x, dtype=tf.float64)

    def test_run_executes_and_returns_output(self):
        out = run_experiments_KF(self.T, self.m0, self.F_mat, self.H_mat, self.Q_mat, self.R_mat, self.P0, label="unit_test", seed=123)
        self.assertIn("kf", out)
        self.assertIn("mu_filt", out["kf"])
        self.assertEqual(out["kf"]["mu_filt"].shape, (self.n_x, self.T))

    def test_outputs_finite(self):
        out = run_experiments_KF(self.T, self.m0, self.F_mat, self.H_mat, self.Q_mat, self.R_mat, self.P0, label="unit_test", seed=123)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["kf"]["mu_filt"])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["kf"]["P_pred"])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["X_true"])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["Y_sim"])))


class TestRunExperimentsIntegration(unittest.TestCase):
    def setUp(self):
        self.T = 5
        self.n_x = 2
        self.n_y = 2
        self.m0 = tf.zeros((self.n_x,), dtype=tf.float64)
        self.F_mat = tf.eye(self.n_x, dtype=tf.float64)
        self.H_mat = tf.eye(self.n_y, self.n_x, dtype=tf.float64)
        self.Q_mat = 0.1 * tf.eye(self.n_x, dtype=tf.float64)
        self.R_mat = 0.1 * tf.eye(self.n_y, dtype=tf.float64)
        self.P0 = tf.eye(self.n_x, dtype=tf.float64)

    def test_pipeline_runs_and_consistent(self):
        out = run_experiments_KF(self.T, self.m0, self.F_mat, self.H_mat, self.Q_mat, self.R_mat, self.P0, label="integration_test", seed=123)
        self.assertIn("label", out)
        self.assertIn("X_true", out)
        self.assertIn("Y_sim", out)
        self.assertIn("kf", out)
        self.assertIn("reconstruction", out)
        self.assertIn("metrics", out)
        self.assertEqual(out["X_true"].shape, (self.n_x, self.T))
        self.assertEqual(out["Y_sim"].shape, (self.n_y, self.T))
        self.assertEqual(out["kf"]["mu_filt"].shape, (self.n_x, self.T))
        self.assertEqual(out["kf"]["P_pred"].shape, (self.T, self.n_x, self.n_x))
        self.assertEqual(out["kf"]["P_filt_joseph"].shape, (self.T, self.n_x, self.n_x))
        self.assertEqual(out["reconstruction"]["mu_rec"].shape, (self.n_x, self.T))
        self.assertEqual(out["reconstruction"]["P_joseph"].shape, (self.T, self.n_x, self.n_x))
        self.assertEqual(out["reconstruction"]["rec_mu"].shape, (self.T,))
        self.assertEqual(out["reconstruction"]["rec_mu_by_state"].shape, (self.n_x, self.T))
        self.assertEqual(out["reconstruction"]["rec_P"].shape, (self.T,))
        self.assertEqual(out["reconstruction"]["rec_P_diag"].shape, (self.T, self.n_x))
        self.assertEqual(out["metrics"]["bias_state"].shape, (self.n_x,))
        self.assertEqual(out["metrics"]["rmse_state"].shape, (self.n_x,))
        self.assertEqual(out["metrics"]["rmse_time"].shape, (self.T,))
        self.assertEqual(out["metrics"]["coverage95_state"].shape, (self.n_x,))
        self.assertEqual(out["metrics"]["avg_ci_width_state"].shape, (self.n_x,))
        self.assertEqual(out["metrics"]["nis"].shape, (self.T,))
        self.assertEqual(out["metrics"]["nees"].shape, (self.T,))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["X_true"])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["Y_sim"])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["kf"]["mu_filt"])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["kf"]["P_pred"])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["kf"]["P_filt_joseph"])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["reconstruction"]["rec_mu"])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["reconstruction"]["rec_P"])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["metrics"]["rmse_state"])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["metrics"]["nees"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
