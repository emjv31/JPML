import sys
import os

import unittest

import numpy as np
import tensorflow as tf
import pandas as pd
import math, time, warnings
from unittest.mock import patch, DEFAULT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap



# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
try:
    THIS_DIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    THIS_DIR = os.getcwd()

PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SV_DIR = os.path.join(PROJECT_ROOT, "EKF_UKF_SV_BPF")

for _path in (THIS_DIR, PROJECT_ROOT, SV_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)



from EKF_UKF_SV_BPF.tests_multiv_sv import BaseResamplingTest
BaseResamplingTest.__test__ = False
BaseResamplingTest.__unittest_skip__ = True
BaseResamplingTest.__unittest_skip_why__ = "Abstract base class"

# ----------------------------------
# IMPORT ALL FUNCTIONS FOR TESTING
# ---------------------------------

from replicate_Li_filters import *
from Hu_filters_utils import *
from differentiablePF_resampling import *
from replicate_Dai import *

from simulator_model_comps import *
from metric_utils import *
from test_helpers import *


from execution_pipeline import (
    run_edh_ledh_pipeline,
    plot_gradient_conditioning_pip,
    plot_flow_norm_boxplot_pip,
    plot_spectral_norms_pip,
    covariance_diag_over_time,
    plot_posterior_spread_comparison,
)


#### HELPERS
def make_simple_likelihood(dtype):
    def log_likelihood_fn(particles, y):
        diff = tf.cast(particles, dtype) - tf.cast(y, dtype)
        return -tf.reduce_sum(diff ** 2, axis=1)
    return log_likelihood_fn


def H_jac_tf(x, t=None):
    d = tf.shape(x)[0]
    return tf.eye(d, dtype=x.dtype)


def H_jac_t_tf(x, t=None):
    return H_jac_t(x, t)


def hu_matrix_kernel(X, alpha, Qinv):
    return make_hu_matrix_kernel(Qinv, dtype=X.dtype)(X, alpha)


###################################
#####  UNIT TESTS CORE FILTERS
###################################

class TestKFKernels(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64

        self.F = tf.constant([[1.0, 1.0],
                              [0.0, 1.0]], dtype=self.dtype)

        self.H = tf.constant([[1.0, 0.0]], dtype=self.dtype)

        self.Q = 0.1 * tf.eye(2, dtype=self.dtype)
        self.R = tf.eye(1, dtype=self.dtype)

    def test_kf_kernels_basic(self):

        predict_fn, update_fn = make_kf_kernels(self.F, self.H, self.Q)

        x = tf.constant([0.0, 1.0], dtype=self.dtype)
        P = tf.eye(2, dtype=self.dtype)
        y = tf.constant([1.0], dtype=self.dtype)

        # --- predict ---
        x_pred, P_pred = predict_fn(x, P, t=0)

        self.assertEqual(x_pred.shape, (2,))
        self.assertEqual(P_pred.shape, (2, 2))

        # --- update ---
        x_filt, P_filt, v, S = update_fn["step"](x_pred, P_pred, y, self.R, t=0)

        self.assertEqual(x_filt.shape, (2,))
        self.assertEqual(P_filt.shape, (2, 2))
        self.assertEqual(v.shape, (1,))
        self.assertEqual(S.shape, (1, 1))

        # --- finite ---
        self.assertTrue(np.all(np.isfinite(x_filt.numpy())))
        self.assertTrue(np.all(np.isfinite(P_filt.numpy())))



class TestKFFilterWrappers(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)

        self.dtype = tf.float64
        self.T = 4
        self.d = 2

        # _filter_core accepts (d, T)
        self.Y = tf.random.normal((self.d, self.T), dtype=self.dtype)

        self.m0 = tf.zeros(self.d, dtype=self.dtype)
        self.P0 = tf.eye(self.d, dtype=self.dtype)

        self.Q = tf.eye(self.d, dtype=self.dtype) * 0.1
        self.R = tf.eye(self.d, dtype=self.dtype) * 0.1

        self.alpha = 0.9

        self.F_mat = self.alpha * tf.eye(self.d, dtype=self.dtype)
        self.H_mat = tf.eye(self.d, dtype=self.dtype)

        self.F = lambda x, t: self.alpha * x
        self.H = lambda x, t: x

        self.F_jac = lambda x, t: self.alpha * tf.eye(self.d, dtype=self.dtype)
        self.H_jac = lambda x, t: tf.eye(self.d, dtype=self.dtype)

    def test_run_ekf_wrap(self):
        out = run_ekf_wrap(
            Y=self.Y,
            m0=self.m0,
            P0=self.P0,
            Q=self.Q,
            R=self.R,
            F=self.F,
            H=self.H,
            F_jac=self.F_jac,
            H_jac=self.H_jac,
            measurement_type="gaussian"
        )

        self.assertIsInstance(out, dict)
        self.assertIn("mu_filt", out)
        self.assertIn("P_filt", out)
        self.assertIn("P_pred", out)
        self.assertIn("loglik", out)

        ests = tf.transpose(out["mu_filt"])   # (d,T) -> (T,d)
        assert_valid_output(self, ests, None, self.T, self.d)
        assert_valid_loglik(self, out["loglik"])

    def test_run_kf_wrap(self):
        ests = run_kf_wrap(
            Y=self.Y,
            m0=self.m0,
            P0=self.P0,
            Q=self.Q,
            R=self.R,
            F_mat=self.F_mat,
            H_mat=self.H_mat,
            measurement_type="gaussian"
        )

        # if run_kf_wrap also returns (d, T), transpose it
        if isinstance(ests, dict):
            if "mu_filt" in ests:
                ests = tf.transpose(ests["mu_filt"])
            else:
                raise KeyError(f"Unexpected keys in run_kf_wrap output: {list(ests.keys())}")
        elif len(ests.shape) == 2 and ests.shape[0] == self.d and ests.shape[1] == self.T:
            ests = tf.transpose(ests)

        assert_valid_output(self, ests, None, self.T, self.d)

    def test_run_ukf_wrap(self):
        ests = run_ukf_wrap(
            Y=self.Y,
            m0=self.m0,
            P0=self.P0,
            Q=self.Q,
            R=self.R,
            F=self.F,
            H=self.H,
            measurement_type="gaussian",
            dtype=self.dtype
        )

        if isinstance(ests, dict):
            if "mu_filt" in ests:
                ests = tf.transpose(ests["mu_filt"])
            else:
                raise KeyError(f"Unexpected keys in run_ukf_wrap output: {list(ests.keys())}")
        elif len(ests.shape) == 2 and ests.shape[0] == self.d and ests.shape[1] == self.T:
            ests = tf.transpose(ests)

        assert_valid_output(self, ests, None, self.T, self.d)

    def test_convergence_kf(self):
        ekf_out = run_ekf_wrap(
            Y=self.Y,
            m0=self.m0,
            P0=self.P0,
            Q=self.Q,
            R=self.R,
            F=self.F,
            H=self.H,
            F_jac=self.F_jac,
            H_jac=self.H_jac,
            measurement_type="gaussian"
        )
        ekf = tf.transpose(ekf_out["mu_filt"])

        ukf = run_ukf_wrap(
            Y=self.Y,
            m0=self.m0,
            P0=self.P0,
            Q=self.Q,
            R=self.R,
            F=self.F,
            H=self.H,
            measurement_type="gaussian",
            dtype=self.dtype
        )
        if isinstance(ukf, dict):
            ukf = tf.transpose(ukf["mu_filt"])
        elif len(ukf.shape) == 2 and ukf.shape[0] == self.d and ukf.shape[1] == self.T:
            ukf = tf.transpose(ukf)

        kf = run_kf_wrap(
            Y=self.Y,
            m0=self.m0,
            P0=self.P0,
            Q=self.Q,
            R=self.R,
            F_mat=self.F_mat,
            H_mat=self.H_mat,
            measurement_type="gaussian"
        )
        if isinstance(kf, dict):
            kf = tf.transpose(kf["mu_filt"])
        elif len(kf.shape) == 2 and kf.shape[0] == self.d and kf.shape[1] == self.T:
            kf = tf.transpose(kf)

        np.testing.assert_allclose(ekf.numpy(), kf.numpy(), atol=1e-5)
        np.testing.assert_allclose(ukf.numpy(), kf.numpy(), atol=1e-5)

    def test_invalid_input_raises(self):
        bad_Y = tf.random.normal((self.d,), dtype=self.dtype)

        def bad_call():
            run_kf_wrap(
                Y=bad_Y,
                m0=self.m0,
                P0=self.P0,
                Q=self.Q,
                R=self.R,
                F_mat=self.F_mat,
                H_mat=self.H_mat
            )

        assert_raises(self, bad_call, ValueError)



class TestESRFFilter(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.Np = 20
        self.T = 5
        self.d = 2

        self.measurements = tf.ones((5, 2), dtype=self.dtype)
        self.measurements = tf.ones((self.T, self.d), dtype=self.dtype)
        self.Q = 0.1 * tf.eye(self.d, dtype=self.dtype) 
        self.R = tf.eye(self.d, dtype=self.dtype) 

        self.F_func = lambda x: x
        self.H_func = lambda x: x

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------
    def _call_filter(self, **kwargs):
        params = dict(
            measurements=self.measurements,
            Np=self.Np,
            F_func=self.F_func,
            Q=self.Q,
            H_func=self.H_func,
            R=self.R,
            dtype=self.dtype
        )
        params.update(kwargs)
        return esrf_filter(**params)

    # --------------------------------------------------------
    # Basic functionality
    # --------------------------------------------------------
    def test_shapes_and_finite(self):

        ests = self._call_filter()
        assert_valid_output(self, ests, None, self.T, self.d)

    # --------------------------------------------------------
    # Runs
    # --------------------------------------------------------
    def test_runs(self):
        ests = self._call_filter()
        self.assertIsNotNone(ests)

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

        with self.assertRaises(ValueError):
            esrf_filter(
                self.measurements,
                1,  # bad value
                self.F_func,
                self.Q,
                self.H_func,
                self.R,
                dtype=self.dtype
            )


    def test_invalid_measurement_type(self):
        assert_raises(self, lambda: self._call_filter(measurement_type="invalid"), ValueError)

    def test_non_tensor_input(self):
        with self.assertRaises(TypeError):
            self._call_filter(measurements=np.ones((5, 2)))

    def test_invalid_Q(self):
        with self.assertRaises(ValueError):
            self._call_filter(Q=tf.ones((2, 3), dtype=self.dtype))

    def test_invalid_R(self):
        with self.assertRaises(ValueError):
            self._call_filter(R=tf.ones((2, 3), dtype=self.dtype))

    def test_invalid_measurements_shape(self):
        with self.assertRaises(ValueError):
            self._call_filter(measurements=tf.ones((5,), dtype=self.dtype))

    # --------------------------------------------------------
    # Function validation (minimal but correct)
    # --------------------------------------------------------
    def test_non_callable_F_H(self):

        with self.assertRaises(TypeError):
            self._call_filter(F_func=None)

        with self.assertRaises(TypeError):
            self._call_filter(H_func=None)

    def test_invalid_F_output_shape(self):

        def bad_F(x):
            return tf.ones((self.Np, 3), dtype=self.dtype)

        with self.assertRaises(Exception):  
            self._call_filter(F_func=bad_F)

    def test_invalid_H_output_shape(self):

        def bad_H(x):
            return tf.ones((self.Np, 3), dtype=self.dtype)

        with self.assertRaises(Exception):
            self._call_filter(H_func=bad_H)



# HELPER UPF
def make_upf_functions(Np, d, dtype):
    def transition_mean_fn(sigma, alpha_dyn):
        return sigma  # identity

    def transition_logpdf_fn(x_new, x_prev, L):
        return tf.zeros((Np,), dtype=dtype)

    def log_likelihood_fn(particles, y):
        return tf.zeros((Np,), dtype=dtype)

    return transition_mean_fn, transition_logpdf_fn, log_likelihood_fn



class TestUPFParticleFilter(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.Np = 20
        self.T = 4
        self.d = 2

        self.Y = tf.ones((self.T, self.d), dtype=self.dtype)
        self.Sigma = tf.eye(self.d, dtype=self.dtype)
        self.Qp = tf.eye(self.d, dtype=self.dtype)

        self.alpha_dyn = tf.ones(self.d, dtype=self.dtype)
        self.gamma = tf.zeros(self.d, dtype=self.dtype)
        self.nu = tf.constant(5.0, dtype=self.dtype)

        (
            self.transition_mean_fn,
            self.transition_logpdf_fn,
            self.log_likelihood_fn
        ) = make_upf_functions(self.Np, self.d, self.dtype)

    # --------------------------------------------------------
    def test_basic_run(self):

        ests, ESSs = upf_filter(
            self.Y,
            self.Np,
            self.alpha_dyn,
            self.Sigma,
            self.gamma,
            self.nu,
            self.Qp,
            self.transition_logpdf_fn,
            self.log_likelihood_fn,
            self.transition_mean_fn,
            dtype=self.dtype
        )

        assert_valid_output(self, ests, ESSs, self.T, self.d)
        assert_valid_ess(self, ESSs, self.Np)

    # --------------------------------------------------------
    def test_numpy_input_not_allowed(self):

        Y = np.ones((self.T, self.d))

        assert_raises(
            self,
            lambda: upf_filter(
                Y,
                self.Np,
                self.alpha_dyn,
                self.Sigma,
                self.gamma,
                self.nu,
                self.Qp,
                self.transition_logpdf_fn,
                self.log_likelihood_fn,
                self.transition_mean_fn
            ),
            error=TypeError
        )

    # --------------------------------------------------------
    def test_invalid_Np(self):

        assert_raises(
            self,
            lambda: upf_filter(
                self.Y,
                1,
                self.alpha_dyn,
                self.Sigma,
                self.gamma,
                self.nu,
                self.Qp,
                self.transition_logpdf_fn,
                self.log_likelihood_fn,
                self.transition_mean_fn
            ),
            error=ValueError
        )

        assert_raises(
            self,
            lambda: upf_filter(
                self.Y,
                "20",
                self.alpha_dyn,
                self.Sigma,
                self.gamma,
                self.nu,
                self.Qp,
                self.transition_logpdf_fn,
                self.log_likelihood_fn,
                self.transition_mean_fn
            ),
            error=ValueError
        )

    # --------------------------------------------------------
    def test_invalid_Y_shape(self):

        Y = tf.ones((self.T,), dtype=self.dtype)

        assert_raises(
            self,
            lambda: upf_filter(
                Y,
                self.Np,
                self.alpha_dyn,
                self.Sigma,
                self.gamma,
                self.nu,
                self.Qp,
                self.transition_logpdf_fn,
                self.log_likelihood_fn,
                self.transition_mean_fn
            ),
            error=ValueError
        )

    # --------------------------------------------------------
    def test_invalid_covariance(self):

        bad_Sigma = tf.ones((self.d, self.d + 1), dtype=self.dtype)

        assert_raises(
            self,
            lambda: upf_filter(
                self.Y,
                self.Np,
                self.alpha_dyn,
                bad_Sigma,
                self.gamma,
                self.nu,
                self.Qp,
                self.transition_logpdf_fn,
                self.log_likelihood_fn,
                self.transition_mean_fn
            ),
            error=ValueError
        )

    # --------------------------------------------------------
    def test_non_callable_functions(self):

        assert_raises(
            self,
            lambda: upf_filter(
                self.Y,
                self.Np,
                self.alpha_dyn,
                self.Sigma,
                self.gamma,
                self.nu,
                self.Qp,
                None,
                None,
                None
            ),
            error=TypeError
        )

    # --------------------------------------------------------
    def test_ESS_properties(self):

        _, ESSs = upf_filter(
            self.Y,
            self.Np,
            self.alpha_dyn,
            self.Sigma,
            self.gamma,
            self.nu,
            self.Qp,
            self.transition_logpdf_fn,
            self.log_likelihood_fn,
            self.transition_mean_fn,
            dtype=self.dtype
        )

        assert_valid_ess(self, ESSs, self.Np)


class TestGSMCParticleFilter(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.Np = 20
        self.T = 4
        self.d = 2

        self.Y = tf.ones((self.T, self.d), dtype=self.dtype)

        self.prop_fn, self.log_likelihood_fn = make_simple_functions(self.Np, self.dtype)

    # --------------------------------------------------------
    def test_basic_run(self):

        ests, ESSs = gsmc_general(
            self.Y,
            self.Np,
            self.prop_fn,
            self.log_likelihood_fn,
            dtype=self.dtype
        )

        assert_valid_output(self, ests, ESSs, self.T, self.d)
        assert_valid_ess(self, ESSs, self.Np)

    # --------------------------------------------------------
    def test_numpy_input_not_allowed(self):

        Y = np.ones((self.T, self.d))

        assert_raises(
            self,
            lambda: gsmc_general(
                Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn
            ),
            error=TypeError
        )

    # --------------------------------------------------------
    def test_invalid_Np(self):

        assert_raises(
            self,
            lambda: gsmc_general(
                self.Y,
                1,
                self.prop_fn,
                self.log_likelihood_fn
            ),
            error=ValueError
        )

        assert_raises(
            self,
            lambda: gsmc_general(
                self.Y,
                "20",
                self.prop_fn,
                self.log_likelihood_fn
            ),
            error=ValueError
        )

    # --------------------------------------------------------
    def test_invalid_Y_shape(self):

        Y = tf.ones((self.T,), dtype=self.dtype)

        assert_raises(
            self,
            lambda: gsmc_general(
                Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn
            ),
            error=ValueError
        )

    # --------------------------------------------------------
    def test_empty_Y(self):

        Y = tf.ones((0, self.d), dtype=self.dtype)

        assert_raises(
            self,
            lambda: gsmc_general(
                Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn
            ),
            error=ValueError
        )

    # --------------------------------------------------------
    def test_non_callable_functions(self):

        assert_raises(
            self,
            lambda: gsmc_general(
                self.Y,
                self.Np,
                None,
                None
            ),
            error=TypeError
        )

    # --------------------------------------------------------
    def test_resampling_fn_not_callable(self):

        assert_raises(
            self,
            lambda: gsmc_general(
                self.Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn,
                resampling_fn=None
            ),
            error=TypeError
        )
    # --------------------------------------------------------    
    def test_invalid_likelihood_output(self):

        def bad_log_likelihood(particles, y):
            return tf.ones((self.Np, 2), dtype=self.dtype)  # wrong shape

        with self.assertRaises(Exception):
            gsmc_general(
                self.Y,
                self.Np,
                self.prop_fn,
                bad_log_likelihood,
                dtype=self.dtype
            )
    # --------------------------------------------------------
    def test_ESS_properties(self):

        _, ESSs = gsmc_general(
            self.Y,
            self.Np,
            self.prop_fn,
            self.log_likelihood_fn,
            dtype=self.dtype
        )

        assert_valid_ess(self, ESSs, self.Np)



class TestBPFBlock(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.Np = 20
        self.T = 6
        self.d = 2

        self.Y = tf.ones((self.T, self.d), dtype=self.dtype)

        self.prop_fn, self.log_likelihood_fn = make_simple_functions(
            self.Np, self.dtype
        )

    # --------------------------------------------------------
    def test_basic_run(self):

        ests, ESSs = bpf_block(
            self.Y,
            self.Np,
            self.prop_fn,
            self.log_likelihood_fn,
            dtype=self.dtype
        )

        assert_valid_output(self, ests, ESSs, self.T, self.d)
        assert_valid_ess(self, ESSs, self.Np)

    # --------------------------------------------------------
    def test_numpy_input_not_allowed(self):

        Y = np.ones((self.T, self.d))

        assert_raises(
            self,
            lambda: bpf_block(
                Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn
            ),
            error=TypeError
        )

    # --------------------------------------------------------
    def test_invalid_Np(self):

        assert_raises(
            self,
            lambda: bpf_block(
                self.Y,
                1,
                self.prop_fn,
                self.log_likelihood_fn
            ),
            error=ValueError
        )

        assert_raises(
            self,
            lambda: bpf_block(
                self.Y,
                "20",
                self.prop_fn,
                self.log_likelihood_fn
            ),
            error=ValueError
        )

    # --------------------------------------------------------
    def test_invalid_Y_shape(self):

        Y = tf.ones((self.T,), dtype=self.dtype)

        assert_raises(
            self,
            lambda: bpf_block(
                Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn
            ),
            error=ValueError
        )

    # --------------------------------------------------------
    def test_empty_Y(self):

        Y = tf.ones((0, self.d), dtype=self.dtype)

        assert_raises(
            self,
            lambda: bpf_block(
                Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn
            ),
            error=ValueError
        )

    # --------------------------------------------------------
    def test_invalid_block_size(self):

        assert_raises(
            self,
            lambda: bpf_block(
                self.Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn,
                block_size=0
            ),
            error=ValueError
        )

    # --------------------------------------------------------
    def test_non_callable_functions(self):

        assert_raises(
            self,
            lambda: bpf_block(
                self.Y,
                self.Np,
                None,
                self.log_likelihood_fn
            ),
            error=TypeError
        )

        assert_raises(
            self,
            lambda: bpf_block(
                self.Y,
                self.Np,
                self.prop_fn,
                None
            ),
            error=TypeError
        )

    # --------------------------------------------------------
    def test_resampling_fn_not_callable(self):

        assert_raises(
            self,
            lambda: bpf_block(
                self.Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn,
                resampling_fn=None
            ),
            error=TypeError
        )

    # --------------------------------------------------------
    def test_ESS_properties(self):

        _, ESSs = bpf_block(
            self.Y,
            self.Np,
            self.prop_fn,
            self.log_likelihood_fn,
            dtype=self.dtype
        )

        assert_valid_ess(self, ESSs, self.Np)



class TestHMCHelper(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.Np = 10
        self.d = 2

        self.particles = tf.zeros((self.Np, self.d), dtype=self.dtype)
        self.y_t = tf.ones((self.d,), dtype=self.dtype)

        self.log_likelihood_fn = make_simple_likelihood(self.dtype)

    # --------------------------------------------------------
    def test_basic_run(self):

        out = helper_hmc_rej_vectorized(
            self.particles,
            self.y_t,
            self.log_likelihood_fn,
            dtype=self.dtype
        )

        self.assertEqual(out.shape, (self.Np, self.d))
        self.assertTrue(np.all(np.isfinite(out.numpy())))

    # --------------------------------------------------------
    def test_output_changes_particles(self):
        # not strict, but ensures something happens

        out = helper_hmc_rej_vectorized(
            self.particles,
            self.y_t,
            self.log_likelihood_fn,
            dtype=self.dtype
        )

        # articles should not all be exactly zero
        self.assertFalse(np.allclose(out.numpy(), self.particles.numpy()))

    # --------------------------------------------------------
    def test_invalid_likelihood_output(self):

        def bad_llk(particles, y):
            return tf.ones((self.Np, 2), dtype=self.dtype)

        with self.assertRaises(Exception):
            helper_hmc_rej_vectorized(
                self.particles,
                self.y_t,
                bad_llk,
                dtype=self.dtype
            )

    # --------------------------------------------------------
    def test_stability_no_nan_inf(self):

        out = helper_hmc_rej_vectorized(
            self.particles,
            self.y_t,
            self.log_likelihood_fn,
            leapfrog_steps=3,
            epsilon=0.1,
            dtype=self.dtype
        )

        self.assertFalse(np.any(np.isnan(out.numpy())))
        self.assertFalse(np.any(np.isinf(out.numpy())))



class TestSMHMCParticleFilter(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.Np = 20
        self.T = 4
        self.d = 2

        self.Y = tf.ones((self.T, self.d), dtype=self.dtype)

        self.prop_fn, self.log_likelihood_fn = make_simple_functions(
            self.Np, self.dtype
        )

    # --------------------------------------------------------
    def test_basic_run(self):

        ests, ESSs = smhmc_helper(
            self.Y,
            self.Np,
            self.prop_fn,
            self.log_likelihood_fn,
            dtype=self.dtype
        )

        assert_valid_output(self, ests, ESSs, self.T, self.d)
        assert_valid_ess(self, ESSs, self.Np)

    # --------------------------------------------------------
    def test_numpy_input_conversion(self):

        Y = np.ones((self.T, self.d))

        ests, ESSs = smhmc_helper(
            Y,
            self.Np,
            self.prop_fn,
            self.log_likelihood_fn,
            dtype=self.dtype
        )

        assert_valid_output(self, ests, ESSs, self.T, self.d)

    # --------------------------------------------------------
    def test_invalid_Np(self):

        assert_raises(
            self,
            lambda: smhmc_helper(
                self.Y,
                1,
                self.prop_fn,
                self.log_likelihood_fn
            ),
            ValueError
        )

        assert_raises(
            self,
            lambda: smhmc_helper(
                self.Y,
                "20",
                self.prop_fn,
                self.log_likelihood_fn
            ),
            ValueError
        )

    # --------------------------------------------------------
    def test_invalid_Y_shape(self):

        Y = tf.ones((self.T,), dtype=self.dtype)

        assert_raises(
            self,
            lambda: smhmc_helper(
                Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn
            ),
            ValueError
        )

    # --------------------------------------------------------
    def test_empty_Y(self):

        Y = tf.ones((0, self.d), dtype=self.dtype)

        assert_raises(
            self,
            lambda: smhmc_helper(
                Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn
            ),
            ValueError
        )

    # --------------------------------------------------------
    def test_non_callable_functions(self):

        assert_raises(
            self,
            lambda: smhmc_helper(
                self.Y,
                self.Np,
                None,
                None
            ),
            TypeError
        )

    # --------------------------------------------------------
    def test_resampling_fn_not_callable(self):

        assert_raises(
            self,
            lambda: smhmc_helper(
                self.Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn,
                resampling_fn=None
            ),
            TypeError
        )

    # --------------------------------------------------------
    def test_invalid_hmc_params(self):

        assert_raises(
            self,
            lambda: smhmc_helper(
                self.Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn,
                leapfrog_steps=0
            ),
            ValueError
        )

        assert_raises(
            self,
            lambda: smhmc_helper(
                self.Y,
                self.Np,
                self.prop_fn,
                self.log_likelihood_fn,
                epsilon=0
            ),
            ValueError
        )

    # --------------------------------------------------------
    def test_invalid_likelihood_output(self):

        def bad_log_likelihood(particles, y):
            return tf.ones((self.Np, 2), dtype=self.dtype)

        with self.assertRaises(Exception):
            smhmc_helper(
                self.Y,
                self.Np,
                self.prop_fn,
                bad_log_likelihood,
                dtype=self.dtype
            )

    # --------------------------------------------------------
    def test_ESS_properties(self):

        _, ESSs = smhmc_helper(
            self.Y,
            self.Np,
            self.prop_fn,
            self.log_likelihood_fn,
            dtype=self.dtype
        )

        assert_valid_ess(self, ESSs, self.Np)


class TestParticleFlowUpdate(unittest.TestCase):
    """Unit tests for EDH/LEDH particle-flow update with beta schedules."""

    def setUp(self):
        self.dtype = tf.float64
        self.Np = 10
        self.d = 2

        self.particles = tf.ones((self.Np, self.d), dtype=self.dtype)
        self.P_pred = tf.eye(self.d, dtype=self.dtype)
        self.R = tf.eye(self.d, dtype=self.dtype)
        self.z = tf.constant([0.5, 1.5], dtype=self.dtype)

        def h_func(x):
            return tf.cast(x, self.dtype)

        def jacobian_func(x, t=None):
            return tf.eye(self.d, dtype=self.dtype)

        self.h_func = h_func
        self.jacobian_func = jacobian_func

    def assert_finite_tf(self, x):
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(x))))

    def assert_allclose_tf(self, a, b, atol=1e-8, rtol=1e-6):
        a = tf.cast(a, self.dtype)
        b = tf.cast(b, self.dtype)
        diff = tf.abs(a - b)
        tol = atol + rtol * tf.abs(b)
        self.assertTrue(bool(tf.reduce_all(diff <= tol)))

    def _run_update(self, flow_type="EDH", diagnostics=False, **kwargs):
        return particle_flow_pf_update_beta_batch(
            self.particles,
            self.P_pred,
            self.R,
            self.z,
            flow_type=flow_type,
            h_func=self.h_func,
            jacobian_func=self.jacobian_func,
            diagnostics=diagnostics,
            dtype=self.dtype,
            **kwargs
        )

    def test_basic_run_edh(self):
        eta, theta = self._run_update(flow_type="EDH", diagnostics=False)

        self.assertEqual(eta.shape, (self.Np, self.d))
        self.assertEqual(theta.shape, ())
        self.assert_finite_tf(eta)
        self.assert_finite_tf(theta)

    def test_basic_run_ledh(self):
        eta, theta = self._run_update(flow_type="LEDH", diagnostics=False)

        self.assertEqual(eta.shape, (self.Np, self.d))
        self.assertEqual(theta.shape, ())
        self.assert_finite_tf(eta)
        self.assert_finite_tf(theta)

    def test_diagnostics_output_edh(self):
        eta, theta, logdet_J, flow_norm, spec_J, cond_J = self._run_update(
            flow_type="EDH",
            diagnostics=True
        )

        self.assertEqual(eta.shape, (self.Np, self.d))
        self.assertEqual(theta.shape, ())
        self.assertEqual(logdet_J.shape, ())
        self.assertEqual(flow_norm.shape, (self.Np,))
        self.assertEqual(spec_J.shape, ())
        self.assertEqual(cond_J.shape, ())

        self.assert_finite_tf(eta)
        self.assert_finite_tf(theta)
        self.assert_finite_tf(logdet_J)
        self.assert_finite_tf(flow_norm)
        self.assert_finite_tf(spec_J)
        self.assert_finite_tf(cond_J)

    def test_diagnostics_output_ledh(self):
        eta, theta, logdet_J, flow_norm, spec_J, cond_J = self._run_update(
            flow_type="LEDH",
            diagnostics=True
        )

        self.assertEqual(eta.shape, (self.Np, self.d))
        self.assertEqual(flow_norm.shape, (self.Np,))

        self.assert_finite_tf(eta)
        self.assert_finite_tf(theta)
        self.assert_finite_tf(logdet_J)
        self.assert_finite_tf(flow_norm)
        self.assert_finite_tf(spec_J)
        self.assert_finite_tf(cond_J)

    def test_beta_schedule_runs(self):
        beta = tf.linspace(
            tf.constant(0.0, dtype=self.dtype),
            tf.constant(1.0, dtype=self.dtype),
            5
        )

        eta, theta = self._run_update(
            flow_type="EDH",
            beta=beta,
            diagnostics=False
        )

        self.assertEqual(eta.shape, (self.Np, self.d))
        self.assert_finite_tf(eta)
        self.assert_finite_tf(theta)

    def test_beta_equivalent_to_single_uniform_step(self):
        beta = tf.constant([0.0, 1.0], dtype=self.dtype)

        eta_beta, theta_beta = self._run_update(
            flow_type="EDH",
            beta=beta,
            diagnostics=False
        )

        eta_uniform, theta_uniform = self._run_update(
            flow_type="EDH",
            N_lambda=1,
            dl=1.0,
            diagnostics=False
        )

        self.assert_allclose_tf(eta_beta, eta_uniform)
        self.assert_allclose_tf(theta_beta, theta_uniform)

    def test_update_changes_particles_when_measurement_differs(self):
        eta, _ = self._run_update(flow_type="EDH", diagnostics=False)

        moved = tf.reduce_any(tf.abs(eta - self.particles) > 1e-10)
        self.assertTrue(bool(moved))

    def test_invalid_flow_type_raises(self):
        with self.assertRaises(ValueError):
            self._run_update(flow_type="BAD", diagnostics=False)

    def test_invalid_beta_shape_raises(self):
        beta_bad = tf.ones((2, 2), dtype=self.dtype)

        with self.assertRaises(ValueError):
            self._run_update(
                flow_type="EDH",
                beta=beta_bad,
                diagnostics=False
            )

    def test_h_and_jacobian_shapes(self):
        x_batch = tf.ones((self.Np, self.d), dtype=self.dtype)
        x_single = tf.ones((self.d,), dtype=self.dtype)

        h_out = self.h_func(x_batch)
        J_out = self.jacobian_func(x_single)

        self.assertEqual(h_out.shape, (self.Np, self.d))
        self.assertEqual(J_out.shape, (self.d, self.d))

        self.assert_finite_tf(h_out)
        self.assert_finite_tf(J_out)

class TestComputeSpectralNorm(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float32

    def test_runs(self):
        A = tf.random.normal((4, 4), dtype=self.dtype)
        val = compute_spectral_norm(A)

        self.assertIsNotNone(val)

    def test_output_finite(self):
        A = tf.random.normal((5, 5), dtype=self.dtype)
        val = compute_spectral_norm(A)

        self.assertTrue(tf.reduce_all(tf.math.is_finite(val)))

    def test_identity_matrix(self):
        d = 4
        A = tf.eye(d, dtype=self.dtype)

        val = compute_spectral_norm(A)

        # ||I||_2 = 1
        self.assertAlmostEqual(val.numpy(), 1.0, places=3)

    def test_wrong_shape(self):
        A = tf.random.normal((3, 4), dtype=self.dtype)

        with self.assertRaises(tf.errors.InvalidArgumentError):
            compute_spectral_norm(A)



class TestMakeBetaSchedule(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64

    # --------------------------------------------------------
    # Basic correctness
    # --------------------------------------------------------
    def test_basic_properties(self):

        N_lambda = 29
        q = 1.2

        beta, steps = make_beta_schedule(N_lambda=N_lambda, q=q, dtype=self.dtype)

        # shapes
        assert_shape(self, beta, (N_lambda + 1,))
        assert_shape(self, steps, (N_lambda,))

        # finite
        assert_finite(self, beta)
        assert_finite(self, steps)

        # endpoints
        self.assertAlmostEqual(beta[0].numpy(), 0.0, places=10)
        self.assertAlmostEqual(beta[-1].numpy(), 1.0, places=10)

        # sum of steps = 1
        self.assertAlmostEqual(tf.reduce_sum(steps).numpy(), 1.0, places=8)

        # increments consistency
        increments = beta[1:] - beta[:-1]
        assert_allclose(self, increments, steps, atol=1e-8)

        # monotonicity
        self.assertTrue(np.all((beta[1:] - beta[:-1]).numpy() > 0))
        self.assertTrue(np.all(steps.numpy() > 0))

    # --------------------------------------------------------
    # Geometric structure
    # --------------------------------------------------------
    def test_geometric_ratio(self):

        N_lambda = 10
        q = 1.5

        _, steps = make_beta_schedule(N_lambda=N_lambda, q=q, dtype=self.dtype)

        ratios = steps[1:] / steps[:-1]
        expected = tf.ones_like(ratios) * q

        assert_allclose(self, ratios, expected, atol=1e-8)

    # --------------------------------------------------------
    # Edge case: single step
    # --------------------------------------------------------
    def test_single_step(self):

        beta, steps = make_beta_schedule(N_lambda=1, q=1.2, dtype=self.dtype)

        assert_shape(self, beta, (2,))
        assert_shape(self, steps, (1,))

        self.assertAlmostEqual(beta[0].numpy(), 0.0, places=10)
        self.assertAlmostEqual(beta[1].numpy(), 1.0, places=10)
        self.assertAlmostEqual(steps[0].numpy(), 1.0, places=10)

    # --------------------------------------------------------
    # Numerical stability (q ≈ 1)
    # --------------------------------------------------------
    def test_q_close_to_one(self):

        beta, steps = make_beta_schedule(N_lambda=20, q=1.0001, dtype=self.dtype)

        assert_finite(self, beta)
        assert_finite(self, steps)

        self.assertAlmostEqual(tf.reduce_sum(steps).numpy(), 1.0, places=8)
        self.assertAlmostEqual(beta[-1].numpy(), 1.0, places=8)

    # --------------------------------------------------------
    # Known singular case (q = 1)
    # --------------------------------------------------------
    def test_q_equal_one_is_singular(self):

        beta, steps = make_beta_schedule(N_lambda=5, q=1.0, dtype=self.dtype)

        # current implementation produces NaNs → document behavior
        self.assertFalse(np.all(np.isfinite(steps.numpy())))

    # --------------------------------------------------------
    # Input handling
    # --------------------------------------------------------
    def test_integer_N_lambda(self):

        beta, steps = make_beta_schedule(N_lambda=7, q=1.2, dtype=self.dtype)

        assert_shape(self, beta, (8,))
        assert_shape(self, steps, (7,))


class TestParticleFlowPFPropagation(unittest.TestCase):
    """Unit tests for particle_flow_pf_vectorized_propagation"""

    def setUp(self):
        tf.random.set_seed(0)

        self.dtype = tf.float32
        self.Np = 15
        self.T = 4
        self.d = 2

        self.measurements = tf.ones((self.T, self.d), dtype=self.dtype)
        self.P_pred = tf.stack([tf.eye(self.d, dtype=self.dtype)] * self.T)
        self.R_mat = tf.eye(self.d, dtype=self.dtype)
        self.prop_noise_bank = tf.zeros((self.T, self.Np, self.d), dtype=self.dtype)

        def h_func(x):
            x = tf.cast(tf.convert_to_tensor(x), self.dtype)
            return tf.ones_like(x) + tf.constant(0.5, dtype=self.dtype)

        def jacobian_func(x, t=None):
            del x, t
            return tf.eye(self.d, dtype=self.dtype)

        def log_likelihood_fn(particles, y):
            particles = tf.cast(particles, self.dtype)
            y = tf.cast(y, self.dtype)
            return -tf.reduce_sum((particles - y) ** 2, axis=1)

        def prop_fn(x):
            return tf.cast(x, self.dtype)

        def prop_fn_stoch_drift(x, t, eps=None):
            del t
            x = tf.cast(x, self.dtype)
            if eps is None:
                return x + tf.constant(0.1, dtype=self.dtype)
            return x + tf.cast(eps, self.dtype)

        def flow_update_fn(
            particles,
            P_pred_t,
            R_mat,
            z,
            flow_type,
            h_func,
            jacobian_func,
            diagnostics
        ):
            del P_pred_t, R_mat, z, flow_type, h_func, jacobian_func

            particles = tf.cast(particles, self.dtype)
            nloc = tf.shape(particles)[0]
            updated_particles = particles + tf.constant(0.1, dtype=self.dtype)
            theta = tf.ones((nloc,), dtype=self.dtype)

            if diagnostics:
                logdet_J = tf.constant(0.0, dtype=self.dtype)
                flow_norm = tf.ones((nloc,), dtype=self.dtype)
                spec_J = tf.constant(1.0, dtype=self.dtype)
                cond_J = tf.constant(2.0, dtype=self.dtype)
                return updated_particles, theta, logdet_J, flow_norm, spec_J, cond_J

            return updated_particles, theta

        def identity_resampling(particles, weights):
            return particles, weights

        def loglik_weight_helper(log_like, theta, Np, dtype):
            del theta
            weights = tf.ones((Np,), dtype=dtype) / tf.cast(Np, dtype)
            loglik_t = tf.reduce_mean(log_like)
            return weights, loglik_t

        self.h_func = h_func
        self.jacobian_func = jacobian_func
        self.log_likelihood_fn = log_likelihood_fn
        self.prop_fn = prop_fn
        self.prop_fn_stoch_drift = prop_fn_stoch_drift
        self.flow_update_fn = flow_update_fn
        self.identity_resampling = identity_resampling
        self.loglik_weight_helper = loglik_weight_helper

    def run_filter(self, **kwargs):
        defaults = dict(
            measurements=self.measurements,
            Np=self.Np,
            P_pred=self.P_pred,
            R_mat=self.R_mat,
            prop_fn=self.prop_fn,
            flow_update_fn=self.flow_update_fn,
            log_likelihood_fn=self.log_likelihood_fn,
            h_func=self.h_func,
            jacobian_func=self.jacobian_func,
            flow_type="EDH",
            use_weights=True,
            measurement_type="gaussian",
            diagnostics=False,
            prop_fn_stoch_drift=None,
            resampling_fn=None,
            flow_constant=True,
            use_fixed_prop_noise=False,
            prop_noise_bank=None,
            loglik_weight_helper=None,
            collect_resampling_examples=None,
            dtype=self.dtype,
        )
        defaults.update(kwargs)
        return particle_flow_pf_vectorized_propagation(**defaults)

    def assert_core_outputs(self, ests, ESSs, particles, diag, use_weights):
        self.assertEqual(tuple(ests.shape), (self.T, self.d))
        self.assertTrue(np.all(np.isfinite(ests.numpy())))

        self.assertEqual(tuple(particles.shape), (self.Np, self.d))
        self.assertTrue(np.all(np.isfinite(particles.numpy())))

        if use_weights:
            self.assertIsNotNone(ESSs)
            self.assertEqual(tuple(ESSs.shape), (self.T,))
            self.assertTrue(np.all(np.isfinite(ESSs.numpy())))
            self.assertTrue(np.all(ESSs.numpy() > 0.0))
        else:
            self.assertIsNone(ESSs)

        self.assertIsInstance(diag, dict)
        self.assertIn("loglik", diag)
        self.assertTrue(np.isfinite(diag["loglik"].numpy()))

    def test_edh_basic_run(self):
        """Checks basic EDH execution returns finite outputs."""
        ests, ESSs, particles, diag = self.run_filter(
            flow_type="EDH",
            use_weights=True,
            diagnostics=False
        )
        self.assert_core_outputs(ests, ESSs, particles, diag, use_weights=True)

    def test_ledh_basic_run(self):
        """Checks basic LEDH execution returns finite outputs."""
        ests, ESSs, particles, diag = self.run_filter(
            flow_type="LEDH",
            use_weights=True,
            diagnostics=False
        )
        self.assert_core_outputs(ests, ESSs, particles, diag, use_weights=True)

    def test_run_without_weights(self):
        """Checks the no-weight branch returns ESS as None."""
        ests, ESSs, particles, diag = self.run_filter(
            use_weights=False,
            diagnostics=False
        )
        self.assert_core_outputs(ests, ESSs, particles, diag, use_weights=False)

    def test_numpy_measurements_input_runs(self):
        """Checks numpy measurements are accepted."""
        ests, ESSs, particles, diag = self.run_filter(
            measurements=np.ones((self.T, self.d), dtype=np.float32)
        )
        self.assert_core_outputs(ests, ESSs, particles, diag, use_weights=True)

    def test_diagnostics_outputs_present(self):
        """Checks diagnostics mode returns flow-related tensors."""
        ests, ESSs, particles, diag = self.run_filter(diagnostics=True)

        self.assert_core_outputs(ests, ESSs, particles, diag, use_weights=True)

        for key in ["flow_norm", "spec_J", "cond_J", "logdet_J"]:
            self.assertIn(key, diag)
            self.assertEqual(diag[key].shape[0], self.T)
            self.assertTrue(np.all(np.isfinite(diag[key].numpy())))

    def test_invalid_measurements_shape_raises(self):
        """Checks measurements must have shape (T, d)."""
        with self.assertRaises(ValueError):
            self.run_filter(measurements=tf.ones((self.T,), dtype=self.dtype))

    def test_empty_measurements_raises(self):
        """Checks empty measurement sequences raise ValueError."""
        with self.assertRaises(ValueError):
            self.run_filter(measurements=tf.ones((0, self.d), dtype=self.dtype))

    def test_invalid_Np_raises(self):
        """Checks Np must be an integer greater than 1."""
        with self.assertRaises(ValueError):
            self.run_filter(Np=1)

    def test_invalid_flow_type_raises(self):
        """Checks invalid flow_type raises ValueError."""
        with self.assertRaises(ValueError):
            self.run_filter(flow_type="BAD")

    def test_invalid_measurement_type_raises(self):
        """Checks invalid measurement_type raises ValueError."""
        with self.assertRaises(ValueError):
            self.run_filter(measurement_type="bad_type")

    def test_non_bool_use_weights_raises(self):
        """Checks use_weights must be boolean."""
        with self.assertRaises(TypeError):
            self.run_filter(use_weights="yes")

    def test_non_bool_diagnostics_raises(self):
        """Checks diagnostics must be boolean."""
        with self.assertRaises(TypeError):
            self.run_filter(diagnostics="yes")

    def test_non_callable_inputs_raise(self):
        """Checks required function arguments must be callable."""
        bad_cases = [
            dict(prop_fn=None),
            dict(flow_update_fn=None),
            dict(log_likelihood_fn=None),
            dict(h_func=None),
            dict(jacobian_func=None),
        ]

        for kwargs in bad_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(TypeError):
                    self.run_filter(**kwargs)

    def test_poisson_measurement_runs(self):
        """Checks Poisson measurement branch executes."""
        ests, ESSs, particles, diag = self.run_filter(
            measurement_type="poisson",
            diagnostics=True
        )
        self.assert_core_outputs(ests, ESSs, particles, diag, use_weights=True)

    def test_stochastic_propagation_runs(self):
        """Checks stochastic propagation branch executes."""
        ests, ESSs, particles, diag = self.run_filter(
            prop_fn_stoch_drift=self.prop_fn_stoch_drift
        )
        self.assert_core_outputs(ests, ESSs, particles, diag, use_weights=True)

    def test_fixed_prop_noise_requires_noise_bank(self):
        """Checks fixed propagation noise mode requires a noise bank."""
        with self.assertRaises(ValueError):
            self.run_filter(
                prop_fn_stoch_drift=self.prop_fn_stoch_drift,
                use_fixed_prop_noise=True,
                prop_noise_bank=None
            )

    def test_fixed_prop_noise_requires_stochastic_propagation(self):
        """Checks fixed propagation noise mode requires prop_fn_stoch_drift."""
        with self.assertRaises(ValueError):
            self.run_filter(
                use_fixed_prop_noise=True,
                prop_noise_bank=self.prop_noise_bank
            )

    def test_invalid_prop_noise_shape_raises(self):
        """Checks prop_noise_bank must have shape [T, Np, d]."""
        bad_bank = tf.zeros((self.T, self.Np), dtype=self.dtype)
        with self.assertRaises(ValueError):
            self.run_filter(
                prop_fn_stoch_drift=self.prop_fn_stoch_drift,
                use_fixed_prop_noise=True,
                prop_noise_bank=bad_bank
            )

    def test_fixed_prop_noise_runs(self):
        """Checks fixed propagation noise branch executes."""
        ests, ESSs, particles, diag = self.run_filter(
            prop_fn_stoch_drift=self.prop_fn_stoch_drift,
            use_fixed_prop_noise=True,
            prop_noise_bank=self.prop_noise_bank
        )
        self.assert_core_outputs(ests, ESSs, particles, diag, use_weights=True)

    def test_loglik_weight_helper_runs(self):
        """Checks external weighting helper branch executes."""
        ests, ESSs, particles, diag = self.run_filter(
            loglik_weight_helper=self.loglik_weight_helper
        )
        self.assert_core_outputs(ests, ESSs, particles, diag, use_weights=True)

    def test_resampling_fn_runs(self):
        """Checks optional resampling branch executes."""
        ests, ESSs, particles, diag = self.run_filter(
            resampling_fn=self.identity_resampling
        )
        self.assert_core_outputs(ests, ESSs, particles, diag, use_weights=True)

    def test_collect_resampling_examples_runs(self):
        """Checks resampling examples are collected when requested."""
        collected = []

        ests, ESSs, particles, diag = self.run_filter(
            collect_resampling_examples=collected
        )
        self.assert_core_outputs(ests, ESSs, particles, diag, use_weights=True)

        self.assertEqual(len(collected), self.T)
        for p, w in collected:
            self.assertEqual(tuple(p.shape), (self.Np, self.d))
            self.assertEqual(tuple(w.shape), (self.Np,))
            self.assertTrue(np.all(np.isfinite(p.numpy())))
            self.assertTrue(np.all(np.isfinite(w.numpy())))

    def test_zero_loglik_gives_zero_total_loglik(self):
        """Checks zero log-likelihood leads to zero accumulated log-likelihood."""
        def zero_loglik(particles, y):
            del particles, y
            return tf.zeros((self.Np,), dtype=self.dtype)

        ests, ESSs, particles, diag = self.run_filter(
            log_likelihood_fn=zero_loglik,
            use_weights=True
        )

        self.assert_core_outputs(ests, ESSs, particles, diag, use_weights=True)
        self.assertAlmostEqual(float(diag["loglik"].numpy()), 0.0, places=6)


##########################################################
# UNIT TETS SIMULATIONS AND MODEL COMPS - paper Li based 
##########################################################

@unittest.skip("Abstract base class")
class BaseSimulationTest(unittest.TestCase):

    def setUp(self):
        self.d = 3
        self.T = 5
        self.dtype = tf.float32

        # common parameters
        self.alpha = 0.9
        self.sigma_z = 0.1
        self.nu = 5.0

        self.Sigma = tf.eye(self.d, dtype=self.dtype)
        self.gamma = tf.ones((self.d,), dtype=self.dtype) * 0.5

    def simulation_fn(self):
        """Override in subclasses"""
        raise NotImplementedError

    # --------------------------------------------------------
    # Basic simulation test
    # --------------------------------------------------------
    def test_simulation_valid(self):

        x, z = self.simulation_fn()()

        assert_valid_output(self, x, None, self.T, self.d)
        assert_valid_output(self, z, None, self.T, self.d)

    # --------------------------------------------------------
    # Determinism (optional but useful)
    # --------------------------------------------------------
    def test_reproducibility(self):

        tf.random.set_seed(0)
        x1, z1 = self.simulation_fn()()

        tf.random.set_seed(0)
        x2, z2 = self.simulation_fn()()

        self.assertTrue(np.allclose(x1.numpy(), x2.numpy()))
        self.assertTrue(np.allclose(z1.numpy(), z2.numpy()))


class TestSimHDLGSSM(BaseSimulationTest):

    def simulation_fn(self):
        return lambda: Sim_HD_LGSSM(
            d=self.d,
            T=self.T,
            alpha=self.alpha,
            sigma_z=self.sigma_z,
            Sigma_tf=self.Sigma,
            dtype=self.dtype
        )

    def test_invalid_sigma_raises(self):

        wrong_sigma = tf.zeros((self.d, self.d), dtype=self.dtype)

        def wrong_call():
            Sim_HD_LGSSM(
                d=self.d,
                T=self.T,
                alpha=self.alpha,
                sigma_z=self.sigma_z,
                Sigma_tf=wrong_sigma,
                dtype=self.dtype
            )

        assert_raises(self, wrong_call, ValueError)

class TestSimSkewTPoisson(BaseSimulationTest):


    def simulation_fn(self):
        return lambda: generate_skt_poi_data(
            T=self.T,
            d=self.d,
            alpha=self.alpha,
            Sigma_proc=self.Sigma,
            gamma=0.5,   # scalar: internally expanded
            nu=self.nu,
            seed=42
        )

    def test_vector_gamma_runs(self):
        x, z = generate_skt_poi_data(
            T=self.T,
            d=self.d,
            alpha=self.alpha,
            Sigma_proc=self.Sigma,
            gamma=self.gamma,
            nu=self.nu,
            seed=42
        )

        assert_valid_output(self, x, None, self.T, self.d)
        assert_valid_output(self, z, None, self.T, self.d)
    def test_poisson_properties(self):
        
        _, z = self.simulation_fn()()

        # check measurement values are: integers, finite, and non-negative
        self.assertTrue(np.allclose(z.numpy(), np.round(z.numpy())))
        self.assertTrue(np.all(np.isfinite(z.numpy())))
        self.assertTrue(np.all(z.numpy() >= 0))

    # ----- Wrong Call raising Error -----# 
    def test_invalid_parameters_raises(self):

        def wrong_call():
            generate_skt_poi_data(
                T=self.T,
                d=self.d,
                alpha=self.alpha,
                Sigma_proc=self.Sigma,
                gamma=self.gamma, #-1.0,  # invalid
                nu=-1.0,
                seed=42
            )

        assert_raises(self, wrong_call, ValueError)


class TestComputeSigma(unittest.TestCase):
    """Unit tests for spatial covariance matrix construction."""

    def setUp(self):
        self.dtype = tf.float32

    def test_output_shape_dtype_and_finite(self):
        d = 4
        Sigma = compute_Sigma_tf(d, dtype=self.dtype)

        self.assertEqual(Sigma.shape, (d, d))
        self.assertEqual(Sigma.dtype, self.dtype)
        self.assertTrue(np.all(np.isfinite(Sigma.numpy())))

    def test_matrix_is_symmetric(self):
        Sigma = compute_Sigma_tf(9, dtype=self.dtype)

        np.testing.assert_allclose(
            Sigma.numpy(),
            tf.transpose(Sigma).numpy(),
            atol=1e-6,
        )

    def test_diagonal_is_alpha0_plus_alpha1(self):
        alpha0 = 0.3
        alpha1 = 0.01

        Sigma = compute_Sigma_tf(
            4,
            alpha0=alpha0,
            alpha1=alpha1,
            dtype=self.dtype,
        )

        expected_diag = np.ones((4,), dtype=np.float32) * (alpha0 + alpha1)

        np.testing.assert_allclose(
            tf.linalg.diag_part(Sigma).numpy(),
            expected_diag,
            atol=1e-6,
        )

    def test_positive_diagonal_and_positive_entries(self):
        Sigma = compute_Sigma_tf(16, dtype=self.dtype)

        self.assertTrue(np.all(tf.linalg.diag_part(Sigma).numpy() > 0.0))
        self.assertTrue(np.all(Sigma.numpy() > 0.0))

    def test_invalid_dimension_raises(self):
        for bad_d in [0, -1, "bad"]:
            with self.subTest(d=bad_d):
                with self.assertRaises(ValueError):
                    compute_Sigma_tf(bad_d)

    def test_accepts_integer_like_dimension(self):
        Sigma = compute_Sigma_tf(4.0, dtype=self.dtype)

        self.assertEqual(Sigma.shape, (4, 4))
        self.assertEqual(Sigma.dtype, self.dtype)


class TestJacobiansEKF(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.d = 3
        self.x = tf.constant([0.1, -0.2, 0.3], dtype=self.dtype)
        self.alpha = 0.9
        self.m1 = 1.0
        self.m2 = 1.0 / 3.0

    def test_F_jac(self):
        J = F_jac(self.x, t=0, alpha=self.alpha)

        self.assertEqual(J.shape, (self.d, self.d))
        np.testing.assert_allclose(J.numpy(), self.alpha * np.eye(self.d))

    def test_H_jac(self):
        J = H_jac(self.x, t=0)

        self.assertEqual(J.shape, (self.d, self.d))
        np.testing.assert_allclose(J.numpy(), np.eye(self.d))
        
    def test_F_jac_t(self):
        J = F_jac_t(self.x, t=0)

        self.assertEqual(J.shape, (self.d, self.d))
        np.testing.assert_allclose(J.numpy(), self.alpha * np.eye(self.d))

    def test_H_jac_t(self):
        J = H_jac_t(self.x, t=0, m1=self.m1, m2=self.m2)

        self.assertEqual(J.shape, (self.d, self.d))

        expected = np.diag(self.m1 * self.m2 * np.exp(self.m2 * self.x.numpy()))
        np.testing.assert_allclose(J.numpy(), expected)



class TestLogLikelihood(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64

        self.Np = 5
        self.d = 3

        self.particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)

        self.y = tf.constant([0.1, -0.2, 0.3], dtype=self.dtype)

        self.sigma_z = tf.constant(0.5, dtype=self.dtype)

    # --------------------------------------------------------
    # Gaussian likelihood
    # --------------------------------------------------------
    def test_loglik_gaussian(self):

        llk = loglik_gaussian(self.particles, self.y, self.sigma_z)

        self.assertEqual(llk.shape, (self.Np,))
        self.assertTrue(np.all(np.isfinite(llk.numpy())))

        # Gaussian likelihood should be <= 0
        self.assertTrue(np.all(llk.numpy() <= 0))

    # --------------------------------------------------------
    # Poisson likelihood
    # --------------------------------------------------------
    def test_loglik_poisson(self):

        llk = log_likelihood_poisson(self.particles, self.y)

        self.assertEqual(llk.shape, (self.Np,))
        self.assertTrue(np.all(np.isfinite(llk.numpy())))

    # --------------------------------------------------------
    # Poisson stability (important!)
    # --------------------------------------------------------
    def test_loglik_poisson_stability(self):

        extreme_particles = tf.constant(
            [[10.0, -10.0, 5.0]] * self.Np,
            dtype=self.dtype
        )

        llk = log_likelihood_poisson(extreme_particles, self.y)

        self.assertTrue(np.all(np.isfinite(llk.numpy())))


class TestComputeSigma(unittest.TestCase):
    """Unit tests for spatial covariance matrix construction."""

    def setUp(self):
        self.dtype = tf.float32

    def test_output_shape_dtype_and_finite(self):
        d = 4
        Sigma = compute_Sigma_tf(d, dtype=self.dtype)

        self.assertEqual(Sigma.shape, (d, d))
        self.assertEqual(Sigma.dtype, self.dtype)
        self.assertTrue(np.all(np.isfinite(Sigma.numpy())))

    def test_matrix_is_symmetric(self):
        Sigma = compute_Sigma_tf(9, dtype=self.dtype)

        np.testing.assert_allclose(
            Sigma.numpy(),
            tf.transpose(Sigma).numpy(),
            atol=1e-6,
        )

    def test_diagonal_is_alpha0_plus_alpha1(self):
        alpha0 = 0.3
        alpha1 = 0.01

        Sigma = compute_Sigma_tf(
            4,
            alpha0=alpha0,
            alpha1=alpha1,
            dtype=self.dtype,
        )

        expected_diag = np.ones((4,), dtype=np.float32) * (alpha0 + alpha1)

        np.testing.assert_allclose(
            tf.linalg.diag_part(Sigma).numpy(),
            expected_diag,
            atol=1e-6,
        )

    def test_positive_diagonal_and_positive_entries(self):
        Sigma = compute_Sigma_tf(16, dtype=self.dtype)

        self.assertTrue(np.all(tf.linalg.diag_part(Sigma).numpy() > 0.0))
        self.assertTrue(np.all(Sigma.numpy() > 0.0))

    def test_invalid_dimension_raises(self):
        for bad_d in [0, -1, "bad"]:
            with self.subTest(d=bad_d):
                with self.assertRaises(ValueError):
                    compute_Sigma_tf(bad_d)

    def test_accepts_integer_like_dimension(self):
        Sigma = compute_Sigma_tf(4.0, dtype=self.dtype)

        self.assertEqual(Sigma.shape, (4, 4))
        self.assertEqual(Sigma.dtype, self.dtype)


class TestJacobiansEKF(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.d = 3
        self.x = tf.constant([0.1, -0.2, 0.3], dtype=self.dtype)
        self.alpha = 0.9
        self.m1 = 1.0
        self.m2 = 1.0 / 3.0

    def test_F_jac(self):
        J = F_jac(self.x, t=0, alpha=self.alpha)

        self.assertEqual(J.shape, (self.d, self.d))
        np.testing.assert_allclose(J.numpy(), self.alpha * np.eye(self.d))

    def test_H_jac(self):
        J = H_jac(self.x, t=0)

        self.assertEqual(J.shape, (self.d, self.d))
        np.testing.assert_allclose(J.numpy(), np.eye(self.d))
    @unittest.skip("Outdated test: F_jac_t uses alpha from current model setup")
    def test_F_jac_t(self):
        J = F_jac_t(self.x, t=0)

        self.assertEqual(J.shape, (self.d, self.d))
        np.testing.assert_allclose(J.numpy(), self.alpha * np.eye(self.d))

    def test_H_jac_t(self):
        J = H_jac_t(self.x, t=0, m1=self.m1, m2=self.m2)

        self.assertEqual(J.shape, (self.d, self.d))

        expected = np.diag(self.m1 * self.m2 * np.exp(self.m2 * self.x.numpy()))
        np.testing.assert_allclose(J.numpy(), expected)



class TestLogLikelihood(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64

        self.Np = 5
        self.d = 3

        self.particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)

        self.y = tf.constant([0.1, -0.2, 0.3], dtype=self.dtype)

        self.sigma_z = tf.constant(0.5, dtype=self.dtype)

    # --------------------------------------------------------
    # Gaussian likelihood
    # --------------------------------------------------------
    def test_loglik_gaussian(self):

        llk = loglik_gaussian(self.particles, self.y, self.sigma_z)

        self.assertEqual(llk.shape, (self.Np,))
        self.assertTrue(np.all(np.isfinite(llk.numpy())))

        # Gaussian likelihood should be <= 0
        self.assertTrue(np.all(llk.numpy() <= 0))

    # --------------------------------------------------------
    # Poisson likelihood
    # --------------------------------------------------------
    def test_loglik_poisson(self):

        llk = log_likelihood_poisson(self.particles, self.y)

        self.assertEqual(llk.shape, (self.Np,))
        self.assertTrue(np.all(np.isfinite(llk.numpy())))

    # --------------------------------------------------------
    # Poisson stability 
    # --------------------------------------------------------
    def test_loglik_poisson_stability(self):

        extreme_particles = tf.constant(
            [[10.0, -10.0, 5.0]] * self.Np,
            dtype=self.dtype
        )

        llk = log_likelihood_poisson(extreme_particles, self.y)

        self.assertTrue(np.all(np.isfinite(llk.numpy())))



class TestPropagationFcts(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)

        self.dtype = tf.float64
        self.x = tf.constant([1.0, 2.0], dtype=self.dtype)

        self.alpha = tf.constant(0.9, dtype=self.dtype)
        self.Sigma = tf.eye(2, dtype=self.dtype)
        self.gamma_vec = tf.constant([0.1, -0.1], dtype=self.dtype)
        self.gamma_scalar = tf.constant(0.2, dtype=self.dtype)
        self.nu = tf.constant(5.0, dtype=self.dtype)

        self.F = tf.eye(2, dtype=self.dtype)
        self.L = tf.eye(2, dtype=self.dtype)

    # --------------------------------------------------------
    # Shared validator
    # --------------------------------------------------------
    def assert_valid_state_output(self, x_next):
        assert_shape(self, x_next, tuple(self.x.shape))
        assert_finite(self, x_next)

    # --------------------------------------------------------
    # prop_linear_gaussian
    # --------------------------------------------------------
    def test_prop_linear_gaussian_output_validity(self):
        x_next = prop_linear_gaussian(self.x, self.F, self.L)
        self.assert_valid_state_output(x_next)
        self.assertEqual(x_next.dtype, self.dtype)

    def test_prop_linear_gaussian_zero_dynamics_zero_noise(self):
        F = tf.zeros((2, 2), dtype=self.dtype)
        L = tf.zeros((2, 2), dtype=self.dtype)

        x_next = prop_linear_gaussian(self.x, F, L)

        assert_shape(self, x_next, (2,))
        assert_allclose(self, x_next, tf.zeros_like(self.x), atol=1e-12)

    def test_prop_linear_gaussian_reproducible_with_seed_reset(self):
        tf.random.set_seed(123)
        x1 = prop_linear_gaussian(self.x, self.F, self.L)

        tf.random.set_seed(123)
        x2 = prop_linear_gaussian(self.x, self.F, self.L)

        assert_allclose(self, x1, x2, atol=1e-12)

    def test_prop_linear_gaussian_respects_dtype(self):
        x32 = tf.constant([1.0, 2.0], dtype=tf.float32)
        F32 = tf.eye(2, dtype=tf.float32)
        L32 = tf.eye(2, dtype=tf.float32)

        x_next = prop_linear_gaussian(x32, F32, L32)

        assert_shape(self, x_next, (2,))
        assert_finite(self, x_next)
        self.assertEqual(x_next.dtype, tf.float32)

    # --------------------------------------------------------
    # prop_gh_skewed_t
    # --------------------------------------------------------
    def test_prop_gh_skewed_t_vector_gamma_output_validity(self):
        x_next = prop_gh_skewed_t(
            self.x, self.alpha, self.Sigma, self.gamma_vec, self.nu
        )
        self.assert_valid_state_output(x_next)

    def test_prop_gh_skewed_t_scalar_gamma_output_validity(self):
        x_next = prop_gh_skewed_t(
            self.x, self.alpha, self.Sigma, self.gamma_scalar, self.nu
        )
        self.assert_valid_state_output(x_next)

    def test_prop_gh_skewed_t_accepts_scalar_and_vector_gamma(self):
        x_vec = prop_gh_skewed_t(
            self.x, self.alpha, self.Sigma, self.gamma_vec, self.nu
        )
        x_sca = prop_gh_skewed_t(
            self.x, self.alpha, self.Sigma, self.gamma_scalar, self.nu
        )

        assert_shape(self, x_vec, (2,))
        assert_shape(self, x_sca, (2,))
        assert_finite(self, x_vec)
        assert_finite(self, x_sca)

    def test_prop_gh_skewed_t_reproducible_with_seed_reset(self):
        tf.random.set_seed(123)
        x1 = prop_gh_skewed_t(
            self.x, self.alpha, self.Sigma, self.gamma_vec, self.nu
        )

        tf.random.set_seed(123)
        x2 = prop_gh_skewed_t(
            self.x, self.alpha, self.Sigma, self.gamma_vec, self.nu
        )

        assert_allclose(self, x1, x2, atol=1e-12)

    def test_prop_gh_skewed_t_scalar_and_vector_gamma_same_shape(self):
        x_vec = prop_gh_skewed_t(
            self.x, self.alpha, self.Sigma, self.gamma_vec, self.nu
        )
        x_sca = prop_gh_skewed_t(
            self.x, self.alpha, self.Sigma, self.gamma_scalar, self.nu
        )

        assert_shape(self, x_vec, (2,))
        assert_shape(self, x_sca, (2,))

    # --------------------------------------------------------
    # Sigma sanity
    # --------------------------------------------------------
    def test_sigma_is_symmetric_and_psd(self):
        assert_symmetric(self, self.Sigma)
        assert_positive_definite(self, self.Sigma)
        validate_covariance_matrix(self.Sigma)


class TestTransitionLogPDFupf(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.Np = 5
        self.d = 3

        self.x = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        self.x_prev = tf.random.normal((self.Np, self.d), dtype=self.dtype)

        # valid covariance
        A = tf.random.normal((self.d, self.d), dtype=self.dtype)
        Sigma = tf.matmul(A, A, transpose_b=True) + tf.eye(self.d, dtype=self.dtype)
        self.Sigma = Sigma
        self.L = tf.linalg.cholesky(Sigma)

        self.gamma = tf.constant([0.1, -0.1, 0.05], dtype=self.dtype)
        self.alpha = tf.constant(0.9, dtype=self.dtype)
        self.nu = tf.constant(5.0, dtype=self.dtype)

    # ==========================================================
    # Gaussian tests
    # ==========================================================
    def test_gaussian_logpdf(self):
        mean = tf.zeros_like(self.x)

        logpdf = gaussian_logpdf_batch_from_chol(self.x, mean, self.L)

        self.assertEqual(logpdf.shape, (self.Np,))
        self.assertTrue(np.all(np.isfinite(logpdf.numpy())))

    def test_gaussian_symmetry(self):
        mean = tf.zeros_like(self.x)

        logpdf1 = gaussian_logpdf_batch_from_chol(self.x, mean, self.L)
        logpdf2 = gaussian_logpdf_batch_from_chol(-self.x, mean, self.L)

        np.testing.assert_allclose(
            logpdf1.numpy(),
            logpdf2.numpy(),
            rtol=1e-5,
            atol=1e-5
        )

    def test_gaussian_against_numpy(self):
        from scipy.stats import multivariate_normal

        mean = tf.zeros_like(self.x)

        tf_out = gaussian_logpdf_batch_from_chol(self.x, mean, self.L).numpy()

        ref = []
        for i in range(self.Np):
            ref.append(
                multivariate_normal.logpdf(
                    self.x[i].numpy(),
                    mean=np.zeros(self.d),
                    cov=self.Sigma.numpy()
                )
            )
        ref = np.array(ref)

        np.testing.assert_allclose(
            tf_out,
            ref,
            rtol=1e-4,
            atol=1e-4
        )

    # ==========================================================
    # Skew-t tests
    # ==========================================================
    def test_skewt_logpdf(self):
        logpdf = skewt_transition_logpdf_batch_from_chol(
            self.x, self.x_prev, self.L,
            self.alpha, self.gamma, self.nu
        )

        self.assertEqual(logpdf.shape, (self.Np,))
        self.assertTrue(np.all(np.isfinite(logpdf.numpy())))

    def test_skewt_gamma_sensitivity(self):
        logpdf1 = skewt_transition_logpdf_batch_from_chol(
            self.x, self.x_prev, self.L,
            self.alpha, self.gamma, self.nu
        )

        gamma2 = self.gamma + 1.0

        logpdf2 = skewt_transition_logpdf_batch_from_chol(
            self.x, self.x_prev, self.L,
            self.alpha, gamma2, self.nu
        )

        self.assertFalse(np.allclose(logpdf1.numpy(), logpdf2.numpy()))

    def test_invalid_nu(self):
        assert_raises(
            self,
            lambda: skewt_transition_logpdf_batch_from_chol(
                self.x, self.x_prev, self.L,
                self.alpha, self.gamma,
                tf.constant(-1.0)
            ),
            error=tf.errors.InvalidArgumentError,
        )

    # ==========================================================
    # Convergence test
    # ==========================================================
    def test_skewt_converges_to_gaussian(self):
        nu_large = tf.constant(1e3, dtype=self.dtype)
        gamma_zero = tf.zeros_like(self.gamma)

        logpdf_skewt = skewt_transition_logpdf_batch_from_chol(
            self.x, self.x_prev, self.L,
            self.alpha, gamma_zero, nu_large
        )

        mean = self.alpha * self.x_prev

        logpdf_gaussian = gaussian_logpdf_batch_from_chol(
            self.x, mean, self.L
        )

        # compare up to constant offset
        diff = logpdf_skewt - logpdf_gaussian
        diff_np = diff.numpy()

        np.testing.assert_allclose(
            diff_np,
            diff_np[0],
            rtol=1e-3,
            atol=1e-3
        )


class TestGaussianLogPDF(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.Np = 5
        self.d = 3

        self.x = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        self.mean = tf.random.normal((self.Np, self.d), dtype=self.dtype)

        # --- batched covariance ---
        covs = []
        for _ in range(self.Np):
            A = tf.random.normal((self.d, self.d), dtype=self.dtype)
            cov = tf.matmul(A, A, transpose_b=True) + tf.eye(self.d, dtype=self.dtype)
            covs.append(cov)

        self.cov = tf.stack(covs)  # (Np, d, d)

    # ---------------------------
    # Output validity
    # ---------------------------
    def test_output_valid(self):
        logpdf = gaussian_logpdf_batch(self.x, self.mean, self.cov)

        self.assertEqual(logpdf.shape, (self.Np,))
        self.assertTrue(np.all(np.isfinite(logpdf.numpy())))

    # ---------------------------
    # Zero mean sanity
    # ---------------------------
    def test_zero_mean(self):
        mean = tf.zeros_like(self.x)
        logpdf = gaussian_logpdf_batch(self.x, mean, self.cov)

        self.assertEqual(logpdf.shape, (self.Np,))
        self.assertTrue(np.all(np.isfinite(logpdf.numpy())))

    # ---------------------------
    # Symmetry 
    # ---------------------------
    def test_symmetry_property(self):
        mean = tf.zeros_like(self.x)

        logpdf1 = gaussian_logpdf_batch(self.x, mean, self.cov)
        logpdf2 = gaussian_logpdf_batch(-self.x, mean, self.cov)

        np.testing.assert_allclose(logpdf1.numpy(), logpdf2.numpy(), rtol=1e-5, atol=1e-5)

    # ---------------------------
    # Check validity
    # ---------------------------
    def test_vs_true(self):
        from scipy.stats import multivariate_normal

        tf_out = gaussian_logpdf_batch(self.x, self.mean, self.cov).numpy()

        ref = []
        for i in range(self.Np):
            ref.append(multivariate_normal.logpdf(self.x[i].numpy(), mean=self.mean[i].numpy(), cov=self.cov[i].numpy()))
        ref = np.array(ref)

        np.testing.assert_allclose(tf_out, ref, rtol=1e-4, atol=1e-4)


class TestMakeTransitionLogpdf(unittest.TestCase):

    def setUp(self):
        tf.config.run_functions_eagerly(True)

        self.dtype = tf.float64
        self.Np = 5
        self.d = 3

        self.x_new = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        self.x_prev = tf.random.normal((self.Np, self.d), dtype=self.dtype)

        A = tf.random.normal((self.d, self.d), dtype=self.dtype)
        Sigma = tf.matmul(A, A, transpose_b=True) + tf.eye(self.d, dtype=self.dtype)
        self.L = tf.linalg.cholesky(Sigma)

        self.alpha = tf.constant(0.9, dtype=self.dtype)
        self.gamma = tf.constant([0.1, -0.1, 0.05], dtype=self.dtype)
        self.nu = tf.constant(5.0, dtype=self.dtype)

    def tearDown(self):
        tf.config.run_functions_eagerly(False)

    def test_gaussian_factory_matches_direct_call(self):
        fn = make_transition_logpdf_gaussian(self.alpha, self.gamma)

        out_factory = fn(self.x_new, self.x_prev, self.L)

        mean = self.alpha * self.x_prev + self.gamma
        out_direct = gaussian_logpdf_batch_from_chol(
            self.x_new, mean, self.L
        )

        self.assertTrue(callable(fn))
        self.assertEqual(out_factory.shape, (self.Np,))
        self.assertTrue(np.all(np.isfinite(out_factory.numpy())))
        np.testing.assert_allclose(out_factory.numpy(), out_direct.numpy())

    def test_skewt_factory_matches_direct_call(self):
        fn = make_transition_logpdf_skewt(self.alpha, self.gamma, self.nu)

        out_factory = fn(self.x_new, self.x_prev, self.L)
        out_direct = skewt_transition_logpdf_batch_from_chol(
            self.x_new, self.x_prev, self.L,
            self.alpha, self.gamma, self.nu
        )

        self.assertTrue(callable(fn))
        self.assertEqual(out_factory.shape, (self.Np,))
        self.assertTrue(np.all(np.isfinite(out_factory.numpy())))
        np.testing.assert_allclose(out_factory.numpy(), out_direct.numpy())

    def test_factory_parameter_capture(self):
        fn_g1 = make_transition_logpdf_gaussian(self.alpha, self.gamma)
        fn_g2 = make_transition_logpdf_gaussian(self.alpha * 0.5, self.gamma)

        out_g1 = fn_g1(self.x_new, self.x_prev, self.L)
        out_g2 = fn_g2(self.x_new, self.x_prev, self.L)

        self.assertFalse(np.allclose(out_g1.numpy(), out_g2.numpy()))

        fn_t1 = make_transition_logpdf_skewt(self.alpha, self.gamma, self.nu)
        fn_t2 = make_transition_logpdf_skewt(self.alpha * 0.5, self.gamma, self.nu)

        out_t1 = fn_t1(self.x_new, self.x_prev, self.L)
        out_t2 = fn_t2(self.x_new, self.x_prev, self.L)

        self.assertFalse(np.allclose(out_t1.numpy(), out_t2.numpy()))



class TestPropagationGSMC(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float32
        self.Np = 10
        self.d = 3

        self.gamma = tf.constant(0.1, dtype=self.dtype) # scalar gamma
        self.nu = tf.constant(5.0, dtype=self.dtype)
        self.alpha = tf.constant(0.9, dtype=self.dtype)

        self.particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        A = tf.random.normal((self.d, self.d), dtype=self.dtype)
        self.Sigma = tf.matmul(A, A, transpose_b=True) + tf.eye(self.d, dtype=self.dtype)

    def test_vectorized_propagation_runs(self):

        @tf.function(reduce_retracing=True)
        def prop_fn_vec(particles):
            return tf.vectorized_map(
                lambda x: prop_gh_skewed_t(
                    x=x,
                    alpha=self.alpha,
                    Sigma=self.Sigma,
                    gamma=self.gamma,
                    nu=self.nu
                ),
                particles
            )

        out = prop_fn_vec(self.particles)

        # shape check
        self.assertEqual(out.shape, self.particles.shape)

        # finite check
        self.assertTrue(np.all(np.isfinite(out.numpy())))


#######################
#### UNIT  TESTS HU 
#######################

class TestHuKernels(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        np.random.seed(0)

        self.Np = 5
        self.d = 3

        self.X = tf.random.normal((self.Np, self.d), dtype=tf.float32)
        self.alpha = 1.0

        self.Sigma = tf.eye(self.d, dtype=tf.float32)
        self.Qinv = tf.linalg.inv(self.Sigma)

    # ============================
    # SHAPE & FINITE CHECK
    # ============================
    def test_scalar_kernel_shapes(self):
        K, gradK, _ = hu_kernel_scalar(self.X, self.alpha)

        self.assertEqual(K.shape, (self.Np, self.Np))
        self.assertEqual(gradK.shape, (self.Np, self.d))

        self.assertTrue(np.all(np.isfinite(K.numpy())))
        self.assertTrue(np.all(np.isfinite(gradK.numpy())))

    def test_matrix_kernel_shapes(self):
        K, _, gradK = hu_matrix_kernel(self.X, self.alpha, self.Qinv)

        self.assertEqual(K.shape, (self.Np, self.Np))
        self.assertEqual(gradK.shape, (self.Np, self.Np, self.d))

        self.assertTrue(np.all(np.isfinite(K.numpy())))
        self.assertTrue(np.all(np.isfinite(gradK.numpy())))

    # ============================
    # SYMMETRY CHECK
    # ============================
    def test_scalar_kernel_symmetry(self):
        K, _, _ = hu_kernel_scalar(self.X, self.alpha)

        K_np = K.numpy()
        self.assertTrue(np.allclose(K_np, K_np.T, atol=1e-5))

    def test_matrix_kernel_symmetry(self):
        K, _, _ = hu_matrix_kernel(self.X, self.alpha, self.Qinv)

        K_np = K.numpy()
        self.assertTrue(np.allclose(K_np, K_np.T, atol=1e-5))

    # ============================
    # GRADIENT CONSISTENCY 
    # ============================
    def test_scalar_gradient_zero_diagonal(self):
        _, gradK, _ = hu_kernel_scalar(self.X, self.alpha)

        # if X[i] == X[j], repulsion term cancels & gradient small
        grad_norms = tf.norm(gradK, axis=1)

        self.assertTrue(tf.reduce_all(grad_norms >= 0.0))

    def test_matrix_gradient_shape(self):
        _, _, gradK = hu_matrix_kernel(self.X, self.alpha, self.Qinv)

        self.assertEqual(gradK.shape, (self.Np, self.Np, self.d))

    # ============================
    # NUMERICAL STABILITY
    # ============================
    def test_positive_kernel(self):
        K, _, _ = hu_kernel_scalar(self.X, self.alpha)

        self.assertTrue(tf.reduce_all(K >= 0.0))

        K2, _, _ = hu_matrix_kernel(self.X, self.alpha, self.Qinv)
        self.assertTrue(tf.reduce_all(K2 >= 0.0))


class TestLogLikPoissonGrad(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        np.random.seed(0)

        self.Np = 5
        self.d = 3

        self.particles = tf.random.normal((self.Np, self.d), dtype=tf.float32)
        self.y = tf.random.uniform((self.d,), minval=0.5, maxval=2.0, dtype=tf.float32)

        self.m1 = 1.0
        self.m2 = 1.0 / 3.0

    # ============================================================
    # SHAPE and FINITE CHECK
    # ============================================================
    def test_shape_and_finite(self):
        grad = loglik_poisson_grad(self.particles, self.y, self.m1, self.m2)

        self.assertEqual(grad.shape, (self.Np, self.d))
        self.assertTrue(np.all(np.isfinite(grad.numpy())))

    # ============================================================
    # MATHEMATICAL CORRECTNESS
    # ============================================================
    def test_exact_formula(self):
        grad = loglik_poisson_grad(self.particles, self.y, self.m1, self.m2)

        lam = self.m1 * tf.exp(self.m2 * self.particles)
        expected = self.m2 * (self.y[None, :] - lam)

        np.testing.assert_allclose(
            grad.numpy(),
            expected.numpy(),
            atol=1e-6
        )

    # ================================
    # SIGN BEHAVIOR 
    # ================================
    def test_sign_behavior(self):
        particles = tf.zeros((self.Np, self.d), dtype=tf.float32)

        # Case 1: y >> lambda, positive gradient
        y_large = tf.ones((self.d,), dtype=tf.float32) * 10.0
        grad_large = loglik_poisson_grad(particles, y_large, self.m1, self.m2)
        self.assertTrue(tf.reduce_all(grad_large > 0.0))

        # Case 2: y << lambda, negative gradient
        y_small = tf.ones((self.d,), dtype=tf.float32) * 0.001
        grad_small = loglik_poisson_grad(particles, y_small, self.m1, self.m2)
        self.assertTrue(tf.reduce_all(grad_small < 0.0))

    # ===============================
    # BROKEN INPUT TEST 
    # ===============================
    def test_invalid_input_shape(self):
        bad_particles = tf.random.normal((self.d,), dtype=tf.float32)  # wrong shape

        def call():
            loglik_poisson_grad(bad_particles, self.y, self.m1, self.m2)



class TestMakeFlowFunctions(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64
        self.Np = 5
        self.d = 2

        self.particles = tf.constant(
            [[0.0, 1.0],
             [1.0, 0.0],
             [1.0, 1.0],
             [2.0, 1.0],
             [0.5, 0.5]],
            dtype=self.dtype
        )

        self.grad_log_post = tf.constant(
            [[1.0, 0.0],
             [0.0, 1.0],
             [1.0, 1.0],
             [0.5, 0.5],
             [2.0, 1.0]],
            dtype=self.dtype
        )

        self.alpha = tf.constant(2.0, dtype=self.dtype)
        self.M = tf.constant([[2.0, 0.0], [0.0, 3.0]], dtype=self.dtype)

    # --------------------------------------------------------
    # scalar factory
    # --------------------------------------------------------
    def test_make_flow_scalar_output_validity(self):
        def mock_scalar_kernel(X, alpha):
            K = tf.eye(tf.shape(X)[0], dtype=X.dtype)
            gradK = tf.ones_like(X)
            return K, gradK, None

        flow_fn = make_flow_scalar(mock_scalar_kernel)
        flow = flow_fn(self.particles, self.grad_log_post, self.alpha)

        assert_shape(self, flow, (self.Np, self.d))
        assert_finite(self, flow)

    def test_make_flow_scalar_matches_manual_formula(self):
        def mock_scalar_kernel(X, alpha):
            K = tf.eye(tf.shape(X)[0], dtype=X.dtype)
            gradK = tf.ones_like(X)
            return K, gradK, None

        flow_fn = make_flow_scalar(mock_scalar_kernel)
        flow = flow_fn(self.particles, self.grad_log_post, self.alpha)

        expected = (self.grad_log_post + tf.ones_like(self.grad_log_post)) / tf.cast(self.Np, self.dtype)
        assert_allclose(self, flow, expected, atol=1e-10)

    # --------------------------------------------------------
    # matrix factory
    # --------------------------------------------------------
    def test_make_flow_matrix_output_validity(self):
        def mock_matrix_kernel(X, alpha):
            Np = tf.shape(X)[0]
            K = tf.eye(Np, dtype=X.dtype)
            gradK_full = tf.ones((Np, Np, tf.shape(X)[1]), dtype=X.dtype)
            return K, None, gradK_full

        flow_fn = make_flow_matrix(mock_matrix_kernel, self.M)
        flow = flow_fn(self.particles, self.grad_log_post, self.alpha)

        assert_shape(self, flow, (self.Np, self.d))
        assert_finite(self, flow)

    def test_make_flow_matrix_matches_manual_formula(self):
        def mock_matrix_kernel(X, alpha):
            Np = tf.shape(X)[0]
            K = tf.eye(Np, dtype=X.dtype)
            gradK_full = tf.ones((Np, Np, tf.shape(X)[1]), dtype=X.dtype)
            return K, None, gradK_full

        flow_fn = make_flow_matrix(mock_matrix_kernel, self.M)
        flow = flow_fn(self.particles, self.grad_log_post, self.alpha)

        repulsion = tf.reduce_sum(tf.ones((self.Np, self.Np, self.d), dtype=self.dtype), axis=1)

        expected_preM = (self.grad_log_post + repulsion) / tf.cast(self.Np, self.dtype)
        expected = tf.matmul(expected_preM, tf.transpose(self.M))

        assert_allclose(self, flow, expected, atol=1e-10)


class TestHuFactories(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(123)

        self.Np = 6
        self.d = 3
        self.alpha = tf.constant(1.0, dtype=tf.float32)

        self.X = tf.random.normal((self.Np, self.d), dtype=tf.float32)
        self.grad_log_post = tf.random.normal((self.Np, self.d), dtype=tf.float32)

        self.Q = tf.eye(self.d, dtype=tf.float32)
        self.Qinv = tf.linalg.inv(self.Q)

    def test_scalar_kernel_factory_output(self):
        K, gradK, extra = hu_kernel_scalar(self.X, self.alpha)

        self.assertEqual(K.shape, (self.Np, self.Np))
        self.assertEqual(gradK.shape, (self.Np, self.d))
        self.assertIsNone(extra)

        self.assertTrue(tf.reduce_all(tf.math.is_finite(K)).numpy())
        self.assertTrue(tf.reduce_all(tf.math.is_finite(gradK)).numpy())
        self.assertTrue(tf.reduce_all(K >= 0.0).numpy())

    def test_matrix_kernel_factory_output(self):
        kernel_fn = make_hu_matrix_kernel(self.Qinv)
        K, extra, gradK_full = kernel_fn(self.X, self.alpha)

        self.assertEqual(K.shape, (self.Np, self.Np))
        self.assertIsNone(extra)
        self.assertEqual(gradK_full.shape, (self.Np, self.Np, self.d))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(K)).numpy())
        self.assertTrue(tf.reduce_all(tf.math.is_finite(gradK_full)).numpy())
        self.assertTrue(tf.reduce_all(K >= 0.0).numpy())

    def test_scalar_flow_factory_output(self):
        scalar_flow_fn = make_flow_scalar(hu_kernel_scalar)

        flow = scalar_flow_fn(
            self.X,
            self.grad_log_post,
            self.alpha
        )

        self.assertEqual(flow.shape, (self.Np, self.d))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(flow)).numpy())

    def test_matrix_flow_factory_output(self):
        matrix_kernel_fn = make_hu_matrix_kernel(self.Qinv)
        matrix_flow_fn = make_flow_matrix(matrix_kernel_fn, self.Q)

        flow = matrix_flow_fn(
            self.X,
            self.grad_log_post,
            self.alpha
        )

        self.assertEqual(flow.shape, (self.Np, self.d))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(flow)).numpy())

    def test_matrix_and_scalar_flows_are_not_identical_when_metric_changes(self):
        Q = tf.constant(
            [[2.0, 0.0, 0.0],
             [0.0, 0.5, 0.0],
             [0.0, 0.0, 1.5]],
            dtype=tf.float32
        )
        Qinv = tf.linalg.inv(Q)

        scalar_flow_fn = make_flow_scalar(hu_kernel_scalar)

        matrix_kernel_fn = make_hu_matrix_kernel(Qinv)
        matrix_flow_fn = make_flow_matrix(matrix_kernel_fn, Q)

        flow_scalar = scalar_flow_fn(
            self.X,
            self.grad_log_post,
            self.alpha
        )

        flow_matrix = matrix_flow_fn(
            self.X,
            self.grad_log_post,
            self.alpha
        )

        self.assertEqual(flow_scalar.shape, flow_matrix.shape)
        self.assertFalse(
            np.allclose(flow_scalar.numpy(), flow_matrix.numpy())
        )


class TestPFFHuFilter(unittest.TestCase):

    def setUp(self):
        # Save current TF mode and switch to eager (safer for this test)
        self._prev_run_eager = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(True)

        tf.random.set_seed(0)

        self.T = 3
        self.d = 2
        self.Np = 5

        self.measurements = tf.random.normal((self.T, self.d), dtype=tf.float32)

    def tearDown(self):
        # Restore TF execution mode
        tf.config.run_functions_eagerly(self._prev_run_eager)

    # ============================================================
    # MAIN TEST
    # ============================================================
    def test_filter_runs(self):

        def prop_fn(x):
            return x

        def loglik_grad_fn(particles, y):
            with tf.GradientTape() as tape:
                tape.watch(particles)
                loss = tf.reduce_sum(particles ** 2)
            return tape.gradient(loss, particles)

        def flow_step_fn(particles, grad_log_post, alpha):
            return tf.zeros_like(particles)

        def alpha_bandwidth_fn(particles):
            return tf.constant(1.0, dtype=particles.dtype)

        def spectral_norm_fn(H):
            return tf.reduce_sum(H)

        ests, particles_all, diagnostics = pff_filter_hu_new(
            self.measurements,
            Np=self.Np,
            prop_fn=prop_fn,
            loglik_grad_fn=loglik_grad_fn,
            flow_step_fn=flow_step_fn,
#            M=None,
            alpha_bandwidth_fn=alpha_bandwidth_fn,
            spectral_norm_fn=spectral_norm_fn,
            n_steps=2,
            eps=0.01
        )

        # ---- checks using helpers ----
        self.assertEqual(ests.shape, (self.T, self.d))
        self.assertTrue(np.all(np.isfinite(ests.numpy())))

        self.assertEqual(particles_all.shape, (self.T, self.Np, self.d))
        self.assertTrue(np.all(np.isfinite(particles_all.numpy())))

        self.assertIn("flow_norm", diagnostics)
        self.assertIn("grad_cond", diagnostics)
        self.assertIn("spec_J", diagnostics)

        self.assertTrue(np.all(np.isfinite(diagnostics["flow_norm"].numpy())))
        self.assertTrue(np.all(np.isfinite(diagnostics["grad_cond"].numpy())))
        self.assertTrue(np.all(np.isfinite(diagnostics["spec_J"].numpy())))



class TestMonteCarloFinal(unittest.TestCase):
    # ----------------------------
    # Internal mock functions
    # ----------------------------
    def mock_filter_fn(self, Y, **kwargs):
        return Y + 1, Y - 1

    def mock_dict_filter_fn(self, Y, **kwargs):
        return {"k1": Y + 1}, {"k2": Y - 1}

    # ----------------------------
    # INPUT VALIDATION TESTS
    # ----------------------------

    def test_invalid_filter_fn(self):
        with self.assertRaises(TypeError):
            monte_carlo_final("not_a_function", tf.random.normal((5, 3)))

    def test_invalid_N_MC(self):
        with self.assertRaises(ValueError):
            monte_carlo_final(self.mock_filter_fn, tf.random.normal((5, 3)), N_MC=0)

    def test_invalid_output_names_type(self):
        with self.assertRaises(TypeError):
            monte_carlo_final(
                self.mock_filter_fn,
                tf.random.normal((5, 3)),
                output_names=123
            )

    def test_invalid_output_names_content(self):
        with self.assertRaises(TypeError):
            monte_carlo_final(
                self.mock_filter_fn,
                tf.random.normal((5, 3)),
                output_names=["ok", 2]
            )

    def test_invalid_Y(self):
        with self.assertRaises(ValueError): 
            monte_carlo_final(self.mock_filter_fn, "bad_input")

    def test_invalid_Y_shape(self):
        with self.assertRaises(ValueError):
            monte_carlo_final(self.mock_filter_fn, tf.constant(5))

    # ----------------------------
    # CORE FUNCTIONAL TEST
    # ----------------------------

    def test_valid_run(self):
        Y = tf.random.normal((5, 3))

        result = monte_carlo_final(self.mock_filter_fn, Y, N_MC=4)

        self.assertIn("out_0", result)
        self.assertIn("out_1", result)

        self.assertEqual(result["out_0"].shape, (4, 5, 3))
        self.assertEqual(result["out_1"].shape, (4, 5, 3))

    # ----------------------------
    # OUTPUT NAME TEST
    # ----------------------------

    def test_custom_output_names(self):
        Y = tf.random.normal((5, 3))

        result = monte_carlo_final(
            self.mock_filter_fn,
            Y,
            N_MC=3,
            output_names=["a", "b"]
        )

        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_output_name_length_mismatch(self):
        Y = tf.random.normal((5, 3))

        with self.assertRaises(ValueError):
            monte_carlo_final(
                self.mock_filter_fn,
                Y,
                N_MC=3,
                output_names=["only_one"]
            )

    # ----------------------------
    # DICT OUTPUT TEST
    # ----------------------------

    def test_dict_output(self):
        Y = tf.random.normal((5, 3))

        result = monte_carlo_final(self.mock_dict_filter_fn, Y, N_MC=3)

        self.assertIsInstance(result["out_0"], dict)
        self.assertIn("k1", result["out_0"])

        self.assertEqual(result["out_0"]["k1"].shape, (3, 5, 3))




class TestPairwiseDistance(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float64

    def compute_np(self, X, Y=None, squared=True):
        X_np = X.numpy()
        Y_np = X_np if Y is None else Y.numpy()

        XX = np.sum(X_np * X_np, axis=1, keepdims=True)
        YY = np.sum(Y_np * Y_np, axis=1, keepdims=True)

        D = XX - 2 * np.dot(X_np, Y_np.T) + YY.T
        D = np.maximum(D, 0.0)

        if not squared:
            D = np.sqrt(D + 1e-12)

        return D

    # --------------------------------------------------------
    # Basic correctness (squared)
    # --------------------------------------------------------
    def test_squared_distance(self):

        X = tf.constant([[1.0], [2.0], [3.0]], dtype=self.dtype)

        D_tf = pairwise_distance(X, squared=True)
        D_np = self.compute_np(X, squared=True)

        np.testing.assert_allclose(D_tf.numpy(), D_np, atol=1e-10)

    # --------------------------------------------------------
    # Basic correctness (non-squared)
    # --------------------------------------------------------
    def test_unsquared_distance(self):

        X = tf.constant([[1.0], [2.0], [3.0]], dtype=self.dtype)

        D_tf = pairwise_distance(X, squared=False)
        D_np = self.compute_np(X, squared=False)

        np.testing.assert_allclose(D_tf.numpy(), D_np, atol=1e-10)

    # --------------------------------------------------------
    # Cross distance (X vs Y)
    # --------------------------------------------------------
    def test_cross_distance(self):

        X = tf.constant([[1.0], [2.0]], dtype=self.dtype)
        Y = tf.constant([[3.0], [4.0]], dtype=self.dtype)

        D_tf = pairwise_distance(X, Y)
        D_np = self.compute_np(X, Y)

        np.testing.assert_allclose(D_tf.numpy(), D_np, atol=1e-10)

    # --------------------------------------------------------
    # Non-negativity 
    # --------------------------------------------------------
    def test_non_negative(self):

        X = tf.random.normal((5, 3), dtype=self.dtype)

        D = pairwise_distance(X)

        self.assertTrue(np.all(D.numpy() >= -1e-10))


class TestMixtureUniformMultinomialResampling(BaseResamplingTest):

    def resampling_fn(self):
        return mixture_unif_multinomial_resampling

    def test_uniform_weights_after_resampling(self):
        particles = tf.constant([[1.0], [2.0], [3.0], [4.0]], dtype=self.dtype)
        weights = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=self.dtype)

        _, new_weights = self.run_resampling(particles, weights, alpha=0.35)
        expected = tf.ones_like(new_weights) / tf.cast(tf.shape(new_weights)[0], self.dtype)

        assert_allclose(self, new_weights, expected, atol=1e-6)

    def test_alpha_extremes_still_return_uniform_weights(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.linspace(
            tf.constant(0.1, dtype=self.dtype),
            tf.constant(1.0, dtype=self.dtype),
            self.Np
        )
        weights = weights / tf.reduce_sum(weights)

        for alpha in [0.0, 1.0]:
            with self.subTest(alpha=alpha):
                _, new_weights = self.run_resampling(particles, weights, alpha=alpha)
                expected = tf.ones_like(new_weights) / tf.cast(tf.shape(new_weights)[0], self.dtype)
                assert_allclose(self, new_weights, expected, atol=1e-6)

TestMixtureUniformMultinomialResampling.__unittest_skip__ = False
TestMixtureUniformMultinomialResampling.__unittest_skip_why__ = ""

# ============================================================
# No resampling
# ============================================================

class TestNoResampling(BaseResamplingTest):

    def resampling_fn(self):
        return no_resampling

    def test_particles_unchanged(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.ones(self.Np, dtype=self.dtype) / self.Np

        new_particles, _ = self.run_resampling(particles, weights)
        assert_allclose(self, new_particles, particles, atol=1e-12)

    def test_weights_unchanged(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.linspace(
            tf.constant(0.1, dtype=self.dtype),
            tf.constant(1.0, dtype=self.dtype),
            self.Np
        )
        weights = weights / tf.reduce_sum(weights)

        _, new_weights = self.run_resampling(particles, weights)
        assert_allclose(self, new_weights, weights, atol=1e-12)


TestNoResampling.__unittest_skip__ = False
TestNoResampling.__unittest_skip_why__ = ""

# ============================================================
# Soft PFNet resampling
# ============================================================

class TestSoftResamplingPFNet(BaseResamplingTest):

    def resampling_fn(self):
        return soft_resampling_pfnet

    def test_weights_are_normalized_but_not_forced_uniform(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.constant(np.arange(1, self.Np + 1), dtype=self.dtype)
        weights = weights / tf.reduce_sum(weights)

        _, new_weights = self.run_resampling(particles, weights, alpha=0.35)

        # normalized
        np.testing.assert_allclose(tf.reduce_sum(new_weights).numpy(), 1.0, atol=1e-6)

        # not necessarily uniform
        uniform = tf.ones_like(new_weights) / tf.cast(self.Np, self.dtype)
        self.assertFalse(np.allclose(new_weights.numpy(), uniform.numpy(), atol=1e-6))

    def test_alpha_one_returns_normalized_nonnegative_weights(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.constant(np.arange(1, self.Np + 1), dtype=self.dtype)
        weights = weights / tf.reduce_sum(weights)

        _, new_weights = self.run_resampling(particles, weights, alpha=1.0)

        assert_finite(self, new_weights)
        assert_shape(self, new_weights, (self.Np,))

        self.assertTrue(np.all(new_weights.numpy() >= -1e-12))
        np.testing.assert_allclose(tf.reduce_sum(new_weights).numpy(), 1.0, atol=1e-6)

    def test_alpha_zero_preserves_importance_structure_after_renormalization(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.constant(np.arange(1, self.Np + 1), dtype=self.dtype)
        weights = weights / tf.reduce_sum(weights)

        _, new_weights = self.run_resampling(particles, weights, alpha=0.0)

        # Should remain normalized and finite
        assert_finite(self, new_weights)
        np.testing.assert_allclose(tf.reduce_sum(new_weights).numpy(), 1.0, atol=1e-6)

TestSoftResamplingPFNet.__unittest_skip__ = False
TestSoftResamplingPFNet.__unittest_skip_why__ = ""

# ============================================================
# OT resampling
# ============================================================

class TestOTResampling(BaseResamplingTest):

    def resampling_fn(self):
        return soft_resample_ot

    def default_kwargs(self):
        return dict(
            epsilon=tf.constant(0.1, dtype=self.dtype),
            sinkhorn_iters=20,
            normalize_cost=False,
            clip_logK=None,
            return_transport=False,
            return_duals=False,
        )

    def test_uniform_weights_after_resampling(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.constant(np.arange(1, self.Np + 1), dtype=self.dtype)
        weights = weights / tf.reduce_sum(weights)

        _, new_weights = self.run_resampling(particles, weights)
        expected = tf.ones_like(new_weights) / tf.cast(self.Np, self.dtype)

        assert_allclose(self, new_weights, expected, atol=1e-6)

    def test_return_transport(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.ones(self.Np, dtype=self.dtype) / self.Np

        new_particles, new_weights, Tmat = self.run_resampling(
            particles, weights, return_transport=True
        )

        assert_shape(self, new_particles, (self.Np, self.d))
        assert_shape(self, new_weights, (self.Np,))
        assert_shape(self, Tmat, (self.Np, self.Np))
        assert_finite(self, Tmat)

    def test_return_duals(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.ones(self.Np, dtype=self.dtype) / self.Np

        new_particles, new_weights, log_u, log_v = self.run_resampling(
            particles, weights, return_duals=True
        )

        assert_shape(self, new_particles, (self.Np, self.d))
        assert_shape(self, new_weights, (self.Np,))
        assert_shape(self, log_u, (self.Np,))
        assert_shape(self, log_v, (self.Np,))
        assert_finite(self, log_u)
        assert_finite(self, log_v)

    def test_return_transport_and_duals(self):
        particles = tf.random.normal((self.Np, self.d), dtype=self.dtype)
        weights = tf.ones(self.Np, dtype=self.dtype) / self.Np

        new_particles, new_weights, Tmat, log_u, log_v = self.run_resampling(
            particles, weights, return_transport=True, return_duals=True
        )

        assert_shape(self, new_particles, (self.Np, self.d))
        assert_shape(self, new_weights, (self.Np,))
        assert_shape(self, Tmat, (self.Np, self.Np))
        assert_shape(self, log_u, (self.Np,))
        assert_shape(self, log_v, (self.Np,))
TestOTResampling.__unittest_skip__ = False
TestOTResampling.__unittest_skip_why__ = ""

class TestSinkhornLog(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.dtype = tf.float64
        self.N = 4

        self.a = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=self.dtype)
        self.b = tf.constant([0.25, 0.25, 0.25, 0.25], dtype=self.dtype)

        x = tf.reshape(tf.range(self.N, dtype=self.dtype), (-1, 1))
        self.C = tf.square(x - tf.transpose(x))  # simple symmetric cost

        self.epsilon = tf.constant(0.5, dtype=self.dtype)
        self.n_iter = 50

    # --------------------------------------------------------
    # shared runner
    # --------------------------------------------------------
    def run_sinkhorn(self, **kwargs):
        defaults = dict(
            a=self.a,
            b=self.b,
            C=self.C,
            epsilon=self.epsilon,
            n_iter=self.n_iter,
            log_u_init=None,
            log_v_init=None,
            normalize_cost=False,
            clip_logK=None,
            return_duals=False,
        )
        defaults.update(kwargs)
        return sinkhorn_log_general(**defaults)

    # --------------------------------------------------------
    # basic output validity
    # --------------------------------------------------------
    def test_output_validity(self):
        T = self.run_sinkhorn()

        assert_shape(self, T, (self.N, self.N))
        assert_finite(self, T)
        self.assertEqual(T.dtype, self.dtype)
        self.assertTrue(np.all(T.numpy() >= 0.0))

    def test_return_duals(self):
        T, log_u, log_v = self.run_sinkhorn(return_duals=True)

        assert_shape(self, T, (self.N, self.N))
        assert_shape(self, log_u, (self.N,))
        assert_shape(self, log_v, (self.N,))

        assert_finite(self, T)
        assert_finite(self, log_u)
        assert_finite(self, log_v)

    # --------------------------------------------------------
    # transport constraints
    # --------------------------------------------------------
    def test_row_and_column_marginals_match(self):
        T = self.run_sinkhorn()

        row_sums = tf.reduce_sum(T, axis=1)
        col_sums = tf.reduce_sum(T, axis=0)

        assert_allclose(self, row_sums, self.a, atol=1e-4)
        assert_allclose(self, col_sums, self.b, atol=1e-4)

    def test_transport_mass_is_one(self):
        T = self.run_sinkhorn()
        total_mass = tf.reduce_sum(T)
        self.assertAlmostEqual(total_mass.numpy(), 1.0, places=5)

    # --------------------------------------------------------
    # warm start / branch coverage
    # --------------------------------------------------------
    def test_warm_start_runs(self):
        log_u0 = tf.zeros((self.N,), dtype=self.dtype)
        log_v0 = tf.zeros((self.N,), dtype=self.dtype)

        T = self.run_sinkhorn(log_u_init=log_u0, log_v_init=log_v0)

        assert_shape(self, T, (self.N, self.N))
        assert_finite(self, T)

    def test_normalize_cost_runs(self):
        T = self.run_sinkhorn(normalize_cost=True)

        assert_shape(self, T, (self.N, self.N))
        assert_finite(self, T)

        row_sums = tf.reduce_sum(T, axis=1)
        col_sums = tf.reduce_sum(T, axis=0)

        assert_allclose(self, row_sums, self.a, atol=1e-4)
        assert_allclose(self, col_sums, self.b, atol=1e-4)

    def test_clip_logK_runs(self):
        T = self.run_sinkhorn(clip_logK=5.0)

        assert_shape(self, T, (self.N, self.N))
        assert_finite(self, T)

        row_sums = tf.reduce_sum(T, axis=1)
        col_sums = tf.reduce_sum(T, axis=0)

        assert_allclose(self, row_sums, self.a, atol=1e-4)
        assert_allclose(self, col_sums, self.b, atol=1e-4)

    # --------------------------------------------------------
    # normalization of a and b inside the function
    # --------------------------------------------------------
    def test_input_weights_are_renormalized(self):
        a = 2.0 * self.a
        b = 3.0 * self.b

        T = self.run_sinkhorn(a=a, b=b)

        row_sums = tf.reduce_sum(T, axis=1)
        col_sums = tf.reduce_sum(T, axis=0)

        a_norm = a / tf.reduce_sum(a)
        b_norm = b / tf.reduce_sum(b)

        assert_allclose(self, row_sums, a_norm, atol=1e-4)
        assert_allclose(self, col_sums, b_norm, atol=1e-4)

    # --------------------------------------------------------
    # simple exact-ish case
    # --------------------------------------------------------
    def test_identity_like_case_with_zero_cost_diagonal_bias(self):
        C = tf.constant(
            [[0.0, 10.0, 10.0, 10.0],
             [10.0, 0.0, 10.0, 10.0],
             [10.0, 10.0, 0.0, 10.0],
             [10.0, 10.0, 10.0, 0.0]],
            dtype=self.dtype
        )
        a = tf.ones((self.N,), dtype=self.dtype) / self.N
        b = tf.ones((self.N,), dtype=self.dtype) / self.N

        T = self.run_sinkhorn(a=a, b=b, C=C, epsilon=tf.constant(0.1, dtype=self.dtype), n_iter=100)

        diag_mass = tf.reduce_sum(tf.linalg.diag_part(T))
        offdiag_mass = tf.reduce_sum(T) - diag_mass

        self.assertGreater(diag_mass.numpy(), offdiag_mass.numpy())

    # --------------------------------------------------------
    # invalid inputs
    # --------------------------------------------------------
    def test_invalid_epsilon_raises(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.run_sinkhorn(epsilon=tf.constant(0.0, dtype=self.dtype))

    def test_non_square_cost_raises(self):
        bad_C = tf.ones((self.N, self.N + 1), dtype=self.dtype)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.run_sinkhorn(C=bad_C)

    def test_mismatched_a_b_lengths_raise(self):
        bad_b = tf.ones((self.N + 1,), dtype=self.dtype)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.run_sinkhorn(b=bad_b)

    def test_mismatched_cost_and_weights_raise(self):
        bad_C = tf.eye(self.N + 1, dtype=self.dtype)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.run_sinkhorn(C=bad_C)

    def test_negative_a_raises(self):
        bad_a = tf.constant([0.1, -0.2, 0.3, 0.8], dtype=self.dtype)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.run_sinkhorn(a=bad_a)

    def test_negative_b_raises(self):
        bad_b = tf.constant([0.25, 0.25, -0.1, 0.6], dtype=self.dtype)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.run_sinkhorn(b=bad_b)

# ============================================================
# Unit tests: robustify_cost
# ============================================================

class TestRobustifyCost(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float32
        self.C = tf.constant(
            [
                [0.0, 1.0, 10.0],
                [2.0, 5.0, 20.0],
                [3.0, 8.0, 50.0],
            ],
            dtype=self.dtype,
        )
        self.lam = tf.constant(5.0, dtype=self.dtype)

    def test_smooth_clip_matches_formula(self):
        C_rob = robustify_cost(self.C, lambda_robust=self.lam, mode="smooth_clip")
        expected = self.lam * (1.0 - tf.exp(-self.C / self.lam))

        self.assertEqual(C_rob.shape, self.C.shape)
        tf.debugging.assert_near(C_rob, expected, atol=1e-6)
        tf.debugging.assert_all_finite(C_rob, "smooth_clip contains NaN/Inf")

    def test_smooth_clip_is_bounded_by_lambda(self):
        C_rob = robustify_cost(self.C, lambda_robust=self.lam, mode="smooth_clip")

        tf.debugging.assert_greater_equal(C_rob, tf.zeros_like(C_rob))
        tf.debugging.assert_less_equal(
            C_rob,
            tf.ones_like(C_rob) * self.lam + tf.constant(1e-6, dtype=self.dtype),
        )

    def test_hard_clip_matches_minimum(self):
        C_rob = robustify_cost(self.C, lambda_robust=self.lam, mode="hard_clip")
        expected = tf.minimum(self.C, self.lam)

        self.assertEqual(C_rob.shape, self.C.shape)
        tf.debugging.assert_near(C_rob, expected, atol=1e-6)
        tf.debugging.assert_all_finite(C_rob, "hard_clip contains NaN/Inf")

    def test_none_returns_original_cost(self):
        C_rob = robustify_cost(self.C, lambda_robust=self.lam, mode="none")

        self.assertEqual(C_rob.shape, self.C.shape)
        tf.debugging.assert_near(C_rob, self.C, atol=1e-7)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            robustify_cost(self.C, lambda_robust=self.lam, mode="invalid_mode")

    def test_nonpositive_lambda_raises(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            robustify_cost(
                self.C,
                lambda_robust=tf.constant(0.0, dtype=self.dtype),
                mode="smooth_clip",
            )


# ============================================================
# Unit tests: robust_cost option in soft_resample_ot
# ============================================================

class TestSoftResampleOTRobustCost(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float32
        self.N = 5
        self.d = 2

        self.particles = tf.constant(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [20.0, 20.0],
            ],
            dtype=self.dtype,
        )

        self.weights = tf.constant(
            [0.10, 0.15, 0.20, 0.25, 0.30],
            dtype=self.dtype,
        )

        self.epsilon = tf.constant(0.1, dtype=self.dtype)

    def _assert_valid_resampling_output(self, particles_new, weights_new):
        self.assertEqual(tuple(particles_new.shape), tuple(self.particles.shape))
        self.assertEqual(tuple(weights_new.shape), (self.N,))

        tf.debugging.assert_all_finite(
            particles_new,
            "particles_new contains NaN/Inf",
        )
        tf.debugging.assert_all_finite(
            weights_new,
            "weights_new contains NaN/Inf",
        )

        tf.debugging.assert_near(
            tf.reduce_sum(weights_new),
            tf.constant(1.0, dtype=self.dtype),
            atol=1e-6,
        )

        tf.debugging.assert_near(
            weights_new,
            tf.ones([self.N], dtype=self.dtype) / tf.cast(self.N, self.dtype),
            atol=1e-6,
        )

    def test_robust_smooth_clip_runs(self):
        particles_new, weights_new = soft_resample_ot(
            particles=self.particles,
            weights=self.weights,
            epsilon=self.epsilon,
            sinkhorn_iters=10,
            robust_cost=True,
            lambda_robust=5.0,
            robust_mode="smooth_clip",
        )

        self._assert_valid_resampling_output(particles_new, weights_new)

    def test_robust_hard_clip_runs(self):
        particles_new, weights_new = soft_resample_ot(
            particles=self.particles,
            weights=self.weights,
            epsilon=self.epsilon,
            sinkhorn_iters=10,
            robust_cost=True,
            lambda_robust=5.0,
            robust_mode="hard_clip",
        )

        self._assert_valid_resampling_output(particles_new, weights_new)

    def test_robust_none_matches_standard_ot(self):
        particles_std, weights_std, T_std = soft_resample_ot(
            particles=self.particles,
            weights=self.weights,
            epsilon=self.epsilon,
            sinkhorn_iters=15,
            robust_cost=False,
            return_transport=True,
        )

        particles_none, weights_none, T_none = soft_resample_ot(
            particles=self.particles,
            weights=self.weights,
            epsilon=self.epsilon,
            sinkhorn_iters=15,
            robust_cost=True,
            lambda_robust=5.0,
            robust_mode="none",
            return_transport=True,
        )

        tf.debugging.assert_near(particles_none, particles_std, atol=1e-6)
        tf.debugging.assert_near(weights_none, weights_std, atol=1e-7)
        tf.debugging.assert_near(T_none, T_std, atol=1e-6)

    def test_robust_return_transport_runs(self):
        particles_new, weights_new, T = soft_resample_ot(
            particles=self.particles,
            weights=self.weights,
            epsilon=self.epsilon,
            sinkhorn_iters=10,
            robust_cost=True,
            lambda_robust=5.0,
            robust_mode="smooth_clip",
            return_transport=True,
        )

        self._assert_valid_resampling_output(particles_new, weights_new)
        self.assertEqual(tuple(T.shape), (self.N, self.N))
        tf.debugging.assert_all_finite(T, "T contains NaN/Inf")
        tf.debugging.assert_greater_equal(T, tf.zeros_like(T))

    def test_robust_return_duals_runs(self):
        particles_new, weights_new, log_u, log_v = soft_resample_ot(
            particles=self.particles,
            weights=self.weights,
            epsilon=self.epsilon,
            sinkhorn_iters=10,
            robust_cost=True,
            lambda_robust=5.0,
            robust_mode="smooth_clip",
            return_duals=True,
        )

        self._assert_valid_resampling_output(particles_new, weights_new)
        self.assertEqual(tuple(log_u.shape), (self.N,))
        self.assertEqual(tuple(log_v.shape), (self.N,))
        tf.debugging.assert_all_finite(log_u, "log_u contains NaN/Inf")
        tf.debugging.assert_all_finite(log_v, "log_v contains NaN/Inf")

    def test_invalid_robust_mode_raises(self):
        with self.assertRaises(ValueError):
            soft_resample_ot(
                particles=self.particles,
                weights=self.weights,
                epsilon=self.epsilon,
                sinkhorn_iters=5,
                robust_cost=True,
                lambda_robust=5.0,
                robust_mode="invalid_mode",
            )

    def test_nonpositive_lambda_robust_raises(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            soft_resample_ot(
                particles=self.particles,
                weights=self.weights,
                epsilon=self.epsilon,
                sinkhorn_iters=5,
                robust_cost=True,
                lambda_robust=0.0,
                robust_mode="smooth_clip",
            )

# ============================================================
# UNIT TESTS FOR OT GRID TUNING
# ============================================================
class TestOTEntropyTuningUnit(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)

        self.dtype = tf.float32
        self.T = 4
        self.d = 2
        self.Np = 6

        self.true_state = tf.zeros((self.T, self.d), dtype=self.dtype)

    def make_mock_filter(self):
        dtype = self.dtype
        T = self.T

        def filter_fn(resampling_fn=None):
            particles = tf.constant(
                [[0.0, 0.0],
                 [1.0, 0.0],
                 [0.0, 1.0],
                 [1.0, 1.0],
                 [2.0, 0.0],
                 [0.0, 2.0]],
                dtype=dtype
            )

            weights = tf.constant(
                [0.05, 0.10, 0.15, 0.20, 0.20, 0.30],
                dtype=dtype
            )

            if resampling_fn is not None:
                p_new, w_new = resampling_fn(particles, weights)
                est = tf.reduce_sum(p_new * w_new[:, None], axis=0)
                ess = 1.0 / tf.reduce_sum(tf.square(w_new))

                ests = tf.repeat(est[None, :], repeats=T, axis=0)
                ESSs = tf.repeat(tf.reshape(ess, [1]), repeats=T, axis=0)

                return ests, ESSs

            ests = tf.ones((T, 2), dtype=dtype) * 0.5
            ESSs = tf.ones((T,), dtype=dtype) * 4.0
            return ests, ESSs

        return filter_fn

    def test_tune_ot_entropy_regularized_basic(self):
        filter_fn = self.make_mock_filter()

        best_params, best_results, results_table = tune_ot_entropy_regularized(
            filter_fn=filter_fn,
            true_state=self.true_state,
            Np=self.Np,
            niter_grid=(2, 3),
            eps_grid=(1e-2, 1e-1),
            lambda_ess=0.96,
            lambda_speed=0.04,
            n_repeats=1,
            dtype=self.dtype,
            seed=123,
        )

        self.assertIsInstance(best_params, dict)

        for key in [
            "epsilon",
            "sinkhorn_iters",
            "lambda_ess",
            "lambda_speed",
            "rmse",
            "mean_ess",
            "score",
        ]:
            self.assertIn(key, best_params)
            self.assertTrue(np.isfinite(best_params[key]))

        self.assertGreater(best_params["epsilon"], 0.0)
        self.assertIn(best_params["sinkhorn_iters"], [2, 3])

        ests, ESSs = best_results
        assert_valid_output(self, ests, ESSs, self.T, self.d)
        assert_valid_ess(self, ESSs, self.Np)

        self.assertIsInstance(results_table, list)
        self.assertEqual(len(results_table), 4)

        for row in results_table:
            self.assertIn("epsilon", row)
            self.assertIn("sinkhorn_iters", row)
            self.assertIn("score_mean", row)
            self.assertTrue(np.isfinite(row["score_mean"]))

    def test_tune_ot_entropy_regularized_invalid_inputs(self):
        filter_fn = self.make_mock_filter()

        bad_calls = [
            lambda: tune_ot_entropy_regularized(
                filter_fn=filter_fn,
                true_state=None,
                Np=self.Np,
            ),
            lambda: tune_ot_entropy_regularized(
                filter_fn=filter_fn,
                true_state=tf.zeros((self.T,), dtype=self.dtype),
                Np=self.Np,
            ),
            lambda: tune_ot_entropy_regularized(
                filter_fn=filter_fn,
                true_state=self.true_state,
                Np=1,
            ),
        ]

        for bad_call in bad_calls:
            with self.subTest():
                with self.assertRaises(ValueError):
                    bad_call()

    def test_tune_ot_entropy_regularized_robust(self):
        filter_fn = self.make_mock_filter()

        best_params, best_results, results_table = tune_ot_entropy_regularized(
            filter_fn=filter_fn,
            true_state=self.true_state,
            Np=self.Np,
            niter_grid=(2, 3),
            eps_grid=(1e-2, 1e-1),
            lambda_ess=0.96,
            lambda_speed=0.04,
            n_repeats=1,
            robust_cost=True,
            lambda_robust=5.0,
            robust_mode="smooth_clip",
            dtype=self.dtype,
            seed=123,
        )

        self.assertIsInstance(best_params, dict)
        self.assertTrue(best_params["robust_cost"])
        self.assertEqual(best_params["robust_mode"], "smooth_clip")
        self.assertAlmostEqual(float(best_params["lambda_robust"]), 5.0)

        ests, ESSs = best_results
        assert_valid_output(self, ests, ESSs, self.T, self.d)
        assert_valid_ess(self, ESSs, self.Np)

        self.assertEqual(len(results_table), 4)

        for row in results_table:
            self.assertTrue(row["robust_cost"])
            self.assertEqual(row["robust_mode"], "smooth_clip")
            self.assertAlmostEqual(float(row["lambda_robust"]), 5.0)
            self.assertTrue(np.isfinite(row["score_mean"]))


# ============================================================
# INTEGRATION TEST: LINEAR GAUSSIAN BPF
# ============================================================

class TestOTEntropyTuningIntegrationLG(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)

        self.dtype = tf.float32
        self.T = 4
        self.d = 2
        self.Np = 12
        self.sigma_z = tf.constant(0.3, dtype=self.dtype)

        self.F_tf = tf.constant(
            [[0.9, 0.0],
             [0.0, 0.9]],
            dtype=self.dtype,
        )

        self.Q_tf = tf.constant(
            [[0.05, 0.0],
             [0.0, 0.05]],
            dtype=self.dtype,
        )

        self.L_tf = tf.linalg.cholesky(self.Q_tf)

        self.true_state = tf.zeros((self.T, self.d), dtype=self.dtype)
        self.Y = tf.ones((self.T, self.d), dtype=self.dtype)

    def prop_fn_b(self, x):
        return prop_linear_gaussian(x, self.F_tf, self.L_tf)

    def llk_fn_b(self, particles, y):
        return loglik_gaussian(particles, y, self.sigma_z)

    def make_bpf_filter(self):
        Y = self.Y
        Np = self.Np
        dtype = self.dtype

        def filter_fn(resampling_fn=None):
            ests, ESSs, total_loglik = bpf_generic_resampling(
                Y=Y,
                Np=Np,
                prop_fn=self.prop_fn_b,
                log_likelihood_fn=self.llk_fn_b,
                resampling_fn=resampling_fn if resampling_fn is not None else multinomial_resampling,
                resample_threshold=False,
                dtype=dtype,
                carry_resampled_weights=False,
            )

            assert_valid_output(self, ests, ESSs, self.T, self.d)
            assert_valid_ess(self, ESSs, self.Np)
            assert_valid_loglik(self, total_loglik)

            return ests, ESSs

        return filter_fn

    def test_tune_ot_entropy_regularized_with_real_bpf(self):
        tf.config.run_functions_eagerly(True)   

        try:
            best_params, best_results, results_table = tune_ot_entropy_regularized(
                filter_fn=self.make_bpf_filter(),
                true_state=self.true_state,
                Np=self.Np,
                niter_grid=(2, 3),
                eps_grid=(1e-1,),
                lambda_ess=0.9,
                lambda_speed=0.1,
                dtype=self.dtype,
                n_repeats=1,
            )

            self.assertIsInstance(best_params, dict)

        finally:
            tf.config.run_functions_eagerly(False)  

    def test_tune_robust_ot_entropy_regularized_with_real_bpf(self):
        tf.config.run_functions_eagerly(True)

        try:
            best_params, best_results, results_table = tune_ot_entropy_regularized(
                filter_fn=self.make_bpf_filter(),
                true_state=self.true_state,
                Np=self.Np,
                niter_grid=(2, 3),
                eps_grid=(1e-1,),
                lambda_ess=0.9,
                lambda_speed=0.1,
                robust_cost=True,
                lambda_robust=5.0,
                robust_mode="smooth_clip",
                dtype=self.dtype,
                n_repeats=1,
            )

            self.assertIsInstance(best_params, dict)
            self.assertTrue(best_params["robust_cost"])
            self.assertEqual(best_params["robust_mode"], "smooth_clip")
            self.assertAlmostEqual(float(best_params["lambda_robust"]), 5.0)

            ests, ESSs = best_results
            assert_valid_output(self, ests, ESSs, self.T, self.d)
            assert_valid_ess(self, ESSs, self.Np)

            self.assertEqual(len(results_table), 2)

        finally:
            tf.config.run_functions_eagerly(False)


# ========================================
# UNIT TESTS:Differentiable particle filters 
# ========================================
class TestKalmanLoglikAlpha(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.dtype = tf.float32
        self.T = 4
        self.d = 2

        self.Y = tf.ones((self.T, self.d), dtype=self.dtype)
        self.Sigma = tf.eye(self.d, dtype=self.dtype) * 0.1
        self.sigma_z = tf.constant(0.3, dtype=self.dtype)
        self.rho = tf.constant(0.0, dtype=self.dtype)

    def test_returns_scalar_finite_tensor(self):
        out = kalman_loglik_alpha(
            self.Y,
            self.rho,
            self.Sigma,
            self.sigma_z,
            dtype=self.dtype,
        )

        self.assertIsInstance(out, tf.Tensor)
        self.assertEqual(out.shape, ())
        self.assertTrue(np.isfinite(out.numpy()))

    def test_has_gradient_wrt_rho(self):
        rho = tf.Variable(0.0, dtype=self.dtype)

        with tf.GradientTape() as tape:
            out = kalman_loglik_alpha(
                self.Y,
                rho,
                self.Sigma,
                self.sigma_z,
                dtype=self.dtype,
            )

        grad = tape.gradient(out, rho)

        self.assertIsNotNone(grad)
        self.assertEqual(grad.shape, ())
        self.assertTrue(np.isfinite(grad.numpy()))

    def test_dtype_matches_requested_dtype(self):
        out = kalman_loglik_alpha(
            self.Y,
            self.rho,
            self.Sigma,
            self.sigma_z,
            dtype=self.dtype,
        )
        self.assertEqual(out.dtype, self.dtype)


class TestPropFnAlpha(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.dtype = tf.float32
        self.d = 3

        self.x_prev = tf.zeros((self.d,), dtype=self.dtype)
        self.Sigma = tf.eye(self.d, dtype=self.dtype) * 0.1
        self.Sigma_chol = tf.linalg.cholesky(self.Sigma)
        self.rho = tf.constant(0.0, dtype=self.dtype)

    def test_returns_correct_shape_and_finite(self):
        x_next = prop_fn_alpha(
            self.x_prev,
            self.Sigma_chol,
            self.rho,
            dtype=self.dtype,
        )

        self.assertIsInstance(x_next, tf.Tensor)
        self.assertEqual(tuple(x_next.shape), (self.d,))
        self.assertTrue(np.all(np.isfinite(x_next.numpy())))

    def test_output_dtype_matches_requested_dtype(self):
        x_next = prop_fn_alpha(
            self.x_prev,
            self.Sigma_chol,
            self.rho,
            dtype=self.dtype,
        )

        self.assertEqual(x_next.dtype, self.dtype)

    def test_nonzero_input_still_returns_finite_output(self):
        x_prev = tf.ones((self.d,), dtype=self.dtype)
        x_next = prop_fn_alpha(
            x_prev,
            self.Sigma_chol,
            self.rho,
            dtype=self.dtype,
        )

        self.assertEqual(tuple(x_next.shape), (self.d,))
        self.assertTrue(np.all(np.isfinite(x_next.numpy())))


class TestLogLikelihoodGaussian(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float32
        self.Np = 8
        self.d = 2
        self.sigma_z = tf.constant(0.3, dtype=self.dtype)

        self.particles = tf.zeros((self.Np, self.d), dtype=self.dtype)
        self.y_t = tf.ones((self.d,), dtype=self.dtype)

    def test_returns_vector_of_correct_shape(self):
        llk = log_likelihood_gaussian(
            self.particles,
            self.y_t,
            self.sigma_z,
            dtype=self.dtype,
        )

        self.assertIsInstance(llk, tf.Tensor)
        self.assertEqual(tuple(llk.shape), (self.Np,))
        self.assertTrue(np.all(np.isfinite(llk.numpy())))

    def test_equal_particles_give_equal_loglikelihoods(self):
        llk = log_likelihood_gaussian(
            self.particles,
            self.y_t,
            self.sigma_z,
            dtype=self.dtype,
        )

        self.assertTrue(np.allclose(llk.numpy(), llk.numpy()[0]))

    def test_dtype_matches_requested_dtype(self):
        llk = log_likelihood_gaussian(
            self.particles,
            self.y_t,
            self.sigma_z,
            dtype=self.dtype,
        )

        self.assertEqual(llk.dtype, self.dtype)

    def test_closer_particle_has_higher_loglikelihood(self):
        particles = tf.constant(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=self.dtype
        )
        y_t = tf.constant([1.0, 1.0], dtype=self.dtype)

        llk = log_likelihood_gaussian(
            particles,
            y_t,
            self.sigma_z,
            dtype=self.dtype,
        ).numpy()

        self.assertGreater(llk[1], llk[0])



class TestMakePropFn(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)
        self.dtype = tf.float32
        self.d = 2
        self.Sigma = tf.eye(self.d, dtype=self.dtype) * 0.1
        self.Sigma_chol = tf.linalg.cholesky(self.Sigma)
        self.rho = tf.constant(0.0, dtype=self.dtype)
        self.x_prev = tf.zeros((self.d,), dtype=self.dtype)

    def test_returns_callable(self):
        fn = make_prop_fn(self.Sigma_chol, self.rho, dtype=self.dtype)
        self.assertTrue(callable(fn))

    def test_callable_returns_valid_tensor(self):
        fn = make_prop_fn(self.Sigma_chol, self.rho, dtype=self.dtype)
        x_next = fn(self.x_prev)

        self.assertIsInstance(x_next, tf.Tensor)
        self.assertEqual(tuple(x_next.shape), (self.d,))
        self.assertEqual(x_next.dtype, self.dtype)
        self.assertTrue(np.all(np.isfinite(x_next.numpy())))



class TestMakeLlkFn(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float32
        self.Np = 5
        self.d = 2
        self.sigma_z = tf.constant(0.3, dtype=self.dtype)

        self.particles = tf.zeros((self.Np, self.d), dtype=self.dtype)
        self.y_t = tf.ones((self.d,), dtype=self.dtype)

    def test_returns_callable(self):
        fn = make_llk_fn(self.sigma_z, dtype=self.dtype)
        self.assertTrue(callable(fn))

    def test_callable_matches_direct_log_likelihood_gaussian(self):
        fn = make_llk_fn(self.sigma_z, dtype=self.dtype)

        out_wrapped = fn(self.particles, self.y_t)
        out_direct = log_likelihood_gaussian(
            self.particles,
            self.y_t,
            self.sigma_z,
            dtype=self.dtype,
        )

        self.assertEqual(tuple(out_wrapped.shape), (self.Np,))
        self.assertTrue(np.allclose(out_wrapped.numpy(), out_direct.numpy()))



# ======================================
# UNIT TESTS: Monte Carlo Functions  
# ======================================
class TestMonteCarloLightLost(unittest.TestCase):

    def setUp(self):
        self.T = 5
        self.d = 2

        self.X_true = tf.ones((self.T, self.d), dtype=tf.float32)

    # ------------------------
    # MOCK FILTERS
    # ------------------------
    def perfect_filter(self):
        ests = tf.ones((self.T, self.d), dtype=tf.float32)
        ESSs = tf.ones((self.T,), dtype=tf.float32)
        return ests, ESSs

    def noisy_filter(self):
        ests = self.X_true + tf.random.normal((self.T, self.d), stddev=0.1)
        return ests, None

    def bad_filter(self):
        return "invalid"

    # ------------------------
    # VALID CASE
    # ------------------------
    def test_perfect_filter(self):
        mse, ess, lost = monte_carlo_light_lost(self.perfect_filter, N_MC=5, X_true=self.X_true)

        self.assertEqual(lost, 0)
        self.assertTrue(np.all(np.isfinite(mse.numpy())))
        self.assertEqual(mse.shape, (self.T,))

    # ------------------------
    # LOST TRACK CASE
    # ------------------------
    def test_lost_tracks(self):
        def always_bad():
            return self.X_true * 100.0, None  # huge error → lost #100.0

        mse, ess, lost = monte_carlo_light_lost(always_bad, N_MC=3, X_true=self.X_true)
        self.assertEqual(lost, 3)

    # ------------------------
    # MULTIPLE MC RUNS
    # ------------------------
    def test_monte_carlo_stability(self):
        mse, ess, lost = monte_carlo_light_lost(self.noisy_filter, N_MC=10, X_true=self.X_true)

        self.assertTrue(np.all(np.isfinite(mse.numpy())))
        self.assertLessEqual(lost, 10)

    # ------------------------
    # INPUT CHECKS
    # ------------------------
    def test_invalid_filter(self):

        with self.assertRaises(TypeError):
            monte_carlo_light_lost("not a function", 5, self.X_true)

    def test_invalid_N_MC(self):

        with self.assertRaises(ValueError):
            monte_carlo_light_lost(self.perfect_filter, 0, self.X_true)

    def test_invalid_X_shape(self):

        with self.assertRaises(ValueError):
            monte_carlo_light_lost(self.perfect_filter, 5, tf.ones((3,), dtype=tf.float32)  # wrong shape
            )

    def test_bad_filter_output(self):
        with self.assertRaises(Exception):  
            monte_carlo_light_lost(self.bad_filter, 3, self.X_true)

    def test_shape_mismatch(self):

        def wrong_shape_filter():
            return tf.ones((self.T + 1, self.d), dtype=tf.float32)

        with self.assertRaises(ValueError):
            monte_carlo_light_lost(wrong_shape_filter, 2, self.X_true)



class TestRunMonteCarloSim(unittest.TestCase):

    def setUp(self):
        self.T = 5
        self.true_data = tf.ones((self.T, 2), dtype=tf.float32)

    # ------------------------
    # MOCK MONTE CARLO FUNCTION
    # ------------------------
    def mock_monte_carlo_fn(self, filter_fn, N_MC, true_data):
        mse = tf.ones((self.T,), dtype=tf.float32)
        ess = tf.ones((self.T,), dtype=tf.float32)
        lost = 0
        return mse, ess, lost

    # ------------------------
    # VALID RUN
    # ------------------------
    def test_valid(self):

        filters = {"f1": lambda: None}

        results = run_monte_carlo_sim(
            filters,
            self.true_data,
            N_MC=2,
            monte_carlo_fn=self.mock_monte_carlo_fn
        )

        self.assertIn("f1", results)
        self.assertEqual(results["f1"]["mse_t"].shape, (self.T,))
        self.assertEqual(results["f1"]["lost_tracks"], 0)

    # ------------------------
    # INPUT CHECKS
    # ------------------------
    def test_invalid_inputs(self):

        with self.assertRaises(TypeError):
            run_monte_carlo_sim("not a dict", self.true_data, monte_carlo_fn=self.mock_monte_carlo_fn)

        with self.assertRaises(ValueError):
            run_monte_carlo_sim({}, self.true_data, monte_carlo_fn=self.mock_monte_carlo_fn)

        with self.assertRaises(TypeError):
            run_monte_carlo_sim({"f1": lambda: None}, self.true_data)

        with self.assertRaises(ValueError):
            run_monte_carlo_sim(
                {"f1": lambda: None},
                self.true_data,
                N_MC=0,
                monte_carlo_fn=self.mock_monte_carlo_fn
            )

    # ------------------------
    # MULTIPLE FILTERS
    # ------------------------
    def test_multiple_filters(self):

        filters = {
            "f1": lambda: None,
            "f2": lambda: None
        }

        results = run_monte_carlo_sim(
            filters,
            self.true_data,
            N_MC=2,
            monte_carlo_fn=self.mock_monte_carlo_fn
        )

        self.assertEqual(len(results), 2)


# =================================
# UNIT TESTS: Dai - Beta Homotopy
# =================================

class TestComputeH(unittest.TestCase):

    def setUp(self):
        self.dtype = tf.float32
        self.x = tf.constant([1.0, 2.0], dtype=tf.float32)

        self.sensors = tf.constant([
            [0.0, 0.0],
            [1.0, 0.5]
        ], dtype=tf.float32)

        self.Ns = 2 

        self._R_backup = globals().get("R", None)
        globals()["R"] = tf.eye(2, dtype=tf.float32)

    def tearDown(self):
        if self._R_backup is not None:
            globals()["R"] = self._R_backup
        else:
            del globals()["R"]

    def test_valid(self):
        H = compute_H(self.x, self.sensors)

        self.assertEqual(H.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(H.numpy())))

    def test_symmetry(self):
        H = compute_H(self.x, self.sensors)
        diff = H - tf.transpose(H)
        self.assertTrue(tf.reduce_max(tf.abs(diff)).numpy() < 1e-5 )

    def test_R_invalid(self):
        bad_R = tf.ones((2, 3), dtype=self.dtype)

        with self.assertRaises(Exception):
            compute_H(self.x, self.sensors, bad_R)
            
    def test_input_shapes(self):
        with self.assertRaises(Exception):
            compute_H(
                tf.constant([1.0, 2.0, 3.0]),  # wrong shape
                self.sensors
            )


class TestComputeJ(unittest.TestCase):

    def setUp(self):
        tf.random.set_seed(0)

        self.dtype = tf.float64  
        self.beta = tf.constant([1.0, 2.0, 3.0], dtype=self.dtype)

        self.H = tf.constant([
            [1.0, 0.2],
            [0.2, 1.0]
        ], dtype=self.dtype)

        self.Q = tf.constant([
            [2.0, 0.0],
            [0.0, 2.0]
        ], dtype=self.dtype)

        self.mu = tf.constant(1.0, dtype=self.dtype)
        self.h = tf.constant(1.0, dtype=self.dtype)

        self._backup = {
            "Q": globals().get("Q", None),
            "mu": globals().get("mu", None),
            "h": globals().get("h", None),
        }

        globals()["Q"] = self.Q
        globals()["mu"] = self.mu
        globals()["h"] = self.h

    def tearDown(self):
        for k, v in self._backup.items():
            if v is not None:
                globals()[k] = v
            else:
                globals().pop(k, None)

    # ------------------------
    # VALID CASE
    # ------------------------
    def test_valid(self):
        J = compute_J(self.beta, self.H)

        self.assertTrue(np.all(np.isfinite(J.numpy())))
        self.assertEqual(J.dtype, self.dtype)

    # ------------------------
    # TYPE CHECKS
    # ------------------------
    def test_beta_not_tensor(self):
        with self.assertRaises(TypeError):
            compute_J([1.0, 2.0, 3.0], self.H)

    def test_beta_wrong_rank(self):
        with self.assertRaises(ValueError):
            compute_J(tf.constant([[1.0, 2.0]], dtype=self.dtype), self.H)

    def test_beta_too_short(self):
        with self.assertRaises(ValueError):
            compute_J(tf.constant([1.0], dtype=self.dtype), self.H)

    def test_H_wrong_shape(self):
        with self.assertRaises(ValueError):
            compute_J(self.beta, tf.eye(3, dtype=self.dtype))

    # ------------------------
    # GLOBAL DEPENDENCIES
    # ------------------------
    def test_Q_wrong_shape(self):
        bad_Q = tf.eye(3, dtype=self.dtype)

        with self.assertRaises(ValueError):
            compute_J(self.beta, self.H, Q=bad_Q)

    def test_singular_matrix(self):
        bad_Q = tf.constant([
            [1.0, 2.0],
            [2.0, 4.0]
        ], dtype=self.dtype)

        with self.assertRaises(ValueError):
            compute_J(self.beta, self.H, Q=bad_Q)

    # ------------------------
    # NUMERICAL
    # ------------------------
    def test_zero_beta_gradient(self):
        beta = tf.constant([1.0, 1.0, 1.0], dtype=self.dtype)

        J = compute_J(beta, self.H)
        self.assertTrue(np.all(np.isfinite(J.numpy())))



class TestOptimBeta(unittest.TestCase):

    def test_optimize_beta_basic(self):
        tf.random.set_seed(0)

        N = 3

        # simple convex loss (minimum at beta = 0.5)
        def loss(beta):
            return tf.reduce_sum((beta - 0.5) ** 2)

        beta_star, J_star = optimize_beta(loss, N, dtype=tf.float64, num_iters=200)

        # --- shape checks ---
        self.assertEqual(beta_star.shape, (N + 1,))

        # --- boundary conditions (enforced by function) ---
        self.assertAlmostEqual(beta_star[0].numpy(), 0.0, places=6)
        self.assertAlmostEqual(beta_star[-1].numpy(), 1.0, places=6)

        # --- monotonic / bounded ---
        self.assertTrue(np.all(beta_star.numpy() >= 0.0))
        self.assertTrue(np.all(beta_star.numpy() <= 1.0))

        # --- loss decreased ---
        self.assertTrue(J_star.numpy() >= 0.0)

    # CHECK OPTIMISATION REDUCES LOSS
    def test_optimization_improves_loss(self):
        tf.random.set_seed(0)

        def loss(beta):
            return tf.reduce_sum((beta - 0.5) ** 2)

        beta_star, J_star = optimize_beta(loss, N=3, num_iters=100)

        # initial loss (approx)
        beta_init = tf.cast(tf.linspace(0., 1., 4), dtype=tf.float64)
        J_init = loss(beta_init)

        self.assertLess(J_star.numpy(), J_init.numpy())
        self.assertTrue(np.isfinite(J_star.numpy()))


######
# INTEGRATION TESTS 
#####
######################
### Integration test Gaussian and Poisson pipelines for reproducing the results of Li.
#####################
# ============================================================
# Shared helpers
# ============================================================

def make_resampling_variants(dtype, include_sinkhorn=True):
    variants = {
        "Multinomial": multinomial_resampling,
        "Mixture_uniform": mixture_unif_multinomial_resampling,
        "Soft_pfnet": soft_resampling_pfnet,
        "No-resampling":no_resampling
    }

    if include_sinkhorn:
        variants["sinkhorn_ot"] = lambda p, w: soft_resample_ot(
            p,
            w,
            epsilon=tf.cast(0.1, dtype),
            sinkhorn_iters=5,
            normalize_cost=True,
        )

    return variants


def build_beta_star(Sigma, dtype):
    sensors = tf.constant([[3.5, 0.0], [-3.5, 0.0]], dtype=dtype)
    x_true = tf.constant([4.0, 4.0], dtype=dtype)

    H = compute_H(x_true, sensors)
    loss = lambda beta: compute_J(beta, H, Q=Sigma)

    beta_star, _ = optimize_beta(
        loss_fct=loss,
        N=20,
        dtype=dtype,
        num_iters=100,
    )

    return beta_star


# ============================================================
# Base integration class
# ============================================================
@unittest.skip("Abstract base class")
class BasePipelineIntegrationTest(unittest.TestCase):
    """
    Shared integration test harness for Gaussian and Poisson pipelines.

    Each subclass must define:
        self.measurements
        self.true_states
        self.Np
        self.T
        self.d
        self.dtype
        self.Sigma
        self.R_mat
        self.P_pred
        self.prop_fn
        self.h_func
        self.llk_fn
        self.jacobian_func
        self.measurement_type
    """

    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------
    def _check_output(self, ests, name):
        self.assertEqual(
            ests.shape,
            (self.T, self.d),
            msg=f"{name}: shape mismatch"
        )
        self.assertTrue(
            np.all(np.isfinite(ests.numpy())),
            msg=f"{name}: contains NaN/Inf"
        )

    def _check_monte_carlo_output(self, result, expected_filters):
        self.assertIsInstance(result, dict)

        for name in expected_filters:
            with self.subTest(mc_filter=name):
                self.assertIn(name, result)

                out = result[name]

                self.assertIn("mse_t", out)
                self.assertIn("ess_mean", out)
                self.assertIn("lost_tracks", out)
                self.assertIn("run_time", out)

                self.assertTrue(np.all(np.isfinite(np.asarray(out["mse_t"]))))
                self.assertGreaterEqual(float(out["lost_tracks"]), 0.0)
                self.assertGreaterEqual(float(out["run_time"]), 0.0)

                if out["ess_mean"] is not None:
                    self.assertTrue(np.all(np.isfinite(np.asarray(out["ess_mean"]))))

    # --------------------------------------------------------
    # Filter registry
    # --------------------------------------------------------
    def get_filter_registry(self):
        raise NotImplementedError

    def get_baseline_filters(self):
        raise NotImplementedError

    def get_extended_filters(self):
        return []

    # --------------------------------------------------------
    # Generic runners
    # --------------------------------------------------------
    def run_filter_group(self, filter_names):
        registry = self.get_filter_registry()
        results = {}

        for name in filter_names:
            with self.subTest(filter=name):
                self.assertIn(name, registry)

                out = registry[name]()
                ests = out[0] if isinstance(out, tuple) else out

                self._check_output(ests, name)
                results[name] = ests

        return results

    def run_monte_carlo_group(self, filter_names, N_MC=2):
        registry = self.get_filter_registry()
        filters_config = {name: registry[name] for name in filter_names}

        result = run_monte_carlo_sim(
            filters_config=filters_config,
            true_data=self.true_states,
            N_MC=N_MC,
            monte_carlo_fn=monte_carlo_light_lost,
        )

        self._check_monte_carlo_output(result, expected_filters=filter_names)

        return result

    # --------------------------------------------------------
    # BPF variants
    # --------------------------------------------------------
    def run_bpf_resampling_variants(self):
        variants = make_resampling_variants(
            dtype=self.dtype,
            include_sinkhorn=True,
        )

        for name, resampling_fn in variants.items():
            with self.subTest(bpf_resampling=name):
                out = run_bpf(
                    Y=self.measurements,
                    Np=self.Np,
                    prop_fn=self.prop_fn,
                    log_likelihood_fn=self.llk_fn,
                    resampling_fn=resampling_fn,
                )

                ests = out[0] if isinstance(out, tuple) else out
                self._check_output(ests, f"BPF-{name}")

    # --------------------------------------------------------
    # PFPF beta / resampling variants
    # --------------------------------------------------------
    def run_pfpf_variants(self):
        beta_star = build_beta_star(
            Sigma=self.Sigma,
            dtype=self.dtype,
        )

        betas = {
            "uniform": None,
            "beta_star": beta_star,
        }

        resampling_variants = make_resampling_variants(
            dtype=self.dtype,
            include_sinkhorn=True,
        )

        for flow_type in ["EDH", "LEDH"]:
            for use_weights in [False, True]:
                for beta_name, beta in betas.items():
                    for resampling_name, resampling_fn in resampling_variants.items():

                        with self.subTest(
                            flow=flow_type,
                            weighted=use_weights,
                            beta=beta_name,
                            resampling=resampling_name,
                        ):
                            out = run_pfpf_fn(
                                flow_type=flow_type,
                                use_weights=use_weights,
                                beta=beta,
                                measurement_type=self.measurement_type,
                            )(
                                Y=self.measurements,
                                Np=self.Np,
                                P_pred=self.P_pred,
                                R_mat=self.R_mat,
                                prop_fn=self.prop_fn,
                                log_likelihood_fn=self.llk_fn,
                                h_func=self.h_func,
                                jacobian_func=self.jacobian_func,
                                resampling_fn=resampling_fn,
                            )[:2]

                            ests = out[0] if isinstance(out, tuple) else out
                            self._check_output(
                                ests,
                                f"{flow_type}-{beta_name}-{resampling_name}"
                            )

    # --------------------------------------------------------
    # Common tests
    # --------------------------------------------------------
    def test_baseline_filters_run(self):
        self.run_filter_group(self.get_baseline_filters())

    def test_extended_filters_run(self):
        names = self.get_extended_filters()
        if names:
            self.run_filter_group(names)

    def test_baseline_monte_carlo_runs(self):
        self.run_monte_carlo_group(self.get_baseline_filters(), N_MC=2)

    def test_bpf_resampling_variants(self):
        self.run_bpf_resampling_variants()

    def test_pfpf_beta_and_resampling_variants(self):
        self.run_pfpf_variants()



class TestGaussianPipelineIntegration(BasePipelineIntegrationTest):

    def setUp(self):
        tf.random.set_seed(0)

        self.d = 2
        self.T = 5
        self.Np = 30
        self.dtype = tf.float64
        self.measurement_type = "gaussian"

        self.alpha = 0.9
        self.sigma_z = 0.1

        self.Sigma = tf.eye(self.d, dtype=self.dtype)
        self.R_mat = tf.eye(self.d, dtype=self.dtype) * self.sigma_z**2
        self.L = tf.linalg.cholesky(self.Sigma)

        self.true_states, self.measurements = Sim_HD_LGSSM(
            d=self.d,
            T=self.T,
            alpha=self.alpha,
            sigma_z=self.sigma_z,
            Sigma_tf=self.Sigma,
            dtype=self.dtype,
        )

        self.F_mat = self.alpha * tf.eye(self.d, dtype=self.dtype)
        self.H_mat = tf.eye(self.d, dtype=self.dtype)

        @tf.function
        def prop_fn(x):
            return prop_linear_gaussian(x, self.F_mat, self.L)

        @tf.function
        def h_func(x):
            return x

        @tf.function
        def llk_fn(x, y):
            return loglik_gaussian(x, y, self.sigma_z)

        self.prop_fn = prop_fn
        self.h_func = h_func
        self.llk_fn = llk_fn
        self.jacobian_func = H_jac_tf

        self.m0 = tf.zeros(self.d, dtype=self.dtype)
        self.P0 = tf.eye(self.d, dtype=self.dtype)

        self.ekf_result = run_ekf_wrap(
            Y=tf.transpose(self.measurements),
            m0=self.m0,
            P0=self.P0,
            Q=self.Sigma,
            R=self.R_mat,
            F=lambda x, t: self.alpha * x,
            H=lambda x, t: x,
            F_jac=F_jac_tf,
            H_jac=H_jac_tf,
            measurement_type="gaussian",
        )

        self.P_pred = tf.cast(self.ekf_result["P_pred"], self.dtype)

        self.transition_fn_gauss = make_transition_logpdf_gaussian(
            alpha=self.alpha,
            gamma=0.0,
        )

    def get_filter_registry(self):
        return {
            "KF": lambda: tf.cast(
                run_kf_wrap(
                    Y=tf.transpose(self.measurements),
                    m0=self.m0,
                    P0=self.P0,
                    Q=self.Sigma,
                    R=self.R_mat,
                    F_mat=self.F_mat,
                    H_mat=self.H_mat,
                    measurement_type="gaussian",
                ),
                self.dtype,
            ),

            "UKF": lambda: tf.cast(
                run_ukf_wrap(
                    Y=tf.transpose(self.measurements),
                    m0=self.m0,
                    P0=self.P0,
                    Q=self.Sigma,
                    R=self.R_mat,
                    F=lambda x, t: self.alpha * x,
                    H=lambda x, t: x,
                    measurement_type="gaussian",
                    dtype=self.dtype,
                ),
                self.dtype,
            ),

            "BPF": lambda: run_bpf(
                Y=self.measurements,
                Np=self.Np,
                prop_fn=self.prop_fn,
                log_likelihood_fn=self.llk_fn,
            ),

            "GSMC": lambda: run_gsmc(
                Y=self.measurements,
                Np=self.Np,
                prop_fn=self.prop_fn,
                log_likelihood_fn=self.llk_fn,
            ),

            "PFPF-EDH": lambda: run_pfpf_fn(
                flow_type="EDH",
                measurement_type="gaussian",
                use_weights=True,
            )(
                Y=self.measurements,
                Np=self.Np,
                P_pred=self.P_pred,
                R_mat=self.R_mat,
                prop_fn=self.prop_fn,
                log_likelihood_fn=self.llk_fn,
                h_func=self.h_func,
                jacobian_func=self.jacobian_func,
            )[:2],

            "PFPF-LEDH": lambda: run_pfpf_fn(
                flow_type="LEDH",
                measurement_type="gaussian",
                use_weights=True,
            )(
                Y=self.measurements,
                Np=self.Np,
                P_pred=self.P_pred,
                R_mat=self.R_mat,
                prop_fn=self.prop_fn,
                log_likelihood_fn=self.llk_fn,
                h_func=self.h_func,
                jacobian_func=self.jacobian_func,
            )[:2],

            "UPF": lambda: run_upf(
                Y=self.measurements,
                Np=self.Np,
                Sigma=self.Sigma,
                alpha=self.alpha,
                nu=200.0,
                gamma=0.0,
                transition_logpdf_fn=self.transition_fn_gauss,
                log_likelihood_fn=self.llk_fn,
                transition_mean_fn=gh_dynamics_mean,
            ),

            "ESRF": lambda: run_esrf(
                Y=self.measurements,
                Np=self.Np,
                F_func=lambda x: prop_linear_gaussian(x, self.F_mat, self.L),
                Q=self.Sigma,
                H_func=lambda x: x,
                R=self.R_mat,
                measurement_type="gaussian",
                dtype=self.dtype,
            ),
        }

    def get_baseline_filters(self):
        return [
            "KF",
            "UKF",
            "BPF",
            "PFPF-EDH",
            "PFPF-LEDH",
            "UPF",
            "ESRF",
            "GSMC",
        ]

class TestSkewTPoissonPipelineIntegration(BasePipelineIntegrationTest):

    def setUp(self):
        tf.random.set_seed(0)

        self.d = 2
        self.T = 10
        self.Np = 30
        self.dtype = tf.float64
        self.measurement_type = "poisson"

        self.alpha = 0.9
        self.nu = 5.0
        self.gamma = tf.constant([0.3] * self.d, dtype=self.dtype)

        self.m1 = 1.0
        self.m2 = 1.0 / 3.0

        self.Sigma = tf.eye(self.d, dtype=self.dtype)
        self.R_mat = tf.eye(self.d, dtype=self.dtype) * 0.3**2

        self.true_states, self.measurements = generate_skt_poi_data(
            T=self.T,
            d=self.d,
            alpha=self.alpha,
            Sigma_proc=self.Sigma,
            gamma=self.gamma,
            nu=self.nu,
            m1=self.m1,
            m2=self.m2,
            dtype=self.dtype,
        )

        @tf.function
        def prop_fn(x):
            return sample_skewed_t_v1(
                x,
                self.alpha,
                self.Sigma,
                self.gamma,
                self.nu,
            )

        @tf.function
        def h_func(x):
            return self.m1 * tf.exp(self.m2 * x)

        @tf.function
        def llk_fn(x, y):
            return log_likelihood_poisson(
                x,
                y,
                m1=self.m1,
                m2=self.m2,
            )

        self.prop_fn = prop_fn
        self.h_func = h_func
        self.llk_fn = llk_fn
        self.jacobian_func = H_jac_t_tf

        self.ekf_result = run_ekf_wrap(
            Y=tf.transpose(self.measurements),
            m0=tf.zeros(self.d, dtype=self.dtype),
            P0=tf.eye(self.d, dtype=self.dtype),
            Q=self.Sigma,
            R=self.R_mat,
            F=lambda x, t: self.alpha * x,
            H=lambda x, t: self.m1 * tf.exp(self.m2 * x),
            F_jac=F_jac_t_tf,
            H_jac=H_jac_t_tf,
            measurement_type="poisson",
        )

        self.P_pred = tf.cast(self.ekf_result["P_pred"], self.dtype)

        self.transition_fn_skewt = make_transition_logpdf_skewt(
            alpha=self.alpha,
            gamma=self.gamma,
            nu=self.nu,
        )

    def get_filter_registry(self):
        return {
            "EKF": lambda: tf.cast(
                tf.transpose(self.ekf_result["mu_filt"]),
                self.dtype,
            ),

            "UKF": lambda: tf.cast(
                run_ukf_wrap(
                    Y=tf.transpose(self.measurements),
                    m0=tf.zeros(self.d, dtype=self.dtype),
                    P0=tf.eye(self.d, dtype=self.dtype),
                    Q=self.Sigma,
                    R=self.R_mat,
                    F=lambda x, t: self.alpha * x,
                    H=lambda x, t: self.m1 * tf.exp(self.m2 * x),
                    measurement_type="poisson",
                ),
                self.dtype,
            ),

            "BPF": lambda: run_bpf(
                Y=self.measurements,
                Np=self.Np,
                prop_fn=self.prop_fn,
                log_likelihood_fn=self.llk_fn,
            ),

            "Block-BPF": lambda: bpf_block(
                Y=self.measurements,
                Np=self.Np,
                prop_fn=self.prop_fn,
                log_likelihood_fn=self.llk_fn,
                resample_threshold=False,
            ),

            "GSMC": lambda: run_gsmc(
                Y=self.measurements,
                Np=self.Np,
                prop_fn=lambda x: tf.map_fn(
                    lambda xi: tf.cast(
                        sample_skewed_t_v1(
                            xi,
                            self.alpha,
                            self.Sigma,
                            self.gamma,
                            self.nu,
                        ),
                        self.dtype,
                    ),
                    x,
                    dtype=self.dtype,
                ),
                log_likelihood_fn=self.llk_fn,
            ),

            "SMHMC": lambda: smhmc_helper(
                Y=self.measurements,
                Np=self.Np,
                prop_fn=lambda x: tf.cast(self.prop_fn(x), self.dtype),
                log_likelihood_fn=self.llk_fn,
                leapfrog_steps=5,
                epsilon=0.03,
                resample_threshold=True,
                dtype=self.dtype,
            ),

            "PFPF-EDH": lambda: run_pfpf_fn(
                flow_type="EDH",
                measurement_type="poisson",
                use_weights=True,
            )(
                Y=self.measurements,
                Np=self.Np,
                P_pred=self.P_pred,
                R_mat=self.R_mat,
                prop_fn=self.prop_fn,
                log_likelihood_fn=self.llk_fn,
                h_func=self.h_func,
                jacobian_func=self.jacobian_func,
            )[:2],

            "PFPF-LEDH": lambda: run_pfpf_fn(
                flow_type="LEDH",
                measurement_type="poisson",
                use_weights=True,
            )(
                Y=self.measurements,
                Np=self.Np,
                P_pred=self.P_pred,
                R_mat=self.R_mat,
                prop_fn=self.prop_fn,
                log_likelihood_fn=self.llk_fn,
                h_func=self.h_func,
                jacobian_func=self.jacobian_func,
            )[:2],

            "UPF": lambda: run_upf(
                Y=self.measurements,
                Np=self.Np,
                Sigma=self.Sigma,
                alpha=self.alpha,
                nu=self.nu,
                gamma=self.gamma,
                transition_logpdf_fn=self.transition_fn_skewt,
                log_likelihood_fn=self.llk_fn,
                transition_mean_fn=gh_dynamics_mean,
            ),

            "ESRF": lambda: run_esrf(
                Y=self.measurements,
                Np=self.Np,
                F_func=lambda x: tf.map_fn(
                    lambda xi: tf.cast(
                        sample_skewed_t_v1(
                            xi,
                            self.alpha,
                            self.Sigma,
                            self.gamma,
                            self.nu,
                        ),
                        self.dtype,
                    ),
                    x,
                    dtype=self.dtype,
                ),
                Q=self.Sigma,
                H_func=self.h_func,
                R=self.R_mat,
                measurement_type="poisson",
                dtype=self.dtype,
            ),
        }

    def get_baseline_filters(self):
        return [
            "EKF",
            "UKF",
            "BPF",
            "PFPF-EDH",
            "PFPF-LEDH",
            "UPF",
            "ESRF",
            "GSMC",
        ]

    def get_extended_filters(self):
        return [
            "Block-BPF",
            "SMHMC",
        ]



class TestRunPFPFWrapper(unittest.TestCase):
    """Wrapper baseline: verify that the factory builds and executes PFPF in standard configurations."""

    @classmethod
    def setUpClass(cls):
        cls._old_eager = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(True)

    @classmethod
    def tearDownClass(cls):
        tf.config.run_functions_eagerly(cls._old_eager)

    def setUp(self):
        tf.random.set_seed(0)

        self.dtype = tf.float32
        self.T = 4
        self.d = 2
        self.Np = 6

        self.Y = tf.random.normal((self.T, self.d), dtype=self.dtype)
        self.P_pred = tf.eye(self.d, batch_shape=[self.T], dtype=self.dtype)
        self.R = tf.eye(self.d, dtype=self.dtype)

        self.prop_noise_bank = tf.zeros((self.T, self.Np, self.d), dtype=self.dtype)

        # --------------------------------------------------
        # Base functions compatible with the wrapper
        # --------------------------------------------------
        def prop_fn(x):
            x = tf.cast(x, self.dtype)
            return x + 0.1

        def log_likelihood_fn(particles, y):
            particles = tf.cast(particles, self.dtype)
            y = tf.cast(y, self.dtype)
            return -tf.reduce_sum((particles - y) ** 2, axis=1)

        def h_func(x, t=None):
            x = tf.cast(x, self.dtype)
            return tf.identity(x)

        def jacobian_func(x, t=None):
            return tf.eye(self.d, dtype=self.dtype)

        def prop_fn_stoch_drift(x, t, eps=None):
            x = tf.cast(x, self.dtype)
            drift = x + 0.1 * tf.cast(t + 1, self.dtype)
            if eps is not None:
                drift = drift + tf.cast(eps, self.dtype)
            return drift

        def identity_resampling(particles, weights):
            return particles, weights

        def loglik_weight_helper(log_like, theta, Np, dtype):
            weights = tf.ones((Np,), dtype=dtype) / tf.cast(Np, dtype)
            loglik_t = tf.reduce_mean(log_like)
            return weights, loglik_t

        self.prop_fn = prop_fn
        self.log_likelihood_fn = log_likelihood_fn
        self.h_func = h_func
        self.jacobian_func = jacobian_func
        self.prop_fn_stoch_drift = prop_fn_stoch_drift
        self.identity_resampling = identity_resampling
        self.loglik_weight_helper = loglik_weight_helper

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def build_pfpf(self, **kwargs):
        """Build the callable returned by the wrapper using a stable default configuration."""
        defaults = dict(
            flow_type="EDH",
            measurement_type="gaussian",
            use_weights=True,
            beta=None,
            resampling_fn=None,
            diagnostics=True,
            prop_fn_stoch_drift=None,
            use_fixed_prop_noise=False,
            prop_noise_bank=None,
            loglik_weight_helper=None,
            collect_resampling_examples=None,
            compile_tf=False,
        )
        defaults.update(kwargs)
        return run_pfpf_fn(**defaults)

    def run_pfpf(self, pfpf_fn, **call_kwargs):
        """Execute the wrapper callable with standard inputs, allowing targeted call-time overrides."""
        defaults = dict(
            Y=self.Y,
            Np=self.Np,
            P_pred=self.P_pred,
            R_mat=self.R,
            prop_fn=self.prop_fn,
            log_likelihood_fn=self.log_likelihood_fn,
            h_func=self.h_func,
            jacobian_func=self.jacobian_func,
        )
        defaults.update(call_kwargs)
        return pfpf_fn(**defaults)

    def assert_valid_outputs(self, ests, ess, particles, diag, use_weights=True, diagnostics_expected=True):
        """Minimal sanity check on the wrapper outputs."""
        self.assertEqual(tuple(ests.shape), (self.T, self.d))
        self.assertTrue(np.all(np.isfinite(ests.numpy())))

        self.assertEqual(tuple(particles.shape), (self.Np, self.d))
        self.assertTrue(np.all(np.isfinite(particles.numpy())))

        if use_weights:
            self.assertIsNotNone(ess)
            self.assertEqual(tuple(ess.shape), (self.T,))
            self.assertTrue(np.all(np.isfinite(ess.numpy())))
        else:
            self.assertIsNone(ess)

        self.assertIsInstance(diag, dict)
        self.assertIn("loglik", diag)
        self.assertTrue(np.isfinite(float(diag["loglik"].numpy())))

        if diagnostics_expected:
            for key in ["flow_norm", "spec_J", "cond_J", "logdet_J"]:
                self.assertIn(key, diag)
                self.assertEqual(diag[key].shape[0], self.T)
                self.assertTrue(np.all(np.isfinite(diag[key].numpy())))

    # --------------------------------------------------
    # Baseline wrapper tests
    # --------------------------------------------------
    def test_factory_returns_callable(self):
        """The factory should return a callable function."""
        pfpf_fn = self.build_pfpf()
        self.assertTrue(callable(pfpf_fn))

    def test_factory_returns_callable_with_compile_tf_true(self):
        """Smoke test: the factory should return a callable in compiled mode"""
        pfpf_fn = run_pfpf_fn(
            flow_type="EDH",
            measurement_type="gaussian",
            compile_tf=True
        )
        self.assertTrue(callable(pfpf_fn))

    def test_edh_runs(self):
        """Weighted baseline EDH configuration: the wrapper should execute and return finite outputs."""
        pfpf_fn = self.build_pfpf(
            flow_type="EDH",
            use_weights=True,
            beta=tf.constant([0.5, 0.5, 0.5], dtype=self.dtype),
            diagnostics=True,
        )

        ests, ess, particles, diag = self.run_pfpf(pfpf_fn)
        self.assert_valid_outputs(ests, ess, particles, diag, use_weights=True, diagnostics_expected=True)

    def test_ledh_runs(self):
        """Weighted baseline LEDH configuration: the wrapper should execute and return finite outputs."""
        pfpf_fn = self.build_pfpf(
            flow_type="LEDH",
            use_weights=True,
            diagnostics=True,
        )

        ests, ess, particles, diag = self.run_pfpf(pfpf_fn)
        self.assert_valid_outputs(ests, ess, particles, diag, use_weights=True, diagnostics_expected=True)

    def test_no_weights_runs(self):
        """With use_weights=False the wrapper should execute and return ESS=None."""
        pfpf_fn = self.build_pfpf(
            flow_type="EDH",
            use_weights=False,
            diagnostics=False,
        )

        ests, ess, particles, diag = self.run_pfpf(pfpf_fn)
        self.assert_valid_outputs(ests, ess, particles, diag, use_weights=False, diagnostics_expected=False)

    def test_beta_configuration_runs(self):
        """The beta parameter passed to the factory should be accepted and lead to a valid execution."""
        pfpf_fn = self.build_pfpf(
            beta=tf.constant([0.5, 0.5, 0.5], dtype=self.dtype),
            diagnostics=False,
        )

        ests, ess, particles, diag = self.run_pfpf(pfpf_fn)
        self.assert_valid_outputs(ests, ess, particles, diag, use_weights=True, diagnostics_expected=False)

    def test_invalid_np_raises(self):
        """The wrapper should raise error when Np is invalid."""
        pfpf_fn = self.build_pfpf()

        with self.assertRaises(ValueError):
            self.run_pfpf(pfpf_fn, Np=1)

    def test_invalid_flow_type_raises(self):
        """Invalid flow_type should fail at execution time."""
        pfpf_fn = self.build_pfpf(flow_type="BAD")

        with self.assertRaises(ValueError):
            self.run_pfpf(pfpf_fn)

    def test_invalid_measurement_type_raises(self):
        """Invalid measurement_type should fail at execution time."""
        pfpf_fn = self.build_pfpf(measurement_type="bad")

        with self.assertRaises(ValueError):
            self.run_pfpf(pfpf_fn)



class TestRunPFPFWrapperResampling(TestRunPFPFWrapper):
    """Wrapper tests for optional resampling: verify wiring and execution of resampling paths."""

    def setUp(self):
        super().setUp()

        # OT resampling used as a nontrivial resampling path
        self.ot_resampling = lambda p, w: soft_resample_ot(
            p, w,
            epsilon=tf.constant(0.1, dtype=self.dtype),
            sinkhorn_iters=2
        )

    def test_runs_with_identity_resampling(self):
        """The wrapper should run with an identity resampler."""
        pfpf_fn = self.build_pfpf(
            resampling_fn=self.identity_resampling,
            diagnostics=False,
        )

        ests, ess, particles, diag = self.run_pfpf(pfpf_fn)
        self.assert_valid_outputs(
            ests, ess, particles, diag,
            use_weights=True,
            diagnostics_expected=False
        )

    def test_runs_with_ot_resampling(self):
        """The wrapper should run with OT resampling and return finite outputs."""
        pfpf_fn = self.build_pfpf(
            resampling_fn=self.ot_resampling,
            diagnostics=False,
        )

        ests, ess, particles, diag = self.run_pfpf(pfpf_fn)
        self.assert_valid_outputs(
            ests, ess, particles, diag,
            use_weights=True,
            diagnostics_expected=False
        )

    def test_resampling_path_is_used(self):
        """The wrapper should actually call the provided resampling function."""
        calls = {"n": 0}

        def tracking_resampling(particles, weights):
            calls["n"] += 1
            return particles, weights

        pfpf_fn = self.build_pfpf(
            resampling_fn=tracking_resampling,
            diagnostics=False,
        )

        ests, ess, particles, diag = self.run_pfpf(pfpf_fn)

        self.assert_valid_outputs(
            ests, ess, particles, diag,
            use_weights=True,
            diagnostics_expected=False
        )

        # Resampling should be called once per time step when weights are used
        self.assertEqual(calls["n"], self.T)


class TestRunPFPFWrapperTimeDependentExogVariable(TestRunPFPFWrapper):
    """Wrapper extensions for time-dependent (i.e. inclusion of an exogenous variable) propagation through prop_fn_stoch_drift."""

    def test_runs_with_stochastic_drift(self):
        """The wrapper should execute when time-dependent propagation is enabled."""
        pfpf_fn = self.build_pfpf(
            prop_fn_stoch_drift=self.prop_fn_stoch_drift,
            diagnostics=False,
        )

        ests, ess, particles, diag = self.run_pfpf(pfpf_fn)
        self.assert_valid_outputs(ests, ess, particles, diag, use_weights=True, diagnostics_expected=False)

    def test_time_dependent_effect(self):
        """Inclusion of a time-dependent drift propagation should produce propagation outputs different from state-only dependent drift."""
        pfpf_td = self.build_pfpf(
            prop_fn_stoch_drift=self.prop_fn_stoch_drift,
            diagnostics=False,
        )
        pfpf_static = self.build_pfpf(
            prop_fn_stoch_drift=None,
            diagnostics=False,
        )

        ests_td, _, _, _ = self.run_pfpf(pfpf_td)
        ests_static, _, _, _ = self.run_pfpf(pfpf_static)

        self.assertFalse(np.allclose(ests_td.numpy(), ests_static.numpy()))


class TestRunPFPFWrapperFixedNoise(TestRunPFPFWrapper):
    """Wrapper extensions for fixed noise propagation under time-dependent drift."""

    def test_fixed_prop_noise_requires_noise_bank(self):
        """If use_fixed_prop_noise=True without prop_noise_bank, the wrapper should fail."""
        pfpf_fn = self.build_pfpf(
            prop_fn_stoch_drift=self.prop_fn_stoch_drift,
            use_fixed_prop_noise=True,
            prop_noise_bank=None,
        )

        with self.assertRaises(ValueError):
            self.run_pfpf(pfpf_fn)

    def test_fixed_prop_noise_requires_stochastic_drift(self):
        """If use_fixed_prop_noise=True without stochastic drift, the wrapper should fail."""
        pfpf_fn = self.build_pfpf(
            prop_fn_stoch_drift=None,
            use_fixed_prop_noise=True,
            prop_noise_bank=self.prop_noise_bank,
        )

        with self.assertRaises(ValueError):
            self.run_pfpf(pfpf_fn)

    def test_fixed_prop_noise_runs(self):
        """With valid stochastic drift and noise bank, the fixed-noise path should execute."""
        pfpf_fn = self.build_pfpf(
            prop_fn_stoch_drift=self.prop_fn_stoch_drift,
            use_fixed_prop_noise=True,
            prop_noise_bank=self.prop_noise_bank,
            diagnostics=False,
        )

        ests, ess, particles, diag = self.run_pfpf(pfpf_fn)
        self.assert_valid_outputs(ests, ess, particles, diag, use_weights=True, diagnostics_expected=False)


class TestRunPFPFWrapperHelpers(TestRunPFPFWrapper):
    """Wrapper extensions for optional helpers and collection of resampling examples."""

    def test_runs_with_loglik_weight_helper(self):
        """The wrapper should accept a custom helper for weights/log-likelihood."""
        pfpf_fn = self.build_pfpf(
            loglik_weight_helper=self.loglik_weight_helper,
            diagnostics=False,
        )

        ests, ess, particles, diag = self.run_pfpf(pfpf_fn)
        self.assert_valid_outputs(ests, ess, particles, diag, use_weights=True, diagnostics_expected=False)

    def test_collect_resampling_examples_requires_compile_tf_false(self):
        """Python-side example collection is not allowed with compile_tf=True."""
        with self.assertRaises(ValueError):
            run_pfpf_fn(
                flow_type="EDH",
                measurement_type="gaussian",
                collect_resampling_examples=[],
                compile_tf=True,
            )

    def test_collect_resampling_examples_runs(self):
        """With compile_tf=False, resampling example collection should work."""
        collected = []

        pfpf_fn = self.build_pfpf(
            collect_resampling_examples=collected,
            resampling_fn=self.identity_resampling,
            diagnostics=False,
            compile_tf=False,
        )

        ests, ess, particles, diag = self.run_pfpf(pfpf_fn)
        self.assert_valid_outputs(ests, ess, particles, diag, use_weights=True, diagnostics_expected=False)

        self.assertEqual(len(collected), self.T)
        for p, w in collected:
            self.assertEqual(tuple(p.shape), (self.Np, self.d))
            self.assertEqual(tuple(w.shape), (self.Np,))
            self.assertTrue(np.all(np.isfinite(p.numpy())))
            self.assertTrue(np.all(np.isfinite(w.numpy())))


@unittest.skip("Outdated pipeline API test: final pipeline does not use model_type")
class TestRunEDHLEDHPipeline(unittest.TestCase):
    """Integration tests for EDH/LEDH pipeline under Poisson and Gaussian models with diagnostics."""

    def _check_common_config(self, out, d, T, Np, expected_names, model_type):
        self.assertIn("config", out)
        self.assertIn("accuracy_mc", out)
        self.assertIn("diagnostics_mc", out)

        cfg = out["config"]

        self.assertEqual(cfg["d"], d)
        self.assertEqual(cfg["T"], T)
        self.assertEqual(cfg["Np"], Np)
        self.assertEqual(cfg["model_type"], model_type)

        self.assertEqual(cfg["Sigma"].shape, (d, d))
        self.assertEqual(cfg["Q"].shape, (d, d))
        self.assertEqual(cfg["R"].shape, (d, d))

        for name in expected_names:
            self.assertIn(name, cfg["filters_config"])
            self.assertIn(name, cfg["filter_fns"])
            self.assertTrue(callable(cfg["filters_config"][name]))
            self.assertTrue(callable(cfg["filter_fns"][name]))

    def _check_accuracy_output(self, acc, names, T):
        for name in names:
            self.assertIn(name, acc)
            self.assertIn("mse_t", acc[name])
            self.assertIn("ess_mean", acc[name])
            self.assertIn("lost_tracks", acc[name])
            self.assertIn("run_time", acc[name])

            self.assertEqual(acc[name]["mse_t"].shape, (T,))
            self.assertTrue(tf.reduce_all(tf.math.is_finite(acc[name]["mse_t"])).numpy())

    def _check_diagnostics_output(self, diag, names, N_MC):
        for name in names:
            self.assertIn(name, diag)

            self.assertIn("ests", diag[name])
            self.assertIn("ESS", diag[name])
            self.assertIn("particles", diag[name])
            self.assertIn("diagnostics", diag[name])

            self.assertEqual(diag[name]["ests"].shape[0], N_MC)
            self.assertEqual(diag[name]["particles"].shape[0], N_MC)

            self.assertTrue(
                tf.reduce_all(tf.math.is_finite(diag[name]["ests"])).numpy()
            )

            self.assertIn("flow_norm", diag[name]["diagnostics"])
            self.assertIn("cond_J", diag[name]["diagnostics"])
            self.assertIn("spec_J", diag[name]["diagnostics"])

    def test_poisson_pipeline_runs_accuracy_and_diagnostics(self):
        out = run_edh_ledh_pipeline(
            d=2,
            T=3,
            Np=5,
            N_MC_accuracy=2,
            N_MC_diagnostics=2,
            alpha=0.9,
            sigma_z=1.0,
            gamma=0.3,
            nu=5,
            m1=1.0,
            m2=1.0 / 3.0,
            beta=None,
            seed=123,
            methods=[("EDH", False), ("LEDH", False)],
            run_accuracy=True,
            run_diagnostics=True,
            model_type="poisson",
        )

        names = ["EDH", "LEDH"]

        self._check_common_config(
            out=out,
            d=2,
            T=3,
            Np=5,
            expected_names=names,
            model_type="poisson",
        )

        cfg = out["config"]
        self.assertIsNotNone(cfg["gamma_vec"])
        self.assertEqual(cfg["gamma_vec"].shape, (2,))
        self.assertEqual(cfg["params"]["m1"], 1.0)
        self.assertEqual(cfg["params"]["m2"], 1.0 / 3.0)

        self._check_accuracy_output(out["accuracy_mc"], names, T=3)
        self._check_diagnostics_output(out["diagnostics_mc"], names, N_MC=2)

    def test_gaussian_pipeline_runs_accuracy_and_diagnostics(self):
        out = run_edh_ledh_pipeline(
            d=2,
            T=3,
            Np=5,
            N_MC_accuracy=2,
            N_MC_diagnostics=2,
            alpha=0.9,
            sigma_z=1.0,
            beta=None,
            seed=123,
            methods=[("EDH", False), ("LEDH", False)],
            run_accuracy=True,
            run_diagnostics=True,
            model_type="gaussian",
        )

        names = ["EDH", "LEDH"]

        self._check_common_config(
            out=out,
            d=2,
            T=3,
            Np=5,
            expected_names=names,
            model_type="gaussian",
        )

        cfg = out["config"]
        self.assertIsNone(cfg["gamma_vec"])
        self.assertIn("F_jac_fn", cfg)

        self._check_accuracy_output(out["accuracy_mc"], names, T=3)
        self._check_diagnostics_output(out["diagnostics_mc"], names, N_MC=2)

    def test_pipeline_default_methods_create_all_four_filters_poisson(self):
        out = run_edh_ledh_pipeline(
            d=2,
            T=3,
            Np=5,
            N_MC_accuracy=1,
            N_MC_diagnostics=1,
            seed=123,
            run_accuracy=False,
            run_diagnostics=False,
            model_type="poisson",
        )

        expected = {"EDH", "wEDH", "LEDH", "wLEDH"}
        cfg = out["config"]

        self.assertEqual(set(cfg["filters_config"].keys()), expected)
        self.assertEqual(set(cfg["filter_fns"].keys()), expected)
        self.assertIsNone(out["accuracy_mc"])
        self.assertIsNone(out["diagnostics_mc"])

    def test_pipeline_default_methods_create_all_four_filters_gaussian(self):
        out = run_edh_ledh_pipeline(
            d=2,
            T=3,
            Np=5,
            N_MC_accuracy=1,
            N_MC_diagnostics=1,
            seed=123,
            run_accuracy=False,
            run_diagnostics=False,
            model_type="gaussian",
        )

        expected = {"EDH", "wEDH", "LEDH", "wLEDH"}
        cfg = out["config"]

        self.assertEqual(set(cfg["filters_config"].keys()), expected)
        self.assertEqual(set(cfg["filter_fns"].keys()), expected)
        self.assertEqual(cfg["model_type"], "gaussian")
        self.assertIsNone(out["accuracy_mc"])
        self.assertIsNone(out["diagnostics_mc"])

    def test_pipeline_runs_only_accuracy(self):
        out = run_edh_ledh_pipeline(
            d=2,
            T=3,
            Np=5,
            N_MC_accuracy=2,
            seed=123,
            methods=[("EDH", True)],
            run_accuracy=True,
            run_diagnostics=False,
            model_type="poisson",
        )

        self.assertIsNotNone(out["accuracy_mc"])
        self.assertIsNone(out["diagnostics_mc"])
        self.assertIn("wEDH", out["accuracy_mc"])

    def test_pipeline_runs_only_diagnostics(self):
        out = run_edh_ledh_pipeline(
            d=2,
            T=3,
            Np=5,
            N_MC_diagnostics=2,
            seed=123,
            methods=[("LEDH", True)],
            run_accuracy=False,
            run_diagnostics=True,
            model_type="poisson",
        )

        self.assertIsNone(out["accuracy_mc"])
        self.assertIsNotNone(out["diagnostics_mc"])
        self.assertIn("wLEDH", out["diagnostics_mc"])

    def test_poisson_pipeline_accepts_different_m1_m2_and_beta(self):
        out = run_edh_ledh_pipeline(
            d=2,
            T=3,
            Np=5,
            N_MC_accuracy=1,
            N_MC_diagnostics=1,
            m1=2.0,
            m2=0.5,
            beta=0.7,
            seed=123,
            methods=[("EDH", False)],
            run_accuracy=False,
            run_diagnostics=False,
            model_type="poisson",
        )

        params = out["config"]["params"]

        self.assertEqual(params["m1"], 2.0)
        self.assertEqual(params["m2"], 0.5)
        self.assertEqual(params["beta"], 0.7)
        self.assertEqual(params["model_type"], "poisson")
        self.assertIn("EDH", out["config"]["filters_config"])

    def test_invalid_model_type_raises(self):
        with self.assertRaises(ValueError):
            run_edh_ledh_pipeline(
                d=2,
                T=3,
                Np=5,
                run_accuracy=False,
                run_diagnostics=False,
                model_type="bad_model",
            )



class TestHuPFFMonteCarloIntegration(unittest.TestCase):
    """Integration tests for Hu-PFF scalar/matrix runners and Monte Carlo wrappers."""

    def setUp(self):
        tf.random.set_seed(0)
        np.random.seed(0)

        self.T = 4
        self.d = 2
        self.Np = 8
        self.N_MC = 2
        self.dtype = tf.float32

        self.Y = tf.random.normal((self.T, self.d), dtype=self.dtype)
        self.X_true = tf.random.normal((self.T, self.d), dtype=self.dtype)

        self.Sigma_tf = tf.eye(self.d, dtype=self.dtype)

        def prop_fn(x):
            return x

        self.prop_fn = prop_fn

        self.loglik_grad_fn = make_poisson_grad_wrapper(
            m1=1.0,
            m2=1.0 / 3.0
        )

        self.hu_runners = make_hu_pff_runners(
            Sigma_tf=self.Sigma_tf,
            prop_fn=self.prop_fn,
            loglik_grad_fn=self.loglik_grad_fn,
            n_steps=2,
            eps=0.01,
        )

    def _check_filter_output(self, ests, particles, diagnostics):
        self.assertEqual(ests.shape, (self.T, self.d))
        self.assertEqual(particles.shape, (self.T, self.Np, self.d))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(ests)).numpy())
        self.assertTrue(tf.reduce_all(tf.math.is_finite(particles)).numpy())

        self.assertIn("flow_norm", diagnostics)
        self.assertIn("grad_cond", diagnostics)
        self.assertIn("spec_J", diagnostics)

        self.assertEqual(diagnostics["flow_norm"].shape, (self.T, self.Np))
        self.assertEqual(diagnostics["grad_cond"].shape, (self.T,))
        self.assertEqual(diagnostics["spec_J"].shape, (self.T,))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(diagnostics["flow_norm"])).numpy())
        self.assertTrue(tf.reduce_all(tf.math.is_finite(diagnostics["grad_cond"])).numpy())
        self.assertTrue(tf.reduce_all(tf.math.is_finite(diagnostics["spec_J"])).numpy())

    def test_hu_scalar_filter_runs(self):
        ests, particles, diagnostics = self.hu_runners["scalar"](
            self.Y,
            Np=self.Np
        )

        self._check_filter_output(ests, particles, diagnostics)

    def test_hu_matrix_filter_runs(self):
        ests, particles, diagnostics = self.hu_runners["matrix"](
            self.Y,
            Np=self.Np
        )

        self._check_filter_output(ests, particles, diagnostics)

    def test_monte_carlo_final_hu_matrix_runs(self):
        out = monte_carlo_final(
            filter_fn=self.hu_runners["matrix"],
            Y=self.Y,
            N_MC=self.N_MC,
            output_names=["ests", "particles", "diagnostics"],
            Np=self.Np
        )

        self.assertIn("ests", out)
        self.assertIn("particles", out)
        self.assertIn("diagnostics", out)

        self.assertEqual(out["ests"].shape, (self.N_MC, self.T, self.d))
        self.assertEqual(out["particles"].shape, (self.N_MC, self.T, self.Np, self.d))

        self.assertIn("flow_norm", out["diagnostics"])
        self.assertIn("grad_cond", out["diagnostics"])
        self.assertIn("spec_J", out["diagnostics"])

        self.assertEqual(out["diagnostics"]["flow_norm"].shape, (self.N_MC, self.T, self.Np))
        self.assertEqual(out["diagnostics"]["grad_cond"].shape, (self.N_MC, self.T))
        self.assertEqual(out["diagnostics"]["spec_J"].shape, (self.N_MC, self.T))

        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["ests"])).numpy())
        self.assertTrue(tf.reduce_all(tf.math.is_finite(out["particles"])).numpy())

    def test_monte_carlo_final_hu_scalar_runs(self):
        out = monte_carlo_final(
            filter_fn=self.hu_runners["scalar"],
            Y=self.Y,
            N_MC=self.N_MC,
            output_names=["ests", "particles", "diagnostics"],
            Np=self.Np
        )

        self.assertEqual(out["ests"].shape, (self.N_MC, self.T, self.d))
        self.assertEqual(out["particles"].shape, (self.N_MC, self.T, self.Np, self.d))
        self.assertIn("flow_norm", out["diagnostics"])
        self.assertIn("grad_cond", out["diagnostics"])
        self.assertIn("spec_J", out["diagnostics"])

    def test_run_monte_carlo_sim_with_hu_runs(self):
        filter_config_hu = {
            "matrix-PFF": lambda: self.hu_runners["matrix"](
                self.Y,
                Np=self.Np
            )[0],
            "scalar-PFF": lambda: self.hu_runners["scalar"](
                self.Y,
                Np=self.Np
            )[0],
        }

        results = run_monte_carlo_sim(
            filters_config=filter_config_hu,
            true_data=self.X_true,
            N_MC=self.N_MC,
            monte_carlo_fn=monte_carlo_light_lost
        )

        self.assertIn("matrix-PFF", results)
        self.assertIn("scalar-PFF", results)

        for name in ["matrix-PFF", "scalar-PFF"]:
            out = results[name]

            self.assertIn("mse_t", out)
            self.assertIn("ess_mean", out)
            self.assertIn("lost_tracks", out)
            self.assertIn("run_time", out)

            self.assertEqual(out["mse_t"].shape, (self.T,))
            self.assertTrue(tf.reduce_all(tf.math.is_finite(out["mse_t"])).numpy())

    def test_monte_carlo_final_invalid_output_names_length_raises(self):
        with self.assertRaises(ValueError):
            monte_carlo_final(
                filter_fn=self.hu_runners["scalar"],
                Y=self.Y,
                N_MC=self.N_MC,
                output_names=["ests", "particles"],
                Np=self.Np
            )

    def test_poisson_gradient_wrapper_changes_with_m1_m2(self):
        particles = tf.ones((self.Np, self.d), dtype=self.dtype)
        y = tf.ones((self.d,), dtype=self.dtype) * 2.0

        grad_default = make_poisson_grad_wrapper(
            m1=1.0,
            m2=1.0 / 3.0
        )(particles, y)

        grad_alt = make_poisson_grad_wrapper(
            m1=2.0,
            m2=0.5
        )(particles, y)

        self.assertEqual(grad_default.shape, (self.Np, self.d))
        self.assertEqual(grad_alt.shape, (self.Np, self.d))
        self.assertFalse(np.allclose(grad_default.numpy(), grad_alt.numpy()))



# ---------------------------------------------------------------------
# Suite construction
# ---------------------------------------------------------------------
def _build_suite(class_names):
    """Build a suite from class names. Missing classes raise immediately."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for class_name in class_names:
        test_case = globals()[class_name]
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    return suite


def _run_suite_to_file(class_names, filename, also_stdout=False):
    suite = _build_suite(class_names)
    if also_stdout:
        unittest.TextTestRunner(verbosity=2).run(suite)
        suite = _build_suite(class_names)

    with open(filename, "w", encoding="utf-8") as f:
        result = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)

    if not result.wasSuccessful():
        raise SystemExit(f"Test suite failed: {filename}")

    return result



UNIT_FILTERS_CLASSES = [
    "TestKFKernels",
    "TestKFFilterWrappers",
    "TestESRFFilter",
    "TestUPFParticleFilter",
    "TestGSMCParticleFilter",
    "TestBPFBlock",
    "TestHMCHelper",
    "TestSMHMCParticleFilter",
    "TestParticleFlowUpdate",
    "TestParticleFlowPFPropagation",
    "TestPFFHuFilter",
    "TestHuFactories",
    "TestMakeFlowFunctions",
]

UNIT_UTILS_CLASSES = [
    "TestComputeSpectralNorm",
    "TestMakeBetaSchedule",
    "TestPairwiseDistance",
]

UNIT_SIM_MODEL_CLASSES = [
    "TestSimHDLGSSM",
    "TestSimSkewTPoisson",
    "TestComputeSigma",
    "TestJacobiansEKF",
    "TestLogLikelihood",
    "TestPropagationFcts",
    "TestTransitionLogPDFupf",
    "TestGaussianLogPDF",
    "TestMakeTransitionLogpdf",
    "TestPropagationGSMC",
    "TestLogLikPoissonGrad",
]

UNIT_DBPF_CLASSES = [
    "TestMixtureUniformMultinomialResampling",
    "TestNoResampling",
    "TestSoftResamplingPFNet",
    "TestOTResampling",
    "TestSinkhornLog",
    "TestOTEntropyTuningUnit",   
    "TestKalmanLoglikAlpha",
    "TestPropFnAlpha",
    "TestLogLikelihoodGaussian",
    "TestMakePropFn",
    "TestMakeLlkFn",
]

UNIT_DAI_CLASSES = [
    "TestComputeH",
    "TestComputeJ",
    "TestOptimBeta",
]

INTEGRATION_PIPELINE_CLASSES = [
    "TestMonteCarloLightLost",
    "TestRunMonteCarloSim",
    "TestMonteCarloFinal",
    "TestGaussianPipelineIntegration",
    "TestSkewTPoissonPipelineIntegration",
    "TestRunEDHLEDHPipeline",
    "TestHuPFFMonteCarloIntegration",
    "TestRunPFPFWrapper",
    "TestOTEntropyTuningIntegrationLG",   
]


def _run_suite_to_file(class_names, filename):
    suite = unittest.TestSuite()

    for name in class_names:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name, module=__name__))

    with open(filename, "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        runner.run(suite)




def run_all_suites():
    """Run all submission suites and save logs separately."""
    _run_suite_to_file(UNIT_FILTERS_CLASSES, "test_unit_filters.txt")
    _run_suite_to_file(UNIT_UTILS_CLASSES, "test_unit_utils.txt")
    _run_suite_to_file(UNIT_SIM_MODEL_CLASSES, "test_unit_sim_model_comps.txt")
    _run_suite_to_file(UNIT_DBPF_CLASSES, "test_unit_dbpf.txt")
    _run_suite_to_file(UNIT_DAI_CLASSES, "test_unit_dai_beta.txt")
    _run_suite_to_file(INTEGRATION_PIPELINE_CLASSES, "test_integration_PF_PFPF_pipeline.txt")



if __name__ == "__main__":
    unittest.main(verbosity=1)

#if __name__ == "__main__":
#    run_all_suites()