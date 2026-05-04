# SV Models, EKF/UKF Diagnostics and Tests

## 📌 Overview

This module extends the project with:

* Multivariate Stochastic Volatility (SV) models  
* Particle Filters (BPF)  
* EKF / UKF implementations  
* Diagnostic tools for nonlinear filtering  
* Unit tests for robustness  

---

## 📁 File Responsibilities

### `multiv_sv_bpf_core.py`
Core SV model and particle filtering utilities (parameters, covariance, resampling).

### `multiv_ekf_ukf_core.py`
Low-level EKF/UKF implementation (sigma points, prediction, update).

### `sv_experiments.py`
Runs experiments, computes RMSE/bias, and benchmarks performance.

### `run_sv_experiments.py`
Entry point to execute SV experiments and generate plots.

### `ekf_ukf_diagnostics.py`
Diagnostic tools for EKF/UKF (Jacobian, sigma points, failure analysis).

### `run_ekf_ukf_diagnostics.py`
Runs diagnostic experiments and produces plots (heatmaps, ACF, PACF).

### `test_helpers.py`
Shared utilities for testing (assertions, validation helpers).

### `tests_multiv_sv.py`
Unit tests for SV model and particle filtering components.

### `test_ekf_ukf_diagnostics_.py`
Unit tests for EKF/UKF and diagnostics.

---

## 🚀 How to Run

Run SV experiments:
```bash
python run_sv_experiments.py
```

Run EKF/UKF diagnostics:
```bash
python run_ekf_ukf_diagnostics.py
```

Run tests:
```bash
python -m unittest tests_multiv_sv.py
python -m unittest test_ekf_ukf_diagnostics_.py
```

---

## ✅ Summary

* Multivariate SV filtering framework  
* EKF, UKF, and particle filters  
* Diagnostic tools for nonlinear models  
* Fully tested and benchmarked  
