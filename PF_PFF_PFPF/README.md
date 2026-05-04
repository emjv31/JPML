# Project Description and File Structure

## 📌 Overview

This project implements and evaluates a range of **state-space filtering methods**, including:

* Classical filters (KF, EKF, UKF)
* Particle filters (BPF, UPF, GSMC)
* Particle Flow Filters (Hu-type flows)
* Monte Carlo evaluation pipelines

The repository is structured so that each file handles a **specific component of the pipeline**, and together they form a complete simulation–filtering–evaluation workflow.

---

## 📁 File Responsibilities

### 🔹 `simulator_model_comps.py`

**Purpose:** Data generation and model definitions

**Expected behavior:**

* Generates synthetic state-space data (e.g., linear Gaussian models)
* Provides likelihood functions (e.g., Gaussian log-likelihood)
* Defines transition dynamics and helper utilities

This file defines the **models that all filters operate on** 

---

### 🔹 `Hu_filters_utils.py`

**Purpose:** Particle Flow Filter components

**Expected behavior:**

* Defines kernel functions (scalar and matrix kernels)
* Computes particle interactions (attraction and repulsion terms)
* Builds flow dynamics used in particle flow filters

These functions implement **nonlinear transformations of particles** using kernel-based flows 

---

### 🔹 `replicate_Li_filters.py`

**Purpose:** Filtering algorithms

**Expected behavior:**

* Implements classical filters (KF, EKF, UKF)
* Provides prediction and update steps
* Includes ensemble-based filtering (e.g., ESRF)
* Contains particle filtering utilities (UPF)

This file contains the **core filtering logic** used across experiments 

---

### 🔹 `replicate_Dai.py`

**Purpose:** Beta scheduling and numerical strategies

**Expected behavior:**

* Generates annealing schedules (beta sequences)
* Used in progressive filtering / homotopy methods
* Controls transition between prior and posterior

---

### 🔹 `metric_utils.py`

**Purpose:** Evaluation and benchmarking

**Expected behavior:**

* Runs Monte Carlo simulations
* Computes metrics such as:

  * Mean Squared Error (MSE)
  * Effective Sample Size (ESS)
* Detects unstable runs ("lost tracks")

Provides **quantitative comparison between filters** 

---

### 🔹 `replicate_Hu_plots.py`

**Purpose:** Plotting utilities for Hu filter experiments  

**Expected behavior:**

* Generates a small set of predefined plots for a given model configuration
* Visualizes selected results of Hu-type particle flow filters
* Used for quick inspection and comparison of outcomes

This file is used to **produce a limited number of diagnostic plots** for the Hu filter experiments.

---

### 🔹 `run_pipeline.py`

**Purpose:** Execution entry point

**Expected behavior:**

* Launches filtering experiments
* Calls simulation, filtering, and evaluation modules
* Produces results for analysis

This is the **recommended file to run the project**.

---

### 🔹 `execution_pipeline.py`

**Purpose:** Full experiment orchestration

**Expected behavior:**

* Runs all configured experiments
* Compares multiple filters and configurations
* Saves outputs (metrics, plots, summaries)

This file represents the **complete experimental pipeline for submission**.

---

### 🔹 `test_all.py`

**Purpose:** Testing and validation

**Expected behavior:**

* Verifies correctness of:

  * filters
  * simulation functions
  * utilities
* Checks numerical stability and input validation

Used to ensure that all components behave correctly before running experiments.

---

## 🔄 Workflow

The project follows this pipeline:

1. **Simulation**

   * `simulator_model_comps.py` generates data

2. **Filtering**

   * `replicate_Li_filters.py` applies filtering methods
   * `Hu_filters_utils.py` provides particle flow transformations

3. **Evaluation**

   * `metric_utils.py` computes performance metrics

4. **Execution**

   * `run_pipeline.py` or `submission_execution_pipeline_FINAL_ALL.py` orchestrates the process

---

## 🚀 How to Run

The pipeline is configured in **quick mode by default**, which runs a reduced set of experiments for faster execution.

### Run the pipeline:

```bash
python run_pipeline.py
```

This will:
- execute a lightweight version of the experiments
- generate a subset of results for validation and testing

If needed, the configuration inside `run_pipeline.py` or  
`execution_pipeline.py` can be modified to run the full set of experiments.

### Run tests:

```bash
python -m unittest test_all.py
```

---

## ⚠️ Notes

* All modules are interconnected; moving files may break imports
* The project uses TensorFlow for numerical computation
* Numerical stability is handled internally (e.g., covariance checks, kernel regularization)

---

## ✅ Summary

* Each file has a **clear, modular role**
* The system is designed as a **full filtering pipeline**
* Running the pipeline reproduces experiments and metrics
* Tests ensure correctness and robustness of all components

---

## 📚 Main references

- Li, T., Coates, M., & Rabbat, M. (2017)  
  *Particle Filtering with Invertible Particle Flow*  
  https://arxiv.org/abs/1703.08931  

- Hu, X., et al. (2016)  
  *A Particle-Flow Filter for High-Dimensional System Applications*  
  https://ieeexplore.ieee.org/document/7460877  

- Dai, H., et al. (2020)  
  *Stiffness Mitigation in Stochastic Particle Flow Filters*  
  https://arxiv.org/abs/2006.12320  
