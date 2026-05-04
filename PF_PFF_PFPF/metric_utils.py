import time

import tensorflow as tf
import pandas as pd
import numpy as np


def make_beta_schedule(N_lambda=29, q=1.2, dtype=tf.float32):
    q = tf.cast(q, dtype)
    N_lambda = int(N_lambda)

    # First step so that sum_j eps_j = 1
    eps1 = (q - 1.0) / (tf.pow(q, N_lambda) - 1.0)

    # Geometric steps: eps_j = eps1 * q^(j-1)
    j = tf.cast(tf.range(N_lambda), dtype)
    steps = eps1 * tf.pow(q, j)

    # beta grid: [0, eps1, eps1+eps2, ..., 1]
    beta = tf.concat([tf.zeros([1], dtype=dtype), tf.cumsum(steps)], axis=0)

    beta = tf.tensor_scatter_nd_update(
        beta,
        indices=[[N_lambda]],
        updates=[tf.cast(1.0, dtype)]
    )

    return beta, steps


def monte_carlo_light_lost(filter_fn, N_MC, X_true, *args, **kwargs):
    """
    Monte Carlo memory-saving:
    - Computes only online MSE and ESS
    - Computes lost-track count (average error > sqrt(d))
    - Updates MSE only for non-lost tracks
    """

    #### INPUT VALIDATION
    if not callable(filter_fn):
        raise TypeError("filter_fn must be callable")

    if not isinstance(N_MC, int) or N_MC <= 0:
        raise ValueError("N_MC must be a positive integer")

    if not isinstance(X_true, (tf.Tensor, np.ndarray)):
        raise TypeError("X_true must be a Tensor or NumPy array")

    X_true = tf.convert_to_tensor(X_true)

    if X_true.shape.rank is not None and X_true.shape.rank != 2:
        raise ValueError("X_true must be rank 2 (T, d)")
    tf.debugging.assert_rank(X_true, 2)

    X_true = tf.cast(X_true, tf.float32)
    T = tf.shape(X_true)[0]
    d = tf.shape(X_true)[1]

    mse_mean = tf.zeros(T, dtype=tf.float32)
    ess_mean = None
    lost_count = 0
    valid_count = 0  # count replicates contributing to MSE

    start_total = time.perf_counter()

    for i in range(N_MC):
        result = filter_fn(*args, **kwargs)

        if isinstance(result, (tuple, list)) and len(result) == 2:
            ests, ESSs = result
            if ESSs is not None:
                ESSs = tf.cast(ESSs, tf.float32)
        else:
            ests = result
            ESSs = None

        ests = tf.cast(ests, tf.float32)
        if ests.shape != X_true.shape:
            raise ValueError("ests must have same shape as X_true")

        # Compute per-replicate average error
        std_error = tf.sqrt(tf.reduce_mean(tf.square(ests - X_true)))  

        if std_error > tf.sqrt(tf.cast(d, tf.float32)):
            lost_count += 1
            continue  # skip this replicate in MSE
        else:
            valid_count += 1

        # Update MSE
        diff = ests - X_true
        mse_curr = tf.reduce_mean(tf.square(diff), axis=1)
        mse_mean += (mse_curr - mse_mean) / valid_count

        # Update ESS
        if ESSs is not None:
            if ess_mean is None:
                ess_mean = ESSs
            else:
                ess_mean += (ESSs - ess_mean) / valid_count

    end_total = time.perf_counter()
    print(f"Monte Carlo done: {N_MC} runs, total time: {end_total - start_total:.3f}s")
    print(f"Lost tracks: {lost_count}/{N_MC}")

    return mse_mean, ess_mean, lost_count




def run_monte_carlo_sim(filters_config, true_data, N_MC=100, monte_carlo_fn=None):
    """
    Runs Monte Carlo simulations for a given set of filters.

    Args:
        filters_config (dict): Dictionary of filter functions, e.g. filters_config_C_tf
        data_true (list or array): List/array of data to run filters on. Uses first element by default.
        N_MC (int): Number of Monte Carlo runs per filter.
        monte_carlo_fn (callable): The Monte Carlo function to run (monte_carlo_light, monte_carlo_hybrid_block, etc.)

    Returns:
        dict: Dictionary with results for each filter.
    """
    
    # ------------------------
    # INPUT VALIDATION 
    # ------------------------
    if not isinstance(filters_config, dict):
        raise TypeError("filters_config must be a dictionary")

    if len(filters_config) == 0:
        raise ValueError("filters_config cannot be empty")

    if not callable(monte_carlo_fn):
        raise TypeError("monte_carlo_fn must be callable")

    if not isinstance(N_MC, int) or N_MC <= 0:
        raise ValueError("N_MC must be a positive integer")
    # ------------------------
    
    results = {}

    for name, filter_fn in filters_config.items():
        print(f"Starting filter: {name}")
        start_total = time.perf_counter()

        result = monte_carlo_fn(filter_fn, N_MC, true_data)

        end_total = time.perf_counter()
        runtime_total = end_total - start_total
        runtime_mean = runtime_total / N_MC

        print(f"Finished filter: {name}")
        print(f"Total time: {runtime_total:.3f} s, Average per run: {runtime_mean:.3f} s\n")

        
        results[name] = {
            "mse_t": result[0],      
            "ess_mean": result[1],   
            "lost_tracks":result[2], 
            "run_time": runtime_mean
        }

    return results


def sims_to_df(results_dict):
    rows = {}

    for name, out in results_dict.items():
        data = {}

        if out.get("mse_t") is not None:
            data["mean_MSE"] = np.mean(out["mse_t"])
        else:
            data["mean_MSE"] = np.nan

        if out.get("ess_mean") is not None:
            data["mean_ESS"] = np.mean(out["ess_mean"])
        else:
            data["mean_ESS"] = np.nan

        if out.get("run_time") is not None:
            data["Run_Time"] = np.mean(out["run_time"])
        else:
            data["Run_Time"] = np.nan

        if out.get("lost_tracks") is not None:
            data["Lost_tracks"] = np.mean(out["lost_tracks"])
        else:
            data["Lost_tracks"] = np.nan

        rows[name] = data

    return pd.DataFrame.from_dict(rows, orient="index")


def monte_carlo_final(
    filter_fn,
    Y,
    N_MC=100,
    output_names=None,   
    **filter_kwargs
):

    # Check if filter_fn is callable
    if not callable(filter_fn):
        raise TypeError("filter_fn must be callable")

    # Check if N_MC is a positive integer
    if not isinstance(N_MC, int) or N_MC <= 0:
        raise ValueError("N_MC must be a positive integer")

    # Ensure Y is a tensor-like object and has at least one dimension
    if not hasattr(Y, 'shape') or len(Y.shape) == 0:
        raise ValueError("Y must have at least one dimension")

    # Check if output_names is either None, a list, or a tuple
    if output_names is not None:
        if not isinstance(output_names, (list, tuple)):
            raise TypeError("output_names must be a list or tuple")
        if not all(isinstance(name, str) for name in output_names):
            raise TypeError("output_names must contain only strings")

    # --------------------------------------------------
    # RUN MONTE CARLO SIMULATIONS
    # --------------------------------------------------
    outputs_all = []

    for _ in range(N_MC):
        out = filter_fn(Y, **filter_kwargs)

        if not isinstance(out, (tuple, list)):
            out = (out,)

        outputs_all.append(out)

    n_out = len(outputs_all[0])

    # -------------------
    # DEFAULT NAMES 
    # -------------------
    if output_names is None:
        output_names = [f"out_{i}" for i in range(n_out)]

    # Ensure the length of output_names matches the number of outputs
    if len(output_names) != n_out:
        raise ValueError("Mismatch in output_names length")

    # ----------------
    # STACK RESULTS
    # ----------------
    stacked = {}

    for i, name in enumerate(output_names):
        elems = [out[i] for out in outputs_all]

        # Check for None values and stack outputs
        if elems[0] is None:
            stacked[name] = None

        elif isinstance(elems[0], dict):
            stacked_dict = {}
            for k in elems[0].keys():
                stacked_tensor = tf.stack([e[k] for e in elems], axis=0)
                # Squeeze the last dimension if it's unnecessary
                if len(stacked_tensor.shape) > 2 and stacked_tensor.shape[-1] == 1:
                    stacked_tensor = tf.squeeze(stacked_tensor, axis=-1)
                stacked_dict[k] = stacked_tensor
            stacked[name] = stacked_dict

        else:
            stacked[name] = tf.stack(elems, axis=0)
            
    return stacked