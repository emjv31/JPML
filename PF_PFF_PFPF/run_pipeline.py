"""Small launcher for execution_pipeline.py.

Run quick smoke test:
    python run_pipeline.py

Run larger submission settings:
    python run_pipeline.py --full
"""

import argparse
import execution_pipeline as pipe


def configure_quick():
    pipe.QUICK_RUN = True
    pipe.SAVE_PICKLES = False
    pipe.RUN_OT_TUNING = False
    pipe.RUN_GRADIENT_COMPARISON = False
    pipe.INCLUDE_HEAVY_BASELINES = False

    pipe.EXAMPLE_B_NP = 10
    pipe.EXAMPLE_B_N_MC = 1
    pipe.EXAMPLE_C_DIMS = [20]
    pipe.EXAMPLE_C_NP = 10
    pipe.EXAMPLE_C_N_MC = 1

    pipe.DIAGNOSTIC_D = 10
    pipe.DIAGNOSTIC_NP = 10
    pipe.DIAGNOSTIC_N_MC_ACCURACY = 1
    pipe.DIAGNOSTIC_N_MC_DIAGNOSTICS = 1

    pipe.HU_DIMS = [10]
    pipe.HU_NP = 10
    pipe.HU_N_MC = 1
    pipe.HU_N_STEPS = 2

    pipe.BETA_COMPARISON_DIMS = [10]
    pipe.BETA_COMPARISON_M1_VALUES = [0.05, 1.0]


def configure_full():
    pipe.QUICK_RUN = False
    pipe.SAVE_PICKLES = False
    pipe.RUN_OT_TUNING = True
    pipe.RUN_GRADIENT_COMPARISON = True
    pipe.INCLUDE_HEAVY_BASELINES = False

    pipe.EXAMPLE_B_NP = 200
    pipe.EXAMPLE_B_N_MC = 100
    pipe.EXAMPLE_C_DIMS = [144, 400]
    pipe.EXAMPLE_C_NP = 200
    pipe.EXAMPLE_C_N_MC = 100

    pipe.DIAGNOSTIC_D = 10
    pipe.DIAGNOSTIC_NP = 200
    pipe.DIAGNOSTIC_N_MC_ACCURACY = 100
    pipe.DIAGNOSTIC_N_MC_DIAGNOSTICS = 100

    pipe.HU_DIMS = [10, 144, 400]
    pipe.HU_NP = 200
    pipe.HU_N_MC = 100
    pipe.HU_N_STEPS = 10

    pipe.BETA_COMPARISON_DIMS = [10, 144, 400]
    pipe.BETA_COMPARISON_M1_VALUES = [0.05, 1.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run full submission settings instead of quick smoke test")
    args = parser.parse_args()

    if args.full:
        configure_full()
    else:
        configure_quick()

    pipe.main()


if __name__ == "__main__":
    main()
