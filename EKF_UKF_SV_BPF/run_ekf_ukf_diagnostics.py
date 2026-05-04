from ekf_ukf_diagnostics import (
    run_heatmap_analysis,
    simulate_quadratic_model,
    plot_heatmaps_quadratic,
    print_summary_stats,
    plot_fraction_lollipops,
    plot_bad_acf,
    plot_bad_pacf,
)


def main():
    phi_grid = [0.3, 0.5, 0.9, 0.95]
    sigma_e_grid = [0.05, 0.50, 1.0, 2.0]

    df_heat, diagnostics_store = run_heatmap_analysis(
        simulator_fun=simulate_quadratic_model,
        phi_grid=phi_grid,
        sigma_e_grid=sigma_e_grid,
        sigma_eta=0.2,
        T=70,
        m0=0.5,
        P0=1.0,
        x_tol=0.05,
        crosscov_tol=1e-10,
        seed=42,
    )

    print_summary_stats(df_heat)

    plot_heatmaps_quadratic(df_heat)
    plot_fraction_lollipops(df_heat)

    selected = [(0.3, 0.05), (0.5, 0.05), (0.95, 0.05), (0.95, 0.50), (0.95, 2.0)]

    plot_bad_acf(diagnostics_store, selected_configs=selected, nlags=25)
    plot_bad_pacf(diagnostics_store, selected_configs=selected, nlags=25)

    return df_heat, diagnostics_store


if __name__ == "__main__":
    df_heat, diagnostics_store = main()