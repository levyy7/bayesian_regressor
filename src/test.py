import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

from src.bayesian_linear_regressor import BayesianLinearRegressor
from verification import compute_credible_bands, top_k_predictors, create_comparison_baselines
from hyperparams import log_marginal_likelihood, maximize_evidence
from math_utils import rmse


def prior_sensitivity_analysis(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    sigma2: float,
    sigma2_v_grid: np.ndarray,
    confidence: float = 0.95,
    k_top: int = 5,
) -> dict:
    """
    Prior sensitivity analysis. For each sigma^2_v in the grid (with sigma^2
    fixed), fit BLR and measure key quantities to evaluate which conclusions
    are robust to the choice of prior and which are not.

    Replicates section 7 of the lab:
    the lab computes test RMSE, ||mu_n||_2 and mean predictive SD for a
    logarithmic grid of sigma^2_v. We also add coverage and top-k
    predictors.

    Args:
        x_train, y_train: train set (with intercept).
        x_test, y_test:   test set (with intercept).
        feature_names:    names of the D columns of Phi (includes intercept).
        sigma2:           fixed sigma^2 (typically the evidence-optimal value).
        sigma2_v_grid:    array of sigma^2_v values to evaluate.
        confidence:       credible interval level for coverage.
        k_top:            number of top predictors to keep per value.

    Returns:
        dict with arrays parallel to the grid:
            'sigma2_v_grid': input grid.
            'test_rmse':     test RMSE per value.
            'norm_mu_n':     ||mu_n||_2 per value.
            'mean_pred_sd':  mean of predictive std on test set.
            'coverage':      empirical coverage at `confidence`% on test.
            'top_k_names':   list[list[str]], top-k names per value.
            'top_k_means':   array (n_grid, k_top), top-k magnitudes (signed).
    """
    n_grid = len(sigma2_v_grid)
    test_rmse_arr = np.zeros(n_grid)
    norm_mu_n_arr = np.zeros(n_grid)
    mean_pred_sd_arr = np.zeros(n_grid)
    coverage_arr = np.zeros(n_grid)
    top_k_names_list = []
    top_k_means_arr = np.zeros((n_grid, k_top))

    for i, sigma2_v in enumerate(sigma2_v_grid):
        blr = BayesianLinearRegressor(sigma2=sigma2, sigma2_v=sigma2_v)
        blr.fit(x_train, y_train)

        mean_pred, var_pred = blr.predict(x_test)
        std_pred = np.sqrt(var_pred)

        test_rmse_arr[i] = rmse(y_test, mean_pred)
        # Euclidean (L2) norm of the posterior mean
        norm_mu_n_arr[i] = float(np.linalg.norm(blr.mean_post, ord=2))
        mean_pred_sd_arr[i] = float(std_pred.mean())

        bands = compute_credible_bands(mean_pred, var_pred, y_test, confidence)
        coverage_arr[i] = bands["coverage"]

        top = top_k_predictors(blr.mean_post, blr.cov_post, feature_names, k=k_top)
        top_k_names_list.append(top["names"])
        top_k_means_arr[i, :] = top["means"]  # signed magnitudes

    return {
        "sigma2_v_grid": sigma2_v_grid,
        "test_rmse":     test_rmse_arr,
        "norm_mu_n":     norm_mu_n_arr,
        "mean_pred_sd":  mean_pred_sd_arr,
        "coverage":      coverage_arr,
        "top_k_names":   top_k_names_list,
        "top_k_means":   top_k_means_arr,
    }


if __name__ == '__main__':
    dataset = fetch_ucirepo(id=183)
    X = dataset.data.features
    y = dataset.data.targets

    # Drop columns with NaNs; keep only numeric features.
    X = X.dropna(axis=1)
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    phi_names = ["intercept"] + feature_names

    X = X.to_numpy()
    y = y.to_numpy().squeeze()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # standardisation: statistics computed only on train
    mu_X, sigma_X = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mu_X) / sigma_X
    X_test  = (X_test  - mu_X) / sigma_X

    mu_y, sigma_y = y_train.mean(), y_train.std()
    y_train = (y_train - mu_y) / sigma_y
    y_test  = (y_test  - mu_y) / sigma_y

    # add intercept (phi_1(x) := 1, slide 14)
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_test  = np.hstack([np.ones((X_test.shape[0],  1)), X_test])

    print(f"Train size : {X_train.shape}")
    print(f"Test size  : {X_test.shape}")
    print(f"Features   : {X_train.shape[1]}  (includes intercept)")

    # log-evidence vs sigma^2_v (1D sweep) (sanity check)
    # sigma^2 fixed at the OLS residual variance (lab section 4)
    v_ols, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    residuals_ols = y_train - X_train @ v_ols
    sigma2 = float(residuals_ols.var())
    print(f"\nsigma^2 (OLS residual variance): {sigma2:.4f}")

    sigma2_v_grid = np.logspace(-5, 3, 80)
    lml = np.array([
        log_marginal_likelihood(X_train, y_train, sigma2, sv) for sv in sigma2_v_grid
    ])

    idx_opt = int(np.argmax(lml))
    sv_opt = sigma2_v_grid[idx_opt]
    print(f"Optimal sigma^2_v on grid: {sv_opt:.6f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sigma2_v_grid, lml, "-o", markersize=3, color="darkgreen")
    ax.axvline(sv_opt, color="firebrick", linestyle="--",
               label=fr"$\sigma_v^2$ optimum approx {sv_opt:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_v^2$")
    ax.set_ylabel(r"$\log p(y \mid \Phi, \sigma^2, \sigma_v^2)$")
    ax.set_title(r"Log-evidence vs prior variance ($\sigma^2$ fixed)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # hyper param tunning
    sigma2_init = sigma2
    sigma2_v_init = sv_opt

    sigma2_opt, sigma2_v_opt, result = maximize_evidence(
        X_train, y_train, sigma2_init, sigma2_v_init
    )

    print("\n--- Evidence maximization ---")
    print(f"Optimal sigma^2:    {sigma2_opt:.6f}  (init: {sigma2_init:.6f})")
    print(f"Optimal sigma^2_v:  {sigma2_v_opt:.6f}  (init: {sigma2_v_init:.6f})")
    print(f"Convergence: {result.success}  ({result.message})")
    print(f"Iterations:  {result.nit}, evaluations: {result.nfev}")

    # refit with optimal parameters
    blr_opt = BayesianLinearRegressor(sigma2=sigma2_opt, sigma2_v=sigma2_v_opt)
    blr_opt.fit(X_train, y_train)

    mean_pred_opt, var_pred_opt = blr_opt.predict(X_test)
    std_pred_opt = np.sqrt(var_pred_opt)

    rmse_opt = rmse(y_test, mean_pred_opt)
    print(f"\nTest RMSE (BLR evidence-optimal): {rmse_opt:.4f}")

    # credible bands
    bands = compute_credible_bands(mean_pred_opt, var_pred_opt, y_test)

    print(f"\n--- Credible bands on test ---")
    print(f"Test RMSE:                {bands['rmse']:.4f}")
    print(f"Empirical coverage @ 95%: {bands['coverage']:.4f}")
    print(f"(z = {bands['z']:.6f})")

    # top 5 predictors
    top = top_k_predictors(
        mean_post=blr_opt.mean_post,
        cov_post=blr_opt.cov_post,
        feature_names=phi_names,
        k=5,
    )

    print("\n--- Top-5 most influential predictors ---")
    print(f"{'Predictor':<20} {'mu_n':>10} {'sigma_n':>10} {'95% CI':>22} {'Excl. 0':>8}")
    print("-" * 75)
    for i in range(5):
        ci_str = f"[{top['ci_lower'][i]:+.4f}, {top['ci_upper'][i]:+.4f}]"
        print(f"{top['names'][i]:<20} "
              f"{top['means'][i]:>+10.4f} "
              f"{top['stds'][i]:>10.4f} "
              f"{ci_str:>22} "
              f"{'Yes' if top['excludes_zero'][i] else 'No':>8}")

    # OLS / Ridge (CV) / Ridge (BLR-lambda) comparison
    results = create_comparison_baselines(
        X_train, y_train, X_test, y_test,
        blr_mean_post=blr_opt.mean_post,
        sigma2_opt=sigma2_opt,
        sigma2_v_opt=sigma2_v_opt,
    )

    print("\n--- Model comparison ---")
    print(f"{'Method':<22} {'Train RMSE':>12} {'Test RMSE':>12}")
    print("-" * 50)
    print(f"{'OLS':<22} {results['ols']['train_rmse']:>12.4f} "
          f"{results['ols']['test_rmse']:>12.4f}")
    print(f"{'Ridge (CV)':<22} {results['ridge_cv']['train_rmse']:>12.4f} "
          f"{results['ridge_cv']['test_rmse']:>12.4f}  alpha={results['ridge_cv']['alpha']:.5f}")
    print(f"{'Ridge (BLR-lambda)':<22} {results['ridge_blr']['train_rmse']:>12.4f} "
          f"{results['ridge_blr']['test_rmse']:>12.4f}  alpha={results['ridge_blr']['alpha']:.5f}")
    print(f"{'BLR (evidence)':<22}            -  {rmse_opt:>12.4f}  lambda={results['ridge_blr']['lambda_blr']:.5f}")

    # Sanity check of the BLR-MAP == Ridge equivalence (slide 16).
    print(f"\nBLR-MAP == Ridge sanity check:")
    print(f"  max|mu_n - v_ridge_blr| = {results['ridge_blr']['max_dev_blr_ridge_coef']:.16f}")

    # prior sensitivity analisys
    # Sweep sigma^2_v over several orders of magnitude, sigma^2 fixed
    # at the optimum. Replicates section 7 of the lab
    sigma2_v_grid_sens = np.logspace(-3, 4, 29)

    sens = prior_sensitivity_analysis(
        X_train, y_train, X_test, y_test,
        feature_names=phi_names,
        sigma2=sigma2_opt,
        sigma2_v_grid=sigma2_v_grid_sens,
    )

    representative_idx = [0, 7, 14, 21, 28]
    idx_opt_sens = int(np.argmin(np.abs(sens['sigma2_v_grid'] - sigma2_v_opt)))
    representative_idx = sorted(set(representative_idx + [idx_opt_sens]))

    print("\n--- Prior sensitivity ---")
    print(f"{'sigma^2_v':>10}  {'Test RMSE':>10}  {'||mu_n||':>10}  {'Mean SD':>10}  {'Coverage':>10}")
    print("-" * 60)
    for i in representative_idx:
        print(f"{sens['sigma2_v_grid'][i]:>10.4g}  "
              f"{sens['test_rmse'][i]:>10.4f}  "
              f"{sens['norm_mu_n'][i]:>10.4f}  "
              f"{sens['mean_pred_sd'][i]:>10.4f}  "
              f"{sens['coverage'][i]:>10.4f}")

    print("\n--- Top-5 predictors at representative sigma^2_v values ---")
    for i in representative_idx:
        print(f"\nsigma^2_v = {sens['sigma2_v_grid'][i]:.4g}:")
        for j, name in enumerate(sens['top_k_names'][i]):
            print(f"  {j + 1}. {name:<20}  mu = {sens['top_k_means'][i, j]:+.4f}")