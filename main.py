import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

from src.bayesian_linear_regressor import BayesianLinearRegressor
from src.plots import plot_posterior_distributions, plot_posterior_predictive, plot_model_comparison, \
    plot_weight_comparison, plot_prior_sensitivity_weights, plot_sensitivity_metrics
from src.verification import compute_credible_bands, top_k_predictors, create_comparison_baselines, \
    prior_sensitivity_analysis
from src.hyperparams import maximize_evidence
from src.math_utils import rmse


if __name__ == '__main__':
    dataset = fetch_ucirepo(id=183)
    X = dataset.data.features
    y = dataset.data.targets

    # Drop columns with NaNs; keep only numeric features (float64 as there were some unreliable int64 features)
    X = X.dropna(axis=1)
    X = X.select_dtypes(include=[np.float64])

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

    # hyper param optimization via evidence maximization (random initial values)
    sigma2_init = 1.0
    sigma2_v_init = 1.0

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

    blr_train_rmse = rmse(y_train, X_train @ blr_opt.mean_post)
    blr_test_rmse = rmse(y_test, mean_pred_opt)
    print(f"\nTrain RMSE (BLR evidence-optimal): {blr_train_rmse:.4f}")
    print(f"\nTest RMSE (BLR evidence-optimal): {blr_test_rmse:.4f}")

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

    plot_posterior_distributions(top)

    plot_posterior_predictive(y_test, mean_pred_opt, std_pred_opt)

    # OLS / Ridge (CV) / Ridge (BLR-lambda) comparison
    results = create_comparison_baselines(
        X_train, y_train, X_test, y_test,
        blr_mean_post=blr_opt.mean_post,
        sigma2_opt=sigma2_opt,
        sigma2_v_opt=sigma2_v_opt,
    )

    weights = {
        "OLS": results["ols"]["coef"],
        "Ridge (CV)": results["ridge_cv"]["coef"],
        "Ridge (BLR-λ)": results["ridge_blr"]["coef"],
        "BLR (evidence)": blr_opt.mean_post,
    }

    print("\n--- Model comparison ---")
    print(f"{'Method':<22} {'Train RMSE':>12} {'Test RMSE':>12}")
    print("-" * 50)
    print(f"{'OLS':<22} {results['ols']['train_rmse']:>12.4f} "
          f"{results['ols']['test_rmse']:>12.4f}")
    print(f"{'Ridge (CV)':<22} {results['ridge_cv']['train_rmse']:>12.4f} "
          f"{results['ridge_cv']['test_rmse']:>12.4f}  alpha={results['ridge_cv']['alpha']:.5f}")
    print(f"{'Ridge (BLR-lambda)':<22} {results['ridge_blr']['train_rmse']:>12.4f} "
          f"{results['ridge_blr']['test_rmse']:>12.4f}  alpha={results['ridge_blr']['alpha']:.5f}")
    print(f"{'BLR (evidence)':<22}            -  {blr_test_rmse:>12.4f}  lambda={results['ridge_blr']['lambda_blr']:.5f}")

    plot_model_comparison(results, blr_train_rmse, blr_test_rmse)
    plot_weight_comparison(weights, phi_names, list(top["indices"]))

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
        track_indices=list(top["indices"]),
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

    plot_prior_sensitivity_weights(sens, top, sigma2_v_opt, sv_values=[0.001, sigma2_v_opt, 0.1, 10.0])
    plot_sensitivity_metrics(sens, sigma2_v_opt)