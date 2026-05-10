import numpy as np
from numpy import sqrt, ndarray, mean, abs, diag, max, logspace
from src.math_utils import rmse
from scipy.stats import norm # TODO verify with professor if its legal
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression

from src.bayesian_linear_regressor import BayesianLinearRegressor


def compute_credible_bands(
    mean_pred: ndarray,
    var_pred: ndarray,
    y_test: ndarray,
    confidence: float = 0.95,
) -> dict:
    """
    Computes credible intervals over the test set

    Args:
        mean_pred:  predictive mean per point, shape (n_test,)
        var_pred:   predictive variance per point, shape (n_test,)
        y_test:     test set targets, shape (n_test,)
        confidence: confidence level (alpha)

    Returns:
        dict with:
            'lower':    lower band, shape (n_test,)
            'upper':    upper band, shape (n_test,)
            'rmse':     test RMSE
            'coverage': fraction of the test set in [lower, upper]
            'z':        used z_{1-alpha/2} quantile
    """

    alpha = 1.0 - confidence
    z = norm.ppf(1 - alpha / 2)

    std_pred = sqrt(var_pred)
    lower = mean_pred - z * std_pred
    upper = mean_pred + z * std_pred

    rmse_result = rmse(y_test, mean_pred)
    coverage = float(mean((y_test >= lower) & (y_test <= upper)))

    return {
        "lower": lower,
        "upper": upper,
        "rmse": rmse_result,
        "coverage": coverage,
        "z": z,
    }

def top_k_predictors(
    mean_post: ndarray,
    cov_post: ndarray,
    feature_names: list[str],
    k: int = 5,
    confidence: float = 0.95,
    exclude_indices: tuple[int, ...] = (0,),
) -> dict:
    """
    Selects top k predictors according to |mu_n^(j)| and return its marginal posteriors

    Marginal of weight w_j is obtained using:
        w_j | D ~ N(mu_n^(j), sum_n^(j,j)) (check the lab section 8)

    The predictors are ranked by their signal-to-noise ratio (SNR):
    SNR_j = |mu_n^(j)| / sqrt(sum_n^(j,j))

    Args:
        mean_post:       posterior mean array, shape (D,)
        cov_post:        posterior covariance matrix, shape (D, D)
        feature_names:   feature namues (including intercept)
        k:               number of predictors
        confidence:      credible interval convidence level
        exclude_indices: indexes to exclude from the ranking (if any)

    Returns:
        dict with:
            'indices':       original indexes of the top predictors
            'names':         top predictors names
            'means':         me_n^(j) of each predictor, shape (k,)
            'stds':          sqrt(sum_n^(j,j)) of each predictor, shape (k,)
            'ci_lower':      credible interval lower band, shape (k,)
            'ci_upper':      credible interval upper band, shape (k,)
            'excludes_zero': bool array, True if the CI does not contain 0
    """

    dimension = len(mean_post)
    if not (mean_post.ndim == 1 and cov_post.shape == (dimension, dimension) and len(feature_names) == dimension):
        raise ValueError("invalid args")

    std_filtered = np.sqrt(np.diag(cov_post))
    snr = np.abs(mean_post) / (std_filtered + 1e-12)  # epsilon for safety

    snr_filtered = snr.copy()
    snr_filtered[list(exclude_indices)] = -np.inf

    top_idx = np.argsort(-snr_filtered)[:k]

    means = mean_post[top_idx]
    variances = diag(cov_post)[top_idx]
    stds  = sqrt(variances)

    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)
    ci_lower, ci_upper = means - z*stds, means + z*stds

    # TODO check
    excludes_zero = (ci_lower > 0) | (ci_upper < 0)

    names = [feature_names[i] for i in top_idx]

    return {
        "indices": top_idx,
        "names": names,
        "means": means,
        "stds": stds,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "excludes_zero": excludes_zero,
    }


def _ols(
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
) -> dict:
    # fit_intercept=False because we already added the column of 1
    ols = LinearRegression(fit_intercept=False)
    ols.fit(x_train, y_train)
    ols_test_pred = ols.predict(x_test)
    ols_train_pred = ols.predict(x_train)
    ols_rmse_test = rmse(y_test, ols_test_pred)
    ols_rmse_train = rmse(y_train, ols_train_pred)
    return {
        "train_rmse": ols_rmse_train,
        "test_rmse": ols_rmse_test,
        "coef": ols.coef_,
    }

def _ridge_cv(
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
    alphas: ndarray = None,
    cv_folds: int = 10,
) -> dict:
    ridge_cv = RidgeCV(alphas=alphas, fit_intercept=False, cv=cv_folds)
    ridge_cv.fit(x_train, y_train)
    alpha_cv = ridge_cv.alpha_
    ridge_test_pred = ridge_cv.predict(x_test)
    ridge_train_pred = ridge_cv.predict(x_train)
    ridge_rmse_test = rmse(y_test, ridge_test_pred)
    ridge_rmse_train = rmse(y_train, ridge_train_pred)
    return {
        "train_rmse": ridge_rmse_train,
        "test_rmse": ridge_rmse_test,
        "coef": ridge_cv.coef_,
        "alpha": alpha_cv,
    }

def _ridge_blr(
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
    blr_mean_post: ndarray,
    lambda_blr: float
) -> dict:
    ridge_blr = Ridge(alpha=lambda_blr, fit_intercept=False)
    ridge_blr.fit(x_train, y_train)
    ridge_blr_test_pred = ridge_blr.predict(x_test)
    ridge_blr_train_pred = ridge_blr.predict(x_train)
    ridge_blr_rmse_test = rmse(y_test, ridge_blr_test_pred)
    ridge_blr_rmse_train = rmse(y_train, ridge_blr_train_pred)

    max_dev_coef = float(max(abs(blr_mean_post - ridge_blr.coef_)))
    return {
        "train_rmse": ridge_blr_rmse_train,
        "test_rmse":  ridge_blr_rmse_test,
        "coef":       ridge_blr.coef_,
        "alpha":      ridge_blr.alpha,
        "lambda_blr": lambda_blr,
        "max_dev_blr_ridge_coef": max_dev_coef,
    }

def create_comparison_baselines(
    x_train: ndarray,
    y_train: ndarray,
    x_test: ndarray,
    y_test: ndarray,
    blr_mean_post: ndarray,
    sigma2_opt: float,
    sigma2_v_opt: float,
    alphas: ndarray = None,
    cv_folds: int = 10,
) -> dict:
    """
    Returns OLS and Ridge baselines to compare against BLR

    Replicates section 9 of the lab

    Args:
        x_train, y_train: train set, x already has the intercept (n_train, D)
        x_test, y_test:   test set, x already has the intercept (n_test, D)
        blr_mean_post:    BLR posterior optimal mean (D,)
        sigma2_opt:       BLR optimal sigma2
        sigma2_v_opt:     BLR optimal sigma2_v
        alphas:           alpha grid for RidgeCV (default: logspace -4..2)
        cv_folds:         folds for CV (default 10, check the lab)

    Returns:
        dict with:
            'ols':        {'train_rmse', 'test_rmse', 'coef'}
            'ridge_cv':   {'train_rmse', 'test_rmse', 'coef', 'alpha'}
            'ridge_blr':  {'train_rmse', 'test_rmse', 'coef', 'alpha', 'lambda_blr', 'max_dev_blr_ridge_coef'}
    """
    if alphas is None:
        alphas = logspace(-4, 2, 50)

    ols_results = _ols(x_train, y_train, x_test, y_test)

    # check lab section 9
    ridge_cv_results = _ridge_cv(x_train, y_train, x_test, y_test, alphas, cv_folds)

    # lamba = sigma2 / sigma2_v — slide 16. TODO document why is different
    lambda_blr = sigma2_opt / sigma2_v_opt
    ridge_blr_results = _ridge_blr(x_train, y_train, x_test, y_test, blr_mean_post, lambda_blr)

    return {
        "ols": ols_results,
        "ridge_cv": ridge_cv_results,
        "ridge_blr": ridge_blr_results,
    }

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
    track_indices: list[int] | None = None,
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
    tracked_means_arr = np.zeros((n_grid, len(track_indices))) if track_indices else None

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

        if track_indices is not None:
            tracked_means_arr[i, :] = blr.mean_post[track_indices]

    return {
        "sigma2_v_grid": sigma2_v_grid,
        "test_rmse":     test_rmse_arr,
        "norm_mu_n":     norm_mu_n_arr,
        "mean_pred_sd":  mean_pred_sd_arr,
        "coverage":      coverage_arr,
        "top_k_names":   top_k_names_list,
        "top_k_means":   top_k_means_arr,
        "tracked_means": tracked_means_arr,
        "track_indices": track_indices,
    }
