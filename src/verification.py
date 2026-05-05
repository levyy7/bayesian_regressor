from numpy import sqrt, ndarray, mean, abs, argsort, diag, inf, max, logspace
from math_utils import rmse
from scipy.stats import norm # TODO verify with professor if its legal
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression

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

    abs_means = abs(mean_post)

    # remove exclusions
    abs_means_filtered = abs_means.copy()
    abs_means_filtered[list(exclude_indices)] = -inf

    top_idx = argsort(-abs_means_filtered)[:k]

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