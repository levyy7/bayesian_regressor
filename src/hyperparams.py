from numpy import ndarray, asarray, float64, eye, sum, log, diag, pi, exp
from pandas import DataFrame

from math_utils import cholesky_inv
from scipy.optimize import OptimizeResult, minimize # TODO verify with professor if its legal

from src.plots import plot_evidence_maximization


def log_marginal_likelihood(x: ndarray, y: ndarray, noise_variance: float, prior_variance: float) -> float:
    """
    Log marginal likelihood using isotropic gaussian prior
    p(v) = N(0, σ^2_v * I) (slide 15)

    Uses lab's formula (Lab1-BayesianLR.html, 1.3):
        log p(y | phi, sigma^2, sigma^2_v) = -1/2 [ n·log(2pi) + log|C| + y^TC⁻^(-1)y ],
        C = sigma^2*I_n + sigma^2_v * phi phi^T

    Args:
        x:        design matrix(n, D), same role as phi.
        y:        targets, shape (n).
        noise_variance:  sigma^2 (>0).
        prior_variance: sigma^2_v (>0).

    Returns:
        log-evidence (scalar)
    """
    x = asarray(x, dtype=float64)
    y = asarray(y, dtype=float64)
    n = x.shape[0]

    # C formula
    c = noise_variance * eye(len(x)) + prior_variance * (x @ x.T)

    (c_inv, lower) = cholesky_inv(c)
    # log|C| = 2 · sum log(L_ii) TODO add reference of why we calculate the determinant like this
    log_det_c = 2.0 * sum(log(diag(lower)))

    # lab formula (1.3)
    return -0.5 * (n * log(2.0 * pi) + log_det_c + (y.T @ c_inv @ y))

def maximize_evidence(
    x: ndarray,
    y: ndarray,
    sigma2_init: float = 1.0,
    sigma2_v_init: float = 1.0,
) -> tuple[float, float, OptimizeResult]:
    """
    Tunes hyperparams (sigma^2, sigma^2_v) por type-II maximum likelihood,
    by maximizing log evidence over the train set

    TODO reference labs bibliography (evidence framework de Bishop (2006, §3.5.2) and MacKay (1992))

    Args:
        x:             design matrix phi of the train set (n, D).
        y:             train set targets (n,).
        sigma2_init:   initial value of sigma^2. Reasonable: OLS residuals
        sigma2_v_init: initial value sigma^2_v

    Returns:
        sigma2_opt:    optimal sigma^2
        sigma2_v_opt:  optimal sigma^2_v
        result:        SciPy OptimizeResult
    """

    # objective function
    def neg_log_evidence(u: ndarray) -> float:
        [u1, u2] = u
        sigma2 = exp(u1)
        sigma2_v = exp(u2)
        return -log_marginal_likelihood(x, y, sigma2, sigma2_v)

    # auxiliary function to store the optimization process
    history = []
    def neg_log_evidence_tracked(u: ndarray) -> float:
        val = neg_log_evidence(u)
        history.append({
            "sigma2": exp(u[0]),
            "sigma2_v": exp(u[1]),
            "neg_log_ev": val
        })
        return val

    # we use log scale and then convert back
    # TODO document why (to avoid the positivity constraint (sigma2 > 0, sigma2_v > 0). The optimizer operates on u = (log sigma2, log sigma2_v), and we exponentiate at the end.)
    u_init = [log(sigma2_init), log(sigma2_v_init)]

    # TODO document why we use this method
    result = minimize(neg_log_evidence_tracked, u_init, method="L-BFGS-B")
    history = DataFrame(history)

    plot_evidence_maximization(history)

    # convert back
    sigma2_opt = exp(result.x[0])
    sigma2_v_opt = exp(result.x[1])

    return sigma2_opt, sigma2_v_opt, result