from numpy import sqrt, ndarray, eye
from scipy.linalg import cho_factor, cho_solve # TODO verify with professor if its legal

def rmse(y_true, y_pred):
    return float(sqrt(((y_true - y_pred) ** 2).mean()))

def cholesky_inv(matrix: ndarray) -> tuple[ndarray, ndarray]: # recommendation taken from the slide 17
    cho_result = cho_factor(matrix, lower=True)
    c_inv = cho_solve(cho_result, eye(matrix.shape[0]))
    return c_inv, cho_result[0]