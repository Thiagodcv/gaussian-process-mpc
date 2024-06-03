import numpy as np
import scipy


def mean_prop(K, Lambda, u, S, X_train, y_train):
    """
    Parameters:
    ----------
    K: np.array
        Evidence covariance matrix
    Lambda: np.array
        Diagonal matrix containing kernel parameters
    u: np.array
        Mean of input distribution (which is assumed to be Gaussian)
    S: np.array
        Covariance of input distribution (which is assumed to be Gaussian)
    X_train: np.array
        GP training data inputs
    y_train: scalar
        GP training data outputs

    Return :
    ------
    scalar
        Mean of predictive distribution of f
    """
    beta = scipy.linalg.solve(K, y_train)
    Lambda_inv = np.linalg.inv(Lambda)
    S_Lambda_inv = np.linalg.inv(S + Lambda)
    l = np.zeros(beta.shape[0])
    d = S.shape[0]

    for j in range(l.shape[0]):
        gauss_cov = (u - X_train[j, :]).T @ S_Lambda_inv @ (u - X_train[j, :])
        l[j] = (np.linalg.det(Lambda_inv @ S + np.identity(d)) ** (-1/2)) * np.exp(-1/2 * gauss_cov)

    return np.dot(beta, l)
