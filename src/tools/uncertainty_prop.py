import numpy as np
import scipy


def mean_prop(K, Lambda, u, S, X_train, y_train):
    """
    Computes the mean of predictive distribution (21) using an exact formula.
    Assumes we are using Gaussian kernels.

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
    dict
        Dictionary containing beta and l (equation 31)
    """
    beta = scipy.linalg.solve(K, y_train)
    Lambda_inv = np.linalg.inv(Lambda)
    S_Lambda_inv = np.linalg.inv(S + Lambda)
    l = np.zeros(beta.shape[0])
    d = S.shape[0]

    for j in range(l.shape[0]):
        gauss_cov = (u - X_train[j, :]).T @ S_Lambda_inv @ (u - X_train[j, :])
        l[j] = (np.linalg.det(Lambda_inv @ S + np.identity(d)) ** (-1/2)) * np.exp(-1/2 * gauss_cov)

    return np.dot(beta, l), {'beta': beta, 'l': l}


def mean_prop_mc(K, Lambda, u, S, X_train, y_train):
    """
    Computes the mean of predictive distribution (21) using Monte Carlo.
    Assumes we are using Gaussian kernels.

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
    def gauss_kern(x1, x2):
        return np.exp(-1/2 * (x1-x2).T @ np.linalg.inv(Lambda) @ (x1-x2))

    T = 10000
    X_star = np.random.multivariate_normal(u, S, size=T)

    num_train = K.shape[0]
    K_inv = np.linalg.inv(K)
    m_est = 0
    for t in range(T):
        k_vec = np.array([gauss_kern(X_star[t, :], X_train[n, :]) for n in range(num_train)])
        mu = k_vec.T @ K_inv @ y_train
        m_est += mu

    return m_est / T


def variance_prop(K, Lambda, u, S, X_train, y_train):
    """
    Computes the variance of predictive distribution (21) using an exact formula.
    Assumes we are using Gaussian kernels.

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
       Variance of predictive distribution of f
    """

    mean, params = mean_prop(K, Lambda, u, S, X_train, y_train)
    beta = params['beta']
    l = params['l']

    num_train = X_train.shape[0]
    d = S.shape[0]
    L = np.zeros((num_train, num_train))
    half_Lam_S_inv = np.linalg.inv(Lambda / 2 + S)
    Lam_inv = np.linalg.inv(Lambda)
    det_part = np.linalg.det(2 * Lam_inv @ S + np.identity(d)) ** (-1/2)
    for i in range(num_train):
        for j in range(num_train):
            x_d = (X_train[i, :] + X_train[j, :]) / 2
            exp_part = np.exp(-1/2 * (u - x_d).T @ half_Lam_S_inv @ (u - x_d) +
                              -1/4 * (X_train[i, :] - X_train[j, :]).T @ Lam_inv @ (X_train[i, :] - X_train[j, :]))

            L[i, j] = det_part * exp_part

    K_inv = np.linalg.inv(K)
    return 1 - np.trace((K_inv - np.outer(beta, beta)) @ L) - mean**2


def variance_prop_mc(K, Lambda, u, S, X_train, y_train):
    """
    Computes the variance of predictive distribution (21) using Monte Carlo.
    Assumes we are using Gaussian kernels.

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
       Variance of predictive distribution of f
    """
    def gauss_kern(x1, x2):
        return np.exp(-1/2 * (x1-x2).T @ np.linalg.inv(Lambda) @ (x1-x2))

    T = 10000
    X_star = np.random.multivariate_normal(u, S, size=T)

    num_train = K.shape[0]
    K_inv = np.linalg.inv(K)
    mu_list = []
    sig_sq_list = []
    for t in range(T):
        k_vec = np.array([gauss_kern(X_star[t, :], X_train[n, :]) for n in range(num_train)])
        mu = k_vec.T @ K_inv @ y_train
        sig_sq = 1 - k_vec.T @ K_inv @ k_vec

        mu_list.append(mu)
        sig_sq_list.append(sig_sq)

    return np.mean(sig_sq_list) + np.var(mu_list)
