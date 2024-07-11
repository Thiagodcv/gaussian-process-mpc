import numpy as np
import scipy
import numba
import warnings


def mean_prop(K, Lambda, u, S, X_train, y_train):
    """
    TODO: Use Numba to speed up computation
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beta = scipy.linalg.solve(K, y_train, assume_a='pos')
        assert np.linalg.norm(K @ beta - y_train) < 1e-5
    # beta = np.linalg.inv(K) @ y_train
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


def covariance_prop(K1, K2, Lambda1, Lambda2, u, S, X_train, y_train):
    """
    Computes the covariance of GP outputs (A14) using an exact formula.
    Assumes we are using Gaussian kernels for both GP models 1 and 2.

    Parameters:
    ----------
    K1 & K2: np.array
       Evidence covariance matrix for GP models 1 and 2
    Lambda1 & Lambda2: np.array
       Diagonal matrix containing kernel parameters for GP models 1 and 2
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
       Covariance of joint predictive distribution of f1 and f2
    """
    mean1, params1 = mean_prop(K1, Lambda1, u, S, X_train, y_train)
    beta1 = params1['beta']

    mean2, params2 = mean_prop(K2, Lambda2, u, S, X_train, y_train)
    beta2 = params2['beta']

    Lambda1_inv = np.linalg.inv(Lambda1)
    Lambda2_inv = np.linalg.inv(Lambda2)

    def gauss_kern(x1, x2, Lambda_inv):
        return np.exp(-1/2 * (x1-x2).T @ Lambda_inv @ (x1-x2))

    num_train = beta1.shape[0]
    d = Lambda1.shape[0]
    Q_tilde = np.zeros((num_train, num_train))
    det_part = np.linalg.det(S @ (Lambda1_inv + Lambda2_inv) + np.identity(d)) ** (-1/2)
    for i in range(num_train):
        for j in range(num_train):
            k1 = gauss_kern(X_train[i, :], u, Lambda1_inv)
            k2 = gauss_kern(X_train[j, :], u, Lambda2_inv)
            z = Lambda1_inv @ (X_train[i, :] - u) + Lambda2_inv @ (X_train[j, :] - u)
            exp_part = np.exp(1/2 * z.T @ np.linalg.inv(S @ (Lambda1_inv + Lambda2_inv) + np.identity(d)) @ S @ z)
            Q_tilde[i, j] = k1 * k2 * det_part * exp_part

    return beta1.T @ Q_tilde @ beta2 - mean1 * mean2


def covariance_prop_mc(K1, K2, Lambda1, Lambda2, u, S, X_train, y_train):
    """
    Computes the covariance of GP outputs (A14) using Monte Carlo.
    Assumes we are using Gaussian kernels for both GP models 1 and 2.

    Parameters:
    ----------
    K1 & K2: np.array
       Evidence covariance matrix for GP models 1 and 2
    Lambda1 & Lambda2: np.array
       Diagonal matrix containing kernel parameters for GP models 1 and 2
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
       Covariance of joint predictive distribution of f1 and f2
    """
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    Lambda1_inv = np.linalg.inv(Lambda1)
    Lambda2_inv = np.linalg.inv(Lambda2)

    def gauss_kern(x1, x2, Lambda_inv):
        return np.exp(-1/2 * (x1-x2).T @ Lambda_inv @ (x1-x2))

    T = 10000
    X_star = np.random.multivariate_normal(u, S, size=T)

    num_train = K1.shape[0]
    f1_list = []
    f2_list = []
    for t in range(T):
        k1_vec = np.array([gauss_kern(X_star[t, :], X_train[n, :], Lambda1_inv) for n in range(num_train)])
        k2_vec = np.array([gauss_kern(X_star[t, :], X_train[n, :], Lambda2_inv) for n in range(num_train)])
        mu1 = k1_vec @ K1_inv @ y_train
        mu2 = k2_vec @ K2_inv @ y_train
        sigma1 = np.sqrt(1 - k1_vec @ K1_inv @ k1_vec)
        sigma2 = np.sqrt(1 - k2_vec @ K2_inv @ k2_vec)
        f1 = np.random.normal(loc=mu1, scale=sigma1)
        f2 = np.random.normal(loc=mu2, scale=sigma2)
        f1_list.append(f1)
        f2_list.append(f2)

    return np.cov(f1_list, f2_list)[0, 1]
