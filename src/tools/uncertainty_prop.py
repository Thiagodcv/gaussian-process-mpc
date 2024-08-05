import numpy as np
import scipy
import numba
import warnings
import torch


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
    beta = scipy.linalg.solve(K, y_train, assume_a='pos')
    Lambda_inv = np.linalg.inv(Lambda)
    S_Lambda_inv = np.linalg.inv(S + Lambda)
    l = np.zeros(beta.shape[0])
    d = S.shape[0]

    for j in range(l.shape[0]):
        gauss_cov = (u - X_train[j, :]).T @ S_Lambda_inv @ (u - X_train[j, :])
        l[j] = (np.linalg.det(Lambda_inv @ S + np.identity(d)) ** (-1/2)) * np.exp(-1/2 * gauss_cov)

    return np.dot(beta, l), {'beta': beta, 'l': l}


def mean_prop_mc(K, Lambda, u, S, X_train, y_train, sigma_f=1):
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
    sigma_f: scalar
        sigma_f parameter of the GP model

    Return :
    ------
    scalar
        Mean of predictive distribution of f
    """
    def gauss_kern(x1, x2):
        return (sigma_f**2) * np.exp(-1/2 * (x1-x2).T @ np.linalg.inv(Lambda) @ (x1-x2))

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
        for j in range(num_train):  # Explicit double for-loop has to go.
            x_d = (X_train[i, :] + X_train[j, :]) / 2
            exp_part = np.exp(-1/2 * (u - x_d).T @ half_Lam_S_inv @ (u - x_d) +
                              -1/4 * (X_train[i, :] - X_train[j, :]).T @ Lam_inv @ (X_train[i, :] - X_train[j, :]))

            L[i, j] = det_part * exp_part

    K_inv = np.linalg.inv(K)
    return 1 - np.trace((K_inv - np.outer(beta, beta)) @ L) - mean**2


def variance_prop_mc(K, Lambda, u, S, X_train, y_train, sigma_f=1):
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
    sigma_f: scalar
        sigma_f parameter of the GP model

    Return :
    ------
    scalar
       Variance of predictive distribution of f
    """
    def gauss_kern(x1, x2):
        return (sigma_f**2) * np.exp(-1/2 * (x1-x2).T @ np.linalg.inv(Lambda) @ (x1-x2))

    T = 10000
    X_star = np.random.multivariate_normal(u, S, size=T)

    num_train = K.shape[0]
    K_inv = np.linalg.inv(K)
    mu_list = []
    sig_sq_list = []
    for t in range(T):
        k_vec = np.array([gauss_kern(X_star[t, :], X_train[n, :]) for n in range(num_train)])
        mu = k_vec.T @ K_inv @ y_train
        sig_sq = sigma_f**2 - k_vec.T @ K_inv @ k_vec

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


def covariance_prop_mc(K1, K2, Lambda1, Lambda2, u, S, X_train, y_train, sigma_f1=1, sigma_f2=1):
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
    sigma_f: scalar
        sigma_f parameter of the GP model

    Return :
    ------
    scalar
       Covariance of joint predictive distribution of f1 and f2
    """
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    Lambda1_inv = np.linalg.inv(Lambda1)
    Lambda2_inv = np.linalg.inv(Lambda2)

    def gauss_kern(x1, x2, Lambda_inv, sigma_f):
        return (sigma_f**2) * np.exp(-1/2 * (x1-x2).T @ Lambda_inv @ (x1-x2))

    T = 10000
    X_star = np.random.multivariate_normal(u, S, size=T)

    num_train = K1.shape[0]
    f1_list = []
    f2_list = []
    for t in range(T):
        k1_vec = np.array([gauss_kern(X_star[t, :], X_train[n, :], Lambda1_inv, sigma_f1) for n in range(num_train)])
        k2_vec = np.array([gauss_kern(X_star[t, :], X_train[n, :], Lambda2_inv, sigma_f2) for n in range(num_train)])
        mu1 = k1_vec @ K1_inv @ y_train
        mu2 = k2_vec @ K2_inv @ y_train
        sigma1 = np.sqrt(sigma_f1**2 - k1_vec @ K1_inv @ k1_vec)
        sigma2 = np.sqrt(sigma_f2**2 - k2_vec @ K2_inv @ k2_vec)
        f1 = np.random.normal(loc=mu1, scale=sigma1)
        f2 = np.random.normal(loc=mu2, scale=sigma2)
        f1_list.append(f1)
        f2_list.append(f2)

    return np.cov(f1_list, f2_list)[0, 1]


# Implementing Torch versions of mv_prop, var_prop, covar_prop
def mean_prop_torch(Ky_inv, lambdas, u, S, X_train, y_train, sigma_f=1, nom_model=None, nom_model_hess=None):
    """
    Computes the mean of predictive distribution (21) using an exact formula. Assumes we are using Gaussian kernels.

    Parameters:
    ----------
    Ky_inv: torch.tensor
        Inverse of evidence covariance matrix
    lambdas: torch.tensor
        Tensor array containing kernel parameters
    u: torch.tensor
        Mean of input distribution (which is assumed to be Gaussian)
    S: torch.tensor
        Covariance of input distribution (which is assumed to be Gaussian)
    X_train: torch.tensor
        GP training data inputs
    y_train: torch.tensor
        GP training data outputs
    sigma_f: scalar
        sigma_f parameter of the GP model
    nom_model: function or None
        Scalar function with input-size u, or None if nominal model not being used
    nom_model_hess: function or None
        Hessian of scalar function with input-size u, or None if nominal model not being used

    Return:
    ------
    torch scalar
        Mean of predictive distribution of f
    dict
        Dictionary containing beta and l (equation 31)
    """
    # TODO: if only one datapoint fed into dynamics, y_train is 0D and this fails
    if nom_model is None:
        beta = Ky_inv @ y_train
    else:
        beta = Ky_inv @ (y_train - nom_model(X_train))

    Lambda = torch.diag(lambdas)
    Lambda_inv = torch.diag(1/lambdas)
    S_Lambda_inv = torch.linalg.inv(S + Lambda)
    d = S.shape[0]

    gauss_cov = torch.sum(((u - X_train) @ S_Lambda_inv) * (u - X_train), dim=1)
    l = ((torch.linalg.det(Lambda_inv @ S + torch.eye(d, device=beta.device)) ** (-1/2)) *
         torch.exp(-1/2 * gauss_cov) * sigma_f**2)

    if nom_model is None:
        return torch.dot(beta, l), {'beta': beta, 'l': l}
    else:
        return nom_model(u) + 1/2 * torch.trace(nom_model_hess(u) @ S) + torch.dot(beta, l), {'beta': beta, 'l': l}


def variance_prop_torch(Ky_inv, lambdas, u, S, X_train, mean, beta, sigma_f=1, nom_model_grad=None):
    """
    Computes the variance of predictive distribution (21) using an exact formula.
    Assumes we are using Gaussian kernels.

    Parameters:
    ----------
    Ky_inv: torch.tensor
        Inverse of evidence covariance matrix
    lambdas: torch.tensor
        Tensor array containing kernel parameters
    u: torch.tensor
        Mean of input distribution (which is assumed to be Gaussian)
    S: torch.tensor
        Covariance of input distribution (which is assumed to be Gaussian)
    X_train: torch.tensor
        GP training data inputs
    y_train: torch.tensor
        GP training data outputs
    mean: scalar
        mean of predictive distribution of GP with uncertain input (equation 21)
    beta: torch.tensor
        vector used in dot-product to compute mean in (21)
    sigma_f: scalar
        sigma_f parameter of the GP model

    Return:
    ------
    torch scalar
       Variance of predictive distribution of f
    """
    num_train = X_train.shape[0]
    d = S.shape[0]
    Lambda = torch.diag(lambdas)
    Lam_inv = torch.diag(1/lambdas)
    half_Lam_S_inv = torch.linalg.inv(Lambda / 2 + S)
    det_part = torch.linalg.det(2 * Lam_inv @ S + torch.eye(d, device=X_train.device)) ** (-1/2)

    # Compute A_part
    u_A_Xi = (u @ half_Lam_S_inv @ X_train.mT)[:, None]
    u_A_Xj = u_A_Xi.mT

    u_Xij_diff = u @ half_Lam_S_inv @ u + X_train @ half_Lam_S_inv @ X_train.mT - u_A_Xi - u_A_Xj
    # assert np.abs((u_Xij_diff[3, 5] - (u - X_train[3, :]) @ half_Lam_S_inv @ (u - X_train[5, :])).item()) < 1e-5

    u_Xi_diff = torch.diag(u_Xij_diff)[:, None]
    u_Xj_diff = u_Xi_diff.mT

    A_part = torch.exp((-1/8) * (u_Xi_diff + 2*u_Xij_diff + u_Xj_diff))

    # Compute Lambda_part
    X_train_mod = X_train * torch.sqrt(1 / lambdas)
    dist_mat = torch.cdist(X_train_mod, X_train_mod, p=2)
    Lambda_part = torch.exp((-1/4) * torch.square(dist_mat))

    # Compute entire L matrix
    L = det_part * A_part * Lambda_part * sigma_f**4

    if nom_model_grad is not None:
        var_nom_model = nom_model_grad(u).T @ S @ nom_model_grad(u)
        return sigma_f**2 - torch.trace((Ky_inv - torch.outer(beta, beta)) @ L) - mean**2 + var_nom_model

    return sigma_f**2 - torch.trace((Ky_inv - torch.outer(beta, beta)) @ L) - mean**2


def covariance_prop_torch(lambdas1, lambdas2, u, S, X_train, mean1, mean2, beta1, beta2, sigma_f1=1, sigma_f2=1):
    """
    Computes the covariance of GP outputs (A14) using an exact formula.
    Assumes we are using Gaussian kernels for both GP models 1 and 2.

    NOTE: functionality for nominal models not implemented in this function.

    Parameters:
    ----------
    K1 & K2: torch.tensor
       Evidence covariance matrix for GP models 1 and 2
    Lambda1 & Lambda2: torch.tensor
       tensor containing kernel parameters for GP models 1 and 2
    u: torch.tensor
       Mean of input distribution (which is assumed to be Gaussian)
    S: torch.tensor
       Covariance of input distribution (which is assumed to be Gaussian)
    X_train: torch.tensor
       GP training data inputs
    y_train: torch.tensor
       GP training data outputs
    mean1 & mean2: torch.tensor
        mean of predictive distribution of GP with uncertain input (equation 21)
    beta1 & beta2: torch.tensor
        vector used in dot-product to compute mean in (21)
    sigma_f1 & sigma_f2: scalar
        sigma_f parameters of the GP models

    Return :
    ------
    torch scalar
       Covariance of joint predictive distribution of f1 and f2
    """
    Lambda1_inv = torch.diag(1/lambdas1)
    Lambda2_inv = torch.diag(1/lambdas2)

    num_train = beta1.shape[0]
    d = Lambda1_inv.shape[0]

    det_part = torch.linalg.det(S @ (Lambda1_inv + Lambda2_inv) + torch.eye(d, device=beta1.device))**(-1/2)

    z1 = Lambda1_inv @ (X_train - u).mT
    z2 = Lambda2_inv @ (X_train - u).mT
    A_mat = torch.linalg.inv(S @ (Lambda1_inv + Lambda2_inv) + torch.eye(d, device=beta1.device)) @ S
    A_z1 = torch.sum((A_mat @ z1) * z1, dim=0)[:, None]
    A_z2 = torch.sum((A_mat @ z2) * z2, dim=0)[:, None]
    exp_part = torch.exp((1 / 2) * (A_z1 + 2 * z2.mT @ A_mat @ z1 + A_z2.mT))

    # Just for testing purposes
    # z_mat = z1[:, :, None] + z2[:, None, :]
    # assert torch.linalg.norm(exp_part[5, 6] - torch.exp((1/2) * z_mat[:, 5, 6] @ A_mat @ z_mat[:, 5, 6])).item() < 1e-5

    X_train_mod1 = X_train * torch.sqrt(1 / lambdas1)
    u_mod1 = u * torch.sqrt(1 / lambdas1)
    dist_mat1 = torch.cdist(X_train_mod1, u_mod1[None, :], p=2)
    k1 = torch.square(dist_mat1)

    X_train_mod2 = X_train * torch.sqrt(1 / lambdas2)
    u_mod2 = u * torch.sqrt(1 / lambdas2)
    dist_mat2 = torch.cdist(X_train_mod2, u_mod2[None, :], p=2)
    k2 = torch.square(dist_mat2)

    cov_part = torch.exp((-1/2)*(k1 + k2.mT)) * (sigma_f1**2) * (sigma_f2**2)
    Q_tilde = det_part * cov_part * exp_part

    return beta1 @ Q_tilde @ beta2 - mean1 * mean2
