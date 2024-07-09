import numpy as np
import torch


class GaussianProcessRegression(object):
    """
    A class which implement Gaussian Process Regression.
    In this class we use the squared-exponential (SE) covariance function.
    """

    def __init__(self, x_dim, nominal_model=None):
        """
        Parameters:
        ----------
        x_dim: int
            Dimension of input variable.
        nominal_model: function
            A nominal mean function. Takes a torch tensor as input (each row is a different observation) and
            returns the nominal model evaluated at those observations.
        """
        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Training data
        self.x_dim = x_dim
        self.num_train = 0
        self.y_train = None
        self.X_train = None

        # Covariance matrix
        self.Kf = None

        # Covariance matrix plus variance of noise, and its inverse
        self.Ky = None
        self.Ky_inv = None

        # Hyperparameters to update via backprop
        self.log_lambdas = torch.zeros(x_dim, device=self.device).type(torch.float64).requires_grad_()
        self.log_sigma_n = torch.tensor(0.0, device=self.device).type(torch.float64).requires_grad_()
        self.log_sigma_f = torch.tensor(0.0, device=self.device).type(torch.float64).requires_grad_()

        # Nominal model (must be capable of batch computation)
        self.f_nom = nominal_model

        # Optimization
        self.optimizer = torch.optim.Adam(params=[self.log_lambdas, self.log_sigma_n, self.log_sigma_f],
                                          lr=0.1,  # 0.005,
                                          betas=(0.9, 0.999),
                                          maximize=True)

    def set_lambdas(self, lambdas):
        """
        If training data already loaded, need to run self.build_Ky_inv_mat() to update matrices.

        Parameters:
        ----------
        lambdas: np.array
        """
        self.log_lambdas = torch.log(torch.tensor(lambdas, device=self.device)).type(torch.float64).requires_grad_()

    def get_lambdas(self):
        return torch.exp(self.log_lambdas).cpu().detach().numpy()

    def set_sigma_f(self, sigma_f):
        """
        If training data already loaded, need to run self.build_Ky_inv_mat() to update matrices.

        Parameters:
        ----------
        sigma_f: scalar
        """
        self.log_sigma_f = torch.log(torch.tensor(sigma_f, device=self.device)).type(torch.float64).requires_grad_()

    def get_sigma_f(self):
        return torch.exp(self.log_sigma_f).item()

    def set_sigma_n(self, sigma_n):
        """
        If training data already loaded, need to run self.build_Ky_inv_mat() to update matrices.

        Parameters:
        ----------
        sigma_n: scalar
        """
        self.log_sigma_n = torch.log(torch.tensor(sigma_n, device=self.device)).type(torch.float64).requires_grad_()

    def get_sigma_n(self):
        return torch.exp(self.log_sigma_n).item()

    def append_train_data(self, x, y):
        """
        Append (x, y) to preexisting training data.

        Parameters
        ----------
        x: (x_dim, ) numpy array
        y: (num_obs, ) numpy array or scalar
        """
        if not np.isscalar(y):
            num_obs = len(y)
            y = y[:, None]
        else:
            num_obs = 1
            y = np.array([y])[:, None]

        if num_obs == 1:
            x = np.reshape(x, (1, self.x_dim))

        x = torch.tensor(x, requires_grad=False).type(torch.float64).to(self.device)
        y = torch.tensor(y, requires_grad=False).type(torch.float64).to(self.device)

        if self.num_train == 0:
            self.num_train += num_obs
            self.X_train = x
            self.y_train = y
        else:
            self.num_train += num_obs
            self.X_train = torch.cat((self.X_train, x), dim=0)
            self.y_train = torch.cat((self.y_train, y), dim=0)

        # Update Kf, Ky, and inverse(Ky) matrices
        self.build_Ky_inv_mat()

    def se_kernel(self, x1, x2):
        """
        The squared exponential kernel function implemented for Torch.
        """
        lambdas = torch.exp(self.log_lambdas)
        sigma_f = torch.exp(self.log_sigma_f)

        x1 = torch.squeeze(x1)
        x2 = torch.squeeze(x2)
        inv_lambda = torch.diag(1 / lambdas)

        return (sigma_f**2) * torch.exp(-1/2 * (x1 - x2) @ inv_lambda @ (x1 - x2))

    def update_Ky_inv_mat(self, k_new):
        """
        TODO: Simply rebuilding covariance matrix is faster. Don't use this.
        Update A_inv_mat to include a new datapoint in X_train.
        """
        sigma_f = torch.exp(self.log_sigma_f)
        sigma_n = torch.exp(self.log_sigma_n)

        B = k_new
        C = k_new.mT
        D = torch.reshape(sigma_n ** 2 + sigma_f ** 2, (1, 1))
        Q = 1./(D - C @ self.Ky_inv @ B)  # Just inverting a scalar

        new_Ky_inv_top_left = self.Ky_inv + self.Ky_inv @ B @ Q @ C @ self.Ky_inv
        new_Ky_inv_top_right = -self.Ky_inv @ B @ Q
        new_Ky_inv_bottom_left = -Q @ C @ self.Ky_inv
        new_Ky_inv_bottom_right = Q

        new_Ky_inv_top = torch.cat((new_Ky_inv_top_left, new_Ky_inv_top_right), dim=1)
        new_Ky_inv_bottom = torch.cat((new_Ky_inv_bottom_left, new_Ky_inv_bottom_right), dim=1)
        self.Ky_inv = torch.cat((new_Ky_inv_top, new_Ky_inv_bottom), dim=0)

    def build_Ky_inv_mat(self):
        """
        Builds Kf, Ky, and Ky_inv from scratch using training data.
        """
        lambdas = torch.exp(self.log_lambdas)
        sigma_f = torch.exp(self.log_sigma_f)
        sigma_n = torch.exp(self.log_sigma_n)

        X_train_mod = self.X_train * torch.sqrt(1 / lambdas)
        dist_mat = torch.cdist(X_train_mod, X_train_mod, p=2)
        self.Kf = (sigma_f ** 2) * torch.exp(-1 / 2 * torch.square(dist_mat))
        self.Ky = self.Kf + sigma_n ** 2 * torch.eye(self.num_train, device=self.device)
        self.Ky_inv = torch.linalg.inv(self.Ky)

    def kernel_matrix_gradient(self):
        """
        Computes the gradients of K_y with respect to lambda_j, sigma_n, and sigma_f.
        Assumes a fully updated K_f matrix has been computed.

        Returns:
        -------
        dict containing gradients in torch.tensor format
        """
        lambdas = torch.exp(self.log_lambdas)
        sigma_f = torch.exp(self.log_sigma_f)
        sigma_n = torch.exp(self.log_sigma_n)

        A = torch.zeros((self.num_train, self.num_train, self.x_dim), device=self.device)
        for k in range(self.x_dim):
            v = torch.reshape(self.X_train[:, k], (self.num_train, 1))
            A[:, :, k] = (1 / (2 * lambdas[k] ** 2)) * torch.square(torch.cdist(v, v, p=2))

        dK_dlambda = torch.zeros((self.num_train, self.num_train, self.x_dim), device=self.device)
        for k in range(self.x_dim):
            dK_dlambda[:, :, k] = torch.multiply(self.Kf, A[:, :, k])

        dK_dsigma_f = 2 / sigma_f * self.Kf
        dK_dsigma_n = 2 * sigma_n * torch.eye(self.num_train, device=self.device)

        return {'lambda': dK_dlambda.type(torch.float64),
                'sigma_f': dK_dsigma_f.type(torch.float64),
                'sigma_n': dK_dsigma_n.type(torch.float64)}

    def marginal_likelihood_grad(self, gradient_dict):
        """
        TODO: seems to be returning wrong gradients. Since using PyTorch autograd, this isn't a priority at the moment.
        Computes the gradients of the marginal likelihood with respect to lambda_j, sigma_n, and sigma_f.
        Assumes K_f and inverse(K_y) have already been fully updated.

        Parameters:
        ----------
        gradient_dict: dict
            A dictionary containing the gradients of K_y w.r.t. hyperparameters

        Returns:
        -------
        dict containing gradients in torch.tensor format
        """
        dml_dlambda = torch.zeros(size=(self.x_dim,), device=self.device)
        dK_dlambda = gradient_dict['lambda']
        dK_dsigma_f = gradient_dict['sigma_f']
        dK_dsigma_n = gradient_dict['sigma_n']

        if self.f_nom is None:
            alpha = self.Ky_inv @ self.y_train
        else:
            alpha = self.Ky_inv @ (self.y_train - self.f_nom(self.X_train))

        B = alpha @ alpha.mT - self.Ky_inv
        for i in range(self.x_dim):
            dml_dlambda[i] = 1/2*torch.trace(B @ dK_dlambda[:, :, i])

        dml_dsigma_f = 1/2*torch.trace(B @ dK_dsigma_f)
        dml_dsigma_n = 1/2*torch.trace(B @ dK_dsigma_n)

        dml_dlog_lambda = torch.multiply(dml_dlambda, torch.exp(self.log_lambdas))
        dml_dlog_sigma_f = dml_dsigma_f * torch.exp(self.log_sigma_f)
        dml_dlog_sigma_n = dml_dsigma_n * torch.exp(self.log_sigma_n)
        return {'lambda': dml_dlambda, 'sigma_f': dml_dsigma_f, 'sigma_n': dml_dsigma_n,
                'log_lambda': dml_dlog_lambda, 'log_sigma_f': dml_dlog_sigma_f, 'log_sigma_n': dml_dlog_sigma_n}

    def compute_marginal_likelihood(self):
        """
        Computes and returns the marginal likelihood.
        """
        if self.f_nom is None:
            return (-1/2 * self.y_train.mT @ self.Ky_inv @ self.y_train -
                    1/2 * torch.log(torch.linalg.det(self.Ky)) -
                    self.num_train/2 * np.log(2*np.pi))
        else:
            return (-1/2 * (self.y_train - self.f_nom(self.X_train)).mT @ self.Ky_inv @ (self.y_train - self.f_nom(self.X_train)) -
                    1/2 * torch.log(torch.linalg.det(self.Ky)) -
                    self.num_train/2 * np.log(2*np.pi))

    def compute_pred_train_covariance(self, X_pred):
        """
        Compute K(X*, X_train) matrix found in equations (2.22), (2.23).

        Parameters:
        ----------
        X_pred: (p, x_dim) or (x_dim,) np.array

        Returns:
        -------
        (p, num_train) torch.tensor if p>1 observations,
        (num_train,) torch.tensor if one observation.
        """
        X_pred = torch.tensor(X_pred, device=self.device).type(torch.float64)

        # If predicting on multiple test points at once
        if len(X_pred.shape) == 2:
            lambdas = torch.exp(self.log_lambdas)
            sigma_f = torch.exp(self.log_sigma_f)

            X_train_mod = self.X_train * torch.sqrt(1 / lambdas)
            X_pred_mod = X_pred * torch.sqrt(1 / lambdas)
            dist_mat = torch.cdist(X_pred_mod, X_train_mod, p=2)
            K_pred_train = (sigma_f ** 2) * torch.exp(-1 / 2 * torch.square(dist_mat))

        # If just one test point
        else:
            # Might be faster to do this fully in Python
            K_pred_train = torch.tensor([self.se_kernel(X_pred, self.X_train[i, :]) for i in range(self.num_train)],
                                        device=self.device).type(torch.float64)
        return K_pred_train

    def predict_latent_vars(self, X_pred, covar=False, targets=False):
        """
        Implements equation 2.23 & 2.24.

        Parameters:
        ----------
        X_pred: (p, x_dim) or (x_dim,) np.array
        covar: boolean
            If set to True, method returns covariance of predictions (whether the method is used
            to predict latent variables f or targets y).
        targets: boolean
            If set to True, method returns covariance of targets y instead of latent variables f.
            If covar=False, changes nothing.

        Returns:
        -------
        np.array and None if covar=False, else two np.array
        """
        K_pred_train = self.compute_pred_train_covariance(X_pred)

        if self.f_nom is None:
            f_pred = K_pred_train @ self.Ky_inv @ self.y_train
        else:
            X_pred_torch = torch.tensor(X_pred, device=self.device).type(torch.float64)
            f_pred = K_pred_train @ self.Ky_inv @ (self.y_train - self.f_nom(self.X_train)) + self.f_nom(X_pred_torch)

        if not covar:
            return f_pred.cpu().detach().numpy(), None
        else:
            # Convert to Torch
            lambdas = torch.exp(self.log_lambdas)
            sigma_f = torch.exp(self.log_sigma_f)
            sigma_n = torch.exp(self.log_sigma_n)
            X_pred = torch.tensor(X_pred, device=self.device).type(torch.float64)

            # Compute covariance matrix between predictions
            X_pred_mod = X_pred * torch.sqrt(1 / lambdas)
            dist_mat = torch.cdist(X_pred_mod, X_pred_mod, p=2)
            K_pred_pred = (sigma_f ** 2) * torch.exp(-1 / 2 * torch.square(dist_mat))

            cov = K_pred_pred - K_pred_train @ self.Ky_inv @ K_pred_train.mT

            # If predicting targets and not latent variables
            if targets:
                p = X_pred.shape[0] if len(X_pred.shape) == 2 else 1
                cov += (sigma_n ** 2) * torch.eye(p, device=self.device).type(torch.float64)

            return f_pred.cpu().detach().numpy(), cov.cpu().detach().numpy()

    def update_hyperparams(self, num_iters=1000):
        for iter in range(num_iters):
            self.optimizer.zero_grad()
            ml = self.compute_marginal_likelihood()
            ml.backward()

            # if torch.isnan(self.log_lambdas.grad).any().item() or torch.isnan(self.log_sigma_f.grad).any().item() or torch.isnan(self.log_sigma_n.grad).any().item():
            #     grad_dict = self.kernel_matrix_gradient()
            #     ml_grad = self.marginal_likelihood_grad(grad_dict)
            #     self.log_lambdas.grad = ml_grad['log_lambda']
            #     self.log_sigma_f.grad = ml_grad['log_sigma_f']
            #     self.log_sigma_n.grad = ml_grad['log_sigma_n']

            self.optimizer.step()
            self.build_Ky_inv_mat()  # Update matrices used to compute marginal likelihood under new hyperparameters

            print('Iter: ', iter)
            print('ml: ', ml.item())
            print('lambdas: ', torch.exp(self.log_lambdas).cpu().detach().numpy())
            print('sigma_f: ', torch.exp(self.log_sigma_f).item())
            print('sigma_n: ', torch.exp(self.log_sigma_n).item())

            log_lambdas_grad = self.log_lambdas.grad.cpu().detach().numpy()
            log_sigma_f_grad = self.log_sigma_f.grad.item()
            log_sigma_n_grad = self.log_sigma_n.grad.item()

            print('log_lambdas.grad: ', log_lambdas_grad)
            print('log_sigma_f.grad: ', log_sigma_f_grad)
            print('log_sigma_n.grad: ', log_sigma_n_grad)

            print('Ky condition number: ', torch.linalg.cond(self.Ky).item())
            print('----------------------------------------')

            if ((np.abs(log_lambdas_grad) < 1e-5).all() and
                    np.abs(log_sigma_f_grad) < 1e-5 and
                    np.abs(log_sigma_n_grad) < 1e-5):
                break
