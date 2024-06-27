import numpy as np
import torch


class GaussianProcessRegression(object):
    """
    A class which implement Gaussian Process Regression.
    In this class we use the squared-exponential (SE) covariance function.
    """

    def __init__(self, x_dim):
        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Training data
        self.x_dim = x_dim
        self.num_train = 0
        self.y_train = None
        self.X_train = None

        # Covariance matrix
        self.K = None

        # Inverse of covariance matrix plus variance of noise "the A-inverse matrix"
        self.A_inv = None

        # Hyperparameters to update via backprop
        self.Lambda = torch.ones(x_dim, device=self.device).type(torch.float32).requires_grad_()
        self.sigma_e = torch.tensor(1., device=self.device).type(torch.float32).requires_grad_()
        self.sigma_f = torch.tensor(1., device=self.device).type(torch.float32).requires_grad_()

        # Optimization
        self.optimizer = torch.optim.LBFGS(params=[self.Lambda, self.sigma_e, self.sigma_f],
                                           lr=1e-1,
                                           max_iter=20)

    def append_train_data(self, x, y):
        """
        Append (x, y) to preexisting training data.

        Parameters
        ----------
        x: (x_dim, ) numpy array
        y: scalar or numpy array
        """
        x = np.reshape(x, (1, self.x_dim))
        y = np.reshape(y, (1, 1))

        x = torch.tensor(x, requires_grad=False).type(torch.float32).to(self.device)
        y = torch.tensor(y, requires_grad=False).type(torch.float32).to(self.device)

        if self.num_train == 0:
            self.X_train = x
            self.y_train = y
            self.K = torch.tensor([[self.se_kernel(x, x)]], requires_grad=False).to(self.device)  # requires_grad?
            self.A_inv = 1/(self.K + self.sigma_e**2)
        else:
            self.X_train = torch.cat((self.X_train, x), dim=0)
            self.y_train = torch.cat((self.y_train, y), dim=0)

            # Update A inverse matrix
            k_new = torch.tensor([self.se_kernel(x, self.X_train[i, :]) for i in range(self.num_train)],
                                 requires_grad=False).to(self.device)
            k_new = torch.reshape(k_new, (k_new.shape[0], 1))
            self.update_A_inv_mat(k_new)

            # Update covariance matrix K
            self.K = torch.cat((self.K, k_new.mT), dim=0)
            k_new_ext = torch.cat((k_new, torch.tensor([[self.se_kernel(x, x)]]).to(self.device)), dim=0)
            self.K = torch.cat((self.K, k_new_ext), dim=1)

        self.num_train += 1

    def se_kernel(self, x1, x2):
        """
        The squared exponential kernel function.
        """
        x1 = torch.squeeze(x1)
        x2 = torch.squeeze(x2)
        inv_lambda = torch.diag(1/self.Lambda)

        return (self.sigma_f**2) * torch.exp(-1/2 * (x1 - x2) @ inv_lambda @ (x1 - x2))

    def update_A_inv_mat(self, k_new):
        """
        Update A_inv_mat to include a new datapoint in X_train.
        """
        B = k_new
        C = k_new.mT
        D = torch.reshape(self.sigma_e**2 + self.sigma_f**2, (1, 1))
        Q = 1./(D - C @ self.A_inv @ B)  # Just inverting a scalar

        new_A_inv_top_left = self.A_inv + self.A_inv @ B @ Q @ C @ self.A_inv
        new_A_inv_top_right = -self.A_inv @ B @ Q
        new_A_inv_bottom_left = -Q @ C @ self.A_inv
        new_A_inv_bottom_right = Q

        new_A_inv_top = torch.cat((new_A_inv_top_left, new_A_inv_top_right), dim=1)
        new_A_inv_bottom = torch.cat((new_A_inv_bottom_left, new_A_inv_bottom_right), dim=1)
        self.A_inv = torch.cat((new_A_inv_top, new_A_inv_bottom), dim=0)

    def build_A_inv_mat(self):
        """
        Builds A_inv from scratch using training data.
        """
        self.K = torch.zeros(size=self.K.shape, device=self.device, requires_grad=False).type(torch.float32)
        for i in range(self.num_train):
            for j in range(self.num_train):
                self.K[i, j] = self.se_kernel(self.X_train[i, :], self.X_train[j, :])

        self.A_inv = torch.linalg.inv(self.K + self.sigma_e**2 * torch.eye(self.num_train, device=self.device))

    def update_hyperparams(self):
        """
        Find estimate of GP hyperparameters (listed in the constructor) by maximizing
        the marginal likelihood.
        """
        pass

    def kernel_matrix_gradient(self):
        """
        Computes the gradients of K_y with respect to lambda_j, sigma_n, and sigma_f.
        Assumes a fully updated K_f matrix has been computed.

        Returns:
        -------
        dict containing gradients in torch.tensor format
        """
        A = torch.zeros((self.num_train, self.num_train, self.x_dim), device=self.device)
        for i in range(self.num_train):
            for j in range(self.num_train):
                for k in range(self.x_dim):
                    A[i, j, k] = (1 / (2 * self.Lambda[k] ** 2)) * (self.X_train[i, k] - self.X_train[j, k]) ** 2

        dK_dlambda = torch.zeros((self.num_train, self.num_train, self.x_dim), device=self.device)
        for k in range(self.x_dim):
            dK_dlambda[:, :, k] = torch.multiply(self.K, A[:, :, k])

        dK_dsigma_f = 2 / self.sigma_f * self.K
        dK_dsigma_n = 2 * self.sigma_e * torch.eye(self.num_train, device=self.device)

        return {'lambda': dK_dlambda, 'sigma_f': dK_dsigma_f, 'sigma_n': dK_dsigma_n}

    def marginal_likelihood_grad(self, gradient_dict):
        """
        TODO: Write a test for this method
        Computes the gradients of the marginal likelihood with respect to lambda_j, sigma_n, and sigma_f.
        Assumes K_f and inverse(K_y) have already been fully updated.

        Parameters:
        ----------
        gradient_dict: dict
            A dictionary containing the gradients of K_y w.r.t. hyperparameters

        Returns:
        -------
        (d+2,) tensor
            The gradient of the marginal likelihood w.r.t hyperparameters. The first d elements are the
            lambda gradients, then the sigma_f and sigma_n gradients respectively.
        """
        dml_dtheta = torch.zeros(size=(self.x_dim+2,), device=self.device)
        dK_dlambda = gradient_dict['lambda']
        dK_dsigma_f = gradient_dict['sigma_f']
        dK_dsigma_n = gradient_dict['sigma_n']

        alpha = self.A_inv @ self.y_train
        B = alpha @ alpha.mT - self.A_inv
        for i in range(self.x_dim):
            dml_dtheta[i] = 1/2*torch.trace(B @ dK_dlambda[:, :, i])

        dml_dtheta[self.x_dim] = 1/2*torch.trace(B @ dK_dsigma_f)
        dml_dtheta[self.x_dim+1] = 1/2*torch.trace(B @ dK_dsigma_n)
        return dml_dtheta
