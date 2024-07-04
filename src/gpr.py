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
        self.Kf = None

        # Covariance matrix plus variance of noise, and its inverse
        self.Ky = None
        self.Ky_inv = None

        # Hyperparameters to update via backprop
        self.log_lambdas = torch.zeros(x_dim, device=self.device).type(torch.float32).requires_grad_()
        self.log_sigma_n = torch.tensor(0.0, device=self.device).type(torch.float32).requires_grad_()
        self.log_sigma_f = torch.tensor(0.0, device=self.device).type(torch.float32).requires_grad_()

        # Optimization
        self.optimizer = torch.optim.Adam(params=[self.log_lambdas, self.log_sigma_n, self.log_sigma_f],
                                          lr=0.005,
                                          betas=(0.9, 0.999),
                                          maximize=True)

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
            self.num_train += 1
            self.X_train = x
            self.y_train = y
            self.Kf = torch.tensor([[self.se_kernel(x, x)]], requires_grad=False).to(self.device)  # requires_grad?
            self.Ky = self.Kf + torch.exp(self.log_sigma_n) ** 2
            self.Ky_inv = 1 / self.Ky
        else:
            self.num_train += 1
            self.X_train = torch.cat((self.X_train, x), dim=0)
            self.y_train = torch.cat((self.y_train, y), dim=0)

            # Update A inverse matrix
            self.build_Ky_inv_mat()

    def se_kernel(self, x1, x2):
        """
        The squared exponential kernel function.
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

        return {'lambda': dK_dlambda, 'sigma_f': dK_dsigma_f, 'sigma_n': dK_dsigma_n}

    def marginal_likelihood_grad(self, gradient_dict):
        """
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

        alpha = self.Ky_inv @ self.y_train
        B = alpha @ alpha.mT - self.Ky_inv
        for i in range(self.x_dim):
            dml_dlambda[i] = 1/2*torch.trace(B @ dK_dlambda[:, :, i])

        dml_dsigma_f = 1/2*torch.trace(B @ dK_dsigma_f)
        dml_dsigma_n = 1/2*torch.trace(B @ dK_dsigma_n)
        return {'lambda': dml_dlambda, 'sigma_f': dml_dsigma_f, 'sigma_n': dml_dsigma_n}

    def compute_marginal_likelihood(self):
        """
        Computes and returns the marginal likelihood.
        """
        return (-1/2 * self.y_train.mT @ self.Ky_inv @ self.y_train -
                1/2 * torch.log(torch.linalg.det(self.Ky)) -
                self.num_train/2 * np.log(2*np.pi))

    def update_hyperparams(self, num_iters=1000):
        for iter in range(num_iters):
            self.optimizer.zero_grad()
            ml = self.compute_marginal_likelihood()
            ml.backward()
            self.optimizer.step()
            self.build_Ky_inv_mat()  # Update matrices used to compute marginal likelihood under new hyperparameters

            print('Iter: ', iter)
            print('ml: ', ml.item())
            print('lambdas: ', torch.exp(self.log_lambdas).cpu().detach().numpy())
            print('sigma_f: ', torch.exp(self.log_sigma_f).item())
            print('sigma_n: ', torch.exp(self.log_sigma_n).item())

            print('log_lambdas.grad: ', self.log_lambdas.grad.cpu().detach().numpy())
            print('log_sigma_f.grad: ', self.log_sigma_f.grad.item())
            print('log_sigma_n.grad: ', self.log_sigma_n.grad.item())
            print('----------------------------------------')

