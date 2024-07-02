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

        # Inverse of covariance matrix plus variance of noise "the A-inverse matrix"
        self.Ky_inv = None

        # Hyperparameters to update via backprop
        self.lambdas = torch.ones(x_dim, device=self.device).type(torch.float32).requires_grad_()
        self.sigma_n = torch.tensor(0.75, device=self.device).type(torch.float32).requires_grad_()
        self.sigma_f = torch.tensor(0.75, device=self.device).type(torch.float32).requires_grad_()

        # Optimization
        self.optimizer = torch.optim.LBFGS(params=[self.lambdas, self.sigma_n, self.sigma_f],
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
            self.Kf = torch.tensor([[self.se_kernel(x, x)]], requires_grad=False).to(self.device)  # requires_grad?
            self.Ky_inv = 1 / (self.Kf + self.sigma_n ** 2)
        else:
            self.X_train = torch.cat((self.X_train, x), dim=0)
            self.y_train = torch.cat((self.y_train, y), dim=0)

            # Update A inverse matrix
            k_new = torch.tensor([self.se_kernel(x, self.X_train[i, :]) for i in range(self.num_train)],
                                 requires_grad=False).to(self.device)
            k_new = torch.reshape(k_new, (k_new.shape[0], 1))
            self.update_Ky_inv_mat(k_new)

            # Update covariance matrix K
            self.Kf = torch.cat((self.Kf, k_new.mT), dim=0)
            k_new_ext = torch.cat((k_new, torch.tensor([[self.se_kernel(x, x)]]).to(self.device)), dim=0)
            self.Kf = torch.cat((self.Kf, k_new_ext), dim=1)

        self.num_train += 1

    def se_kernel(self, x1, x2):
        """
        The squared exponential kernel function.
        """
        x1 = torch.squeeze(x1)
        x2 = torch.squeeze(x2)
        inv_lambda = torch.diag(1 / self.lambdas)

        return (self.sigma_f**2) * torch.exp(-1/2 * (x1 - x2) @ inv_lambda @ (x1 - x2))

    def update_Ky_inv_mat(self, k_new):
        """
        TODO: See if just doing build_Ky_inv_mat is faster.
        Update A_inv_mat to include a new datapoint in X_train.
        """
        B = k_new
        C = k_new.mT
        D = torch.reshape(self.sigma_n ** 2 + self.sigma_f ** 2, (1, 1))
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
        Builds A_inv from scratch using training data.
        """
        X_train_mod = self.X_train * torch.sqrt(1 / self.lambdas)
        dist_mat = torch.cdist(X_train_mod, X_train_mod, p=2)
        self.Kf = (self.sigma_f ** 2) * torch.exp(-1 / 2 * torch.square(dist_mat))

        self.Ky_inv = torch.linalg.inv(self.Kf + self.sigma_n ** 2 * torch.eye(self.num_train, device=self.device))

    def update_hyperparams(self):
        """
        TODO: Figure out constraints and convergence condition
        Find estimate of GP hyperparameters (listed in the constructor) by running
        gradient ascent on marginal likelihood. Does num_iters number of iterations
        unless gradient reaches a local min beforehand.
        """
        num_iters = 10
        alpha = 0.01
        for iter in range(num_iters):
            print('iter: {}, lambda: {}, sigma_f: {}, sigma_n: {}'.format(iter,
                                                                          self.lambdas.cpu().detach().numpy(),
                                                                          self.sigma_f.cpu().detach().numpy(),
                                                                          self.sigma_n.cpu().detach().numpy()))
            Ky_grad_dict = self.kernel_matrix_gradient()
            ml_grad_dict = self.marginal_likelihood_grad(Ky_grad_dict)

            # Update hyperparameters
            with torch.no_grad():
                self.lambdas += alpha * ml_grad_dict['lambda']
                self.sigma_f += alpha * ml_grad_dict['sigma_f']
                self.sigma_n += alpha * ml_grad_dict['sigma_n']

                # If sigmas negative set to zero
                self.lambdas[self.lambdas < 0] = 0.
                self.sigma_f[self.sigma_f < 0] = 0.
                self.sigma_n[self.sigma_n < 0] = 0.

            print('lambda gradient norm: ', torch.linalg.norm(ml_grad_dict['lambda']))
            print('sigma_f gradient: ', ml_grad_dict['sigma_f'])
            print('sigma_e gradient: ', ml_grad_dict['sigma_n'])
            print('----------------------------------------------')

            norm_sum = (torch.linalg.norm(ml_grad_dict['lambda']) +
                        torch.linalg.norm(ml_grad_dict['sigma_f']) +
                        torch.linalg.norm(ml_grad_dict['sigma_n']))
            if norm_sum.item() < 1e-5:
                # Some parts of likelihood can be really flat, so maybe should use a different condition?
                break

            # Update K_f and inverse(K_y)
            self.build_Ky_inv_mat()

    def kernel_matrix_gradient(self):
        """
        Computes the gradients of K_y with respect to lambda_j, sigma_n, and sigma_f.
        Assumes a fully updated K_f matrix has been computed.

        Returns:
        -------
        dict containing gradients in torch.tensor format
        """
        A = torch.zeros((self.num_train, self.num_train, self.x_dim), device=self.device)
        for k in range(self.x_dim):
            v = torch.reshape(self.X_train[:, k], (self.num_train, 1))
            A[:, :, k] = (1 / (2 * self.lambdas[k] ** 2)) * torch.square(torch.cdist(v, v, p=2))

        dK_dlambda = torch.zeros((self.num_train, self.num_train, self.x_dim), device=self.device)
        for k in range(self.x_dim):
            dK_dlambda[:, :, k] = torch.multiply(self.Kf, A[:, :, k])

        dK_dsigma_f = 2 / self.sigma_f * self.Kf
        dK_dsigma_n = 2 * self.sigma_n * torch.eye(self.num_train, device=self.device)

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
