import numpy as np


class GaussianProcessRegression(object):
    """
    A class which implement Gaussian Process Regression.
    In this class we use the squared-exponential (SE) covariance function.
    """

    def __init__(self, x_dim):
        # Training data
        self.x_dim = x_dim
        self.num_train = 0
        self.y_train = None
        self.X_train = None

        # Covariance matrix
        self.K = None

        # Inverse of covariance matrix plus variance of noise "the A-inverse matrix"
        self.A_inv = None

        # Hyperparameters
        self.Lambda = np.identity(self.x_dim)
        self.sigma_e = 1
        self.sigma_f = 1

    def append_train_data(self, x, y):
        """
        Append (x, y) to preexisting training data.

        Parameters
        ----------
        x: (x_dim) numpy array
        y: (1, 1) numpy array
        """
        x = np.reshape(x, (1, self.x_dim))
        y = np.reshape(y, (1, 1))

        if self.num_train == 0:
            self.X_train = x
            self.y_train = y
            self.K = np.array([[self.se_kernel(x, x)]])
            self.A_inv = 1/(self.K + self.sigma_e**2)
        else:
            self.X_train = np.concatenate((self.X_train, x), axis=0)
            self.y_train = np.concatenate((self.y_train, y), axis=0)

            # Update A inverse matrix
            k_new = np.array([self.se_kernel(x, self.X_train[i, :]) for i in range(self.num_train)])
            k_new = np.reshape(k_new, (k_new.shape[0], 1))
            self.update_A_inv_mat(k_new)

            # Update covariance matrix K
            self.K = np.concatenate((self.K, k_new.T), axis=0)
            k_new_ext = np.concatenate((k_new, np.array([[self.se_kernel(x, x)]])), axis=0)
            self.K = np.concatenate((self.K, k_new_ext), axis=1)

        self.num_train += 1

    def se_kernel(self, x1, x2):
        """
        The squared exponential kernel function
        """
        x1 = np.squeeze(x1)
        x2 = np.squeeze(x2)
        return (self.sigma_f**2) * np.exp(-1/2 * (x1 - x2).T @ np.linalg.inv(self.Lambda) @ (x1 - x2))

    def update_A_inv_mat(self, k_new):
        B = k_new
        C = k_new.T
        D = np.array([[self.sigma_e**2 + self.sigma_f**2]])
        Q = 1./(D - C @ self.A_inv @ B)  # Just inverting a scalar

        new_A_inv_top_left = self.A_inv + self.A_inv @ B @ Q @ C @ self.A_inv
        new_A_inv_top_right = -self.A_inv @ B @ Q
        new_A_inv_bottom_left = -Q @ C @ self.A_inv
        new_A_inv_bottom_right = Q

        new_A_inv_top = np.concatenate((new_A_inv_top_left, new_A_inv_top_right), axis=1)
        new_A_inv_bottom = np.concatenate((new_A_inv_bottom_left, new_A_inv_bottom_right), axis=1)
        self.A_inv = np.concatenate((new_A_inv_top, new_A_inv_bottom), axis=0)
