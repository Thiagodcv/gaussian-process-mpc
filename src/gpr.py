import numpy as np


class KNNGaussianProcessRegression(object):
    """
    A class which implement Gaussian Process Regression using K-Nearest neighbours.
    In this class we use the squared-exponential (SE) covariance function.
    """

    def __init__(self, x_dim):
        # Training data
        self.x_dim = x_dim
        self.y_train = None
        self.X_train = None

        # Covariance matrix
        self.K = None

        # Inverse of covariance matrix plus variance of noise
        self.inv_K_sigma = None

        # Hyperparameters
        self.Lambda = np.identity(self.x_dim)
        self.sigma = 1

    def append_train_data(self, x, y):
        """
        Append (x, y) to preexisting training data.

        Parameters
        ----------
        x: (x_dim) numpy array
        y: scalar
        """
        if self.X_train is None:
            self.X_train = x
            self.y_train = y
            self.K = np.array([[1.]])
            self.inv_K_sigma = 1/(self.K[0][0] + self.sigma**2)
        else:
            self.X_train = np.concatenate((self.X_train, x), axis=0)
            self.y_train = np.concatenate((self.y_train, y))
            # TODO: Update K and inv_K_sigma
            