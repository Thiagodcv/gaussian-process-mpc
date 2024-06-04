from unittest import TestCase
import numpy as np
import GPy
from src.tools.uncertainty_prop import mean_prop, mean_prop_mc


class TestUncertaintyProp(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mean_prop(self):
        num_train = 100

        # Generate noise
        sigma = 0.5
        err = np.random.normal(loc=0, scale=sigma, size=num_train)

        # True function is f(x) = x_1^2 + x_2^2
        def f(x):
            return x.T @ np.identity(2) @ x

        # Generate 2D Gaussian TRAINING inputs
        u = np.array([0., 1.])
        S = np.array([[1., 0.5],
                      [0.5, 2.]])
        X_train = np.random.multivariate_normal(u, S, num_train)  # This might not be right

        # Generate scalar outputs
        y_train = np.array([f(X_train[j, :]) for j in range(num_train)]) + err

        self.assertEqual(X_train.shape[0], num_train)
        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(y_train.shape[0], num_train)

        # Generate kernel hyperparameters
        Lambda = np.array([[1., 0.],
                           [0., 1.]])

        def gauss_kern(x1, x2):
            return np.exp(-1 / 2 * (x1 - x2).T @ np.linalg.inv(Lambda) @ (x1 - x2))

        # Define covariance matrix. Variance from noise term added to K as per equation (6). Forgetting to do so
        # really messes up the conditioning on K and gives bad values for mean of predictive distribution.
        K = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K[i, j] = gauss_kern(X_train[i, :], X_train[j, :])
        K += (sigma**2) * np.identity(num_train)

        self.assertTrue(np.linalg.norm(K-K.T) < 1e-5)
        self.assertEqual(K.shape, (num_train, num_train))

        analytical_mu = mean_prop(K, Lambda, u, S, X_train, y_train)
        print(analytical_mu)

        mc_mu = mean_prop_mc(K, Lambda, u, S, X_train, y_train)
        print(mc_mu)

        # True if difference is less than 2% of the average mean estimate
        self.assertTrue(np.abs(analytical_mu - mc_mu)/((analytical_mu + mc_mu)/2) < 0.02)
