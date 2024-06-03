from unittest import TestCase
import numpy as np
import GPy


class TestUncertaintyProp(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mean_prop(self):
        num_train = 100

        # Generate noise
        sigma = 0.5
        err = np.random.normal(loc=1, scale=sigma, size=num_train)

        # True function is f(x) = x_1^2 + x_2^2
        def f(x):
            return x.T @ np.identity(2) @ x

        # Generate 2D Gaussian inputs
        mu = np.array([5., 4.])
        S = np.array([[2., 0.5],
                      [0.5, 2.]])
        X_train = np.random.multivariate_normal(mu, S, num_train)

        # Generate scalar outputs
        y_train = np.array([f(X_train[j, :]) for j in range(num_train)]) + err

        self.assertEqual(X_train.shape[0], num_train)
        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(y_train.shape[0], num_train)
