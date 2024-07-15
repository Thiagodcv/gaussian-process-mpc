from unittest import TestCase
import numpy as np
import torch
import GPy
from src.tools.uncertainty_prop import (mean_prop, mean_prop_mc,
                                        variance_prop, variance_prop_mc,
                                        covariance_prop, covariance_prop_mc,
                                        mean_prop_torch, variance_prop_torch,
                                        covariance_prop_torch)
import time


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
        u = np.array([2., 1.])
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

        analytical_mu, _ = mean_prop(K, Lambda, u, S, X_train, y_train)
        print(analytical_mu)

        mc_mu = mean_prop_mc(K, Lambda, u, S, X_train, y_train)
        print(mc_mu)

        # True if difference is less than 2% of the average mean estimate
        self.assertTrue(np.abs(analytical_mu - mc_mu)/((analytical_mu + mc_mu)/2) < 0.02)

    def test_variance_prop(self):
        num_train = 100

        # Generate noise
        sigma = 0.5
        err = np.random.normal(loc=0, scale=sigma, size=num_train)

        # True function is f(x) = x_1^2 + x_2^2
        def f(x):
            return x.T @ np.identity(2) @ x

        # Generate 2D Gaussian TRAINING inputs
        u = np.array([2., 1.])
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
        K += (sigma ** 2) * np.identity(num_train)

        self.assertTrue(np.linalg.norm(K - K.T) < 1e-5)
        self.assertEqual(K.shape, (num_train, num_train))

        analytical_var = variance_prop(K, Lambda, u, S, X_train, y_train)
        print(analytical_var)

        mc_var = variance_prop_mc(K, Lambda, u, S, X_train, y_train)
        print(mc_var)

        # True if difference is less than 5% of the average mean estimate
        self.assertTrue(np.abs(analytical_var - mc_var) / ((analytical_var + mc_var) / 2) < 0.05)

    def test_cov_prop(self):
        num_train = 100

        # Generate noise
        sigma = 0.5
        err = np.random.normal(loc=0, scale=sigma, size=num_train)

        # True function is f(x) = x_1^2 + x_2^2
        def f(x):
            return x.T @ np.identity(2) @ x

        # Generate 2D Gaussian TRAINING inputs
        u = np.array([2., 1.])
        S = np.array([[1., 0.5],
                      [0.5, 2.]])
        X_train = np.random.multivariate_normal(u, S, num_train)  # This might not be right

        # Generate scalar outputs
        y_train = np.array([f(X_train[j, :]) for j in range(num_train)]) + err

        self.assertEqual(X_train.shape[0], num_train)
        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(y_train.shape[0], num_train)

        # Generate kernel hyperparameters
        Lambda1 = np.array([[1., 0.],
                           [0., 1.]])
        Lambda2 = np.array([[2., 0.],
                            [0., 2.]])
        Lambda1_inv = np.linalg.inv(Lambda1)
        Lambda2_inv = np.linalg.inv(Lambda2)

        def gauss_kern(x1, x2, Lambda_inv):
            return np.exp(-1 / 2 * (x1 - x2).T @ Lambda_inv @ (x1 - x2))

        # Define covariance matrices. Variance from noise term added to K as per equation (6). Forgetting to do so
        # really messes up the conditioning on K and gives bad values for mean of predictive distribution.
        K1 = np.zeros((num_train, num_train))
        K2 = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K1[i, j] = gauss_kern(X_train[i, :], X_train[j, :], Lambda1_inv)
                K2[i, j] = gauss_kern(X_train[i, :], X_train[j, :], Lambda2_inv)
        K1 += (sigma ** 2) * np.identity(num_train)
        K2 += (sigma ** 2) * np.identity(num_train)

        self.assertTrue(np.linalg.norm(K1 - K1.T) < 1e-5)
        self.assertEqual(K1.shape, (num_train, num_train))
        self.assertTrue(np.linalg.norm(K2 - K2.T) < 1e-5)
        self.assertEqual(K2.shape, (num_train, num_train))

        analytical_covar = covariance_prop(K1, K2, Lambda1, Lambda2, u, S, X_train, y_train)
        print(analytical_covar)

        mc_covar = covariance_prop_mc(K1, K2, Lambda1, Lambda2, u, S, X_train, y_train)
        print(mc_covar)

        # True if difference is less than 2% of the average mean estimate
        self.assertTrue(np.abs(analytical_covar - mc_covar) / ((analytical_covar + mc_covar) / 2) < 0.02)

    def test_mean_prop_torch(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_train = 100

        # Generate noise
        sigma = 0.5
        err = np.random.normal(loc=0, scale=sigma, size=num_train)

        # True function is f(x) = x_1^2 + x_2^2
        def f(x):
            return x.T @ np.identity(2) @ x

        # Generate 2D Gaussian TRAINING inputs
        u = np.array([2., 1.])
        S = np.array([[1., 0.5],
                      [0.5, 2.]])
        X_train = np.random.multivariate_normal(u, S, num_train)  # This might not be right

        # Generate scalar outputs
        y_train = np.array([f(X_train[j, :]) for j in range(num_train)]) + err

        self.assertEqual(X_train.shape[0], num_train)
        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(y_train.shape[0], num_train)

        # Generate kernel hyperparameters
        lambdas = np.array([1., 1.])

        def gauss_kern(x1, x2):
            return np.exp(-1 / 2 * (x1 - x2).T @ np.diag(1/lambdas) @ (x1 - x2))

        # Define covariance matrix. Variance from noise term added to K as per equation (6). Forgetting to do so
        # really messes up the conditioning on K and gives bad values for mean of predictive distribution.
        K = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K[i, j] = gauss_kern(X_train[i, :], X_train[j, :])
        Ky = K + (sigma ** 2) * np.identity(num_train)

        self.assertTrue(np.linalg.norm(Ky - Ky.T) < 1e-5)
        self.assertEqual(K.shape, (num_train, num_train))

        np_mu, np_dict = mean_prop(Ky, np.diag(lambdas), u, S, X_train, y_train)

        # Convert tensors to torch
        X_train = torch.tensor(X_train, device=device)
        y_train = torch.tensor(y_train, device=device)
        lambdas = torch.tensor(lambdas, device=device)
        u = torch.tensor(u, device=device)
        S = torch.tensor(S, device=device)
        Ky = torch.tensor(Ky, device=device)
        Ky_inv = torch.linalg.inv(Ky)
        torch_mu, torch_dict = mean_prop_torch(Ky_inv, lambdas, u, S, X_train, y_train)

        self.assertTrue(np.abs(np_mu - torch_mu) < 1e-7)
        self.assertTrue(np.linalg.norm(np_dict['beta'] - torch_dict['beta'].cpu().detach().numpy()) < 1e-5)
        self.assertTrue(np.linalg.norm(np_dict['l'] - torch_dict['l'].cpu().detach().numpy()) < 1e-5)

    def test_variance_prop_torch(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_train = 100

        # Generate noise
        sigma = 0.5
        err = np.random.normal(loc=0, scale=sigma, size=num_train)

        # True function is f(x) = x_1^2 + x_2^2
        def f(x):
            return x.T @ np.identity(2) @ x

        # Generate 2D Gaussian TRAINING inputs
        u = np.array([2., 1.])
        S = np.array([[1., 0.5],
                      [0.5, 2.]])
        X_train = np.random.multivariate_normal(u, S, num_train)  # This might not be right

        # Generate scalar outputs
        y_train = np.array([f(X_train[j, :]) for j in range(num_train)]) + err

        self.assertEqual(X_train.shape[0], num_train)
        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(y_train.shape[0], num_train)

        # Generate kernel hyperparameters
        lambdas = np.array([1., 1.])
        Lambda = np.diag(lambdas)

        def gauss_kern(x1, x2):
            return np.exp(-1 / 2 * (x1 - x2).T @ np.linalg.inv(Lambda) @ (x1 - x2))

        # Define covariance matrix. Variance from noise term added to K as per equation (6). Forgetting to do so
        # really messes up the conditioning on K and gives bad values for mean of predictive distribution.
        K = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K[i, j] = gauss_kern(X_train[i, :], X_train[j, :])
        Ky = K + (sigma ** 2) * np.identity(num_train)

        self.assertTrue(np.linalg.norm(Ky - Ky.T) < 1e-5)
        self.assertEqual(Ky.shape, (num_train, num_train))

        start = time.time()
        np_var = variance_prop(Ky, Lambda, u, S, X_train, y_train)
        end = time.time()

        # Convert tensors to torch
        X_train = torch.tensor(X_train, device=device)
        y_train = torch.tensor(y_train, device=device)
        lambdas = torch.tensor(lambdas, device=device)
        u = torch.tensor(u, device=device)
        S = torch.tensor(S, device=device)
        Ky = torch.tensor(Ky, device=device)
        Ky_inv = torch.linalg.inv(Ky)
        torch_mu, torch_dict = mean_prop_torch(Ky_inv, lambdas, u, S, X_train, y_train)

        torch_start = time.time()
        torch_var = variance_prop_torch(Ky_inv, lambdas, u, S, X_train, mean=torch_mu, beta=torch_dict['beta'])
        torch_end = time.time()

        self.assertTrue(np.abs(np_var - torch_var.item()) < 1e-5)
        print("NumPy version time: ", end-start)
        print("Torch version time: ", torch_end-torch_start)
        print(np_var)
        print(torch_var.item())

    def test_covariance_prop_torch(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_train = 200

        # Generate noise
        sigma = 0.5
        err = np.random.normal(loc=0, scale=sigma, size=num_train)

        # True function is f(x) = x_1^2 + x_2^2
        def f(x):
            return x.T @ np.identity(2) @ x

        # Generate 2D Gaussian TRAINING inputs
        u = np.array([2., 1.])
        S = np.array([[1., 0.5],
                      [0.5, 2.]])
        X_train = np.random.multivariate_normal(u, S, num_train)  # This might not be right

        # Generate scalar outputs
        y_train = np.array([f(X_train[j, :]) for j in range(num_train)]) + err

        self.assertEqual(X_train.shape[0], num_train)
        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(y_train.shape[0], num_train)

        # Generate kernel hyperparameters
        lambdas1 = np.array([1., 1.])
        lambdas2 = np.array([2., 2.])
        Lambda1_inv = np.diag(1/lambdas1)
        Lambda2_inv = np.diag(1/lambdas2)

        def gauss_kern(x1, x2, Lambda_inv):
            return np.exp(-1 / 2 * (x1 - x2).T @ Lambda_inv @ (x1 - x2))

        # Define covariance matrices. Variance from noise term added to K as per equation (6). Forgetting to do so
        # really messes up the conditioning on K and gives bad values for mean of predictive distribution.
        K1 = np.zeros((num_train, num_train))
        K2 = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K1[i, j] = gauss_kern(X_train[i, :], X_train[j, :], Lambda1_inv)
                K2[i, j] = gauss_kern(X_train[i, :], X_train[j, :], Lambda2_inv)
        Ky1 = K1 + (sigma ** 2) * np.identity(num_train)
        Ky2 = K2 + (sigma ** 2) * np.identity(num_train)

        self.assertTrue(np.linalg.norm(K1 - K1.T) < 1e-5)
        self.assertEqual(K1.shape, (num_train, num_train))
        self.assertTrue(np.linalg.norm(K2 - K2.T) < 1e-5)
        self.assertEqual(K2.shape, (num_train, num_train))

        np_start = time.time()
        np_covar = covariance_prop(Ky1, Ky2, np.diag(lambdas1), np.diag(lambdas2), u, S, X_train, y_train)
        np_end = time.time()

        # Convert tensors to torch
        X_train = torch.tensor(X_train, device=device)
        y_train = torch.tensor(y_train, device=device)
        lambdas1 = torch.tensor(lambdas1, device=device)
        lambdas2 = torch.tensor(lambdas2, device=device)
        u = torch.tensor(u, device=device)
        S = torch.tensor(S, device=device)
        Ky1 = torch.tensor(Ky1, device=device)
        Ky2 = torch.tensor(Ky2, device=device)
        Ky1_inv = torch.linalg.inv(Ky1)
        Ky2_inv = torch.linalg.inv(Ky2)
        torch_mu1, torch_dict1 = mean_prop_torch(Ky1_inv, lambdas1, u, S, X_train, y_train)
        torch_mu2, torch_dict2 = mean_prop_torch(Ky2_inv, lambdas2, u, S, X_train, y_train)

        torch_start = time.time()
        torch_covar = covariance_prop_torch(Ky1, Ky2, lambdas1, lambdas2, u, S, X_train, y_train,
                                            torch_mu1, torch_mu2, torch_dict1['beta'], torch_dict2['beta'])
        torch_end = time.time()

        print("NumPy version time: ", np_end - np_start)
        print("Torch version time: ", torch_end - torch_start)
        self.assertTrue(np.abs(np_covar - torch_covar.item()) < 1e-5)

        print(np_covar)
        print(torch_covar.item())
