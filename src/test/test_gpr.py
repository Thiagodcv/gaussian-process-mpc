from unittest import TestCase
import time
import numpy as np
from numba import njit


class TestGaussianProcessRegression(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_matrix_inverse(self):
        # Results: 1000: 0.077s, 2000: 0.34s, 3000: 0.88s
        avg_time = []
        n_trials = 20
        for i in range(1, 3 + 1):
            print(i)
            size = (i * 1000, i * 1000)
            avg_time.append(0)
            for j in range(n_trials):
                start = time.time()
                np.linalg.inv(np.random.normal(size=size))
                end = time.time()
                avg_time[i - 1] += end - start
            avg_time[i - 1] = avg_time[i - 1] / n_trials
        print(avg_time)

    def test_nn_time(self):
        start = time.time()
        for i in range(100_000):
            a = 5 * 5
            a = 5 * 5
            a = 5 * 5
            if i == 5000:
                print(i)
        end = time.time()
        print(end-start)

    def test_partition_inverse_formula(self):
        num_train = 3000
        sigma_e = 1
        sigma_f = 1
        X_train = np.random.standard_normal(size=(num_train, 2))
        x = np.random.standard_normal(size=(2,))

        def gauss_kern(x1, x2):
            return sigma_f**2 * np.exp(-1 / 2 * (x1 - x2).T @ (x1 - x2))

        K = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K[i, j] = gauss_kern(X_train[i, :], X_train[j, :])

        k_new = np.array([gauss_kern(x, X_train[i, :]) for i in range(X_train.shape[0])])
        k_new = np.reshape(k_new, (num_train, 1))

        A = K + (sigma_e**2) * np.identity(num_train)
        A_inv = np.linalg.inv(A)
        B = k_new
        C = k_new.T
        D = np.array([[sigma_e**2 + sigma_f**2]])

        efficient_start = time.time()
        Q = np.linalg.inv(D - C @ A_inv @ B)

        new_K_inv_top_left = A_inv + A_inv @ B @ Q @ C @ A_inv
        new_K_inv_top_right = -A_inv @ B @ Q
        new_K_inv_bottom_left = -Q @ C @ A_inv
        new_K_inv_bottom_right = Q

        new_K_inv_top = np.concatenate((new_K_inv_top_left, new_K_inv_top_right), axis=1)
        new_K_inv_bottom = np.concatenate((new_K_inv_bottom_left, new_K_inv_bottom_right), axis=1)
        new_K_inv = np.concatenate((new_K_inv_top, new_K_inv_bottom), axis=0)
        efficient_end = time.time()

        # Now using numpy inverse
        ineff_start = time.time()
        new_K = np.concatenate((np.concatenate((A, B), axis=1),
                                np.concatenate((C, D), axis=1)),
                               axis=0)
        new_K_inv_ineff = np.linalg.inv(new_K)
        ineff_end = time.time()

        print("Efficient: ", efficient_end - efficient_start)
        print("Inefficient: ", ineff_end - ineff_start)
        self.assertTrue(np.linalg.norm(new_K_inv - new_K_inv_ineff) < 1e-4)

    def test_partition_inverse_formula_simple(self):
        A = np.array([[1., 0.],
                      [3., 3.]])

        b = np.array([[0.], [0.]])
        c = np.array([[5., 2.]])
        d = np.array([[-1.]])

        Ab = np.concatenate((A, b), axis=1)
        cd = np.concatenate((c, d), axis=1)
        new_A = np.concatenate((Ab, cd), axis=0)

        new_A_inv_ineff = np.linalg.inv(new_A)

        A_inv = np.linalg.inv(A)
        Q_inv = np.linalg.inv(d - c @ A_inv @ b)

        new_A_inv_top_left = A_inv + A_inv @ b @ Q_inv @ c @ A_inv
        new_A_inv_top_right = - A_inv @ b @ Q_inv
        new_A_inv_bottom_left = - Q_inv @ c @ A_inv
        new_A_inv_bottom_right = Q_inv

        new_A_inv_top = np.concatenate((new_A_inv_top_left, new_A_inv_top_right), axis=1)
        new_A_inv_bottom = np.concatenate((new_A_inv_bottom_left, new_A_inv_bottom_right), axis=1)
        new_A_inv = np.concatenate((new_A_inv_top, new_A_inv_bottom), axis=0)

        print(new_A_inv_ineff)
        print(new_A_inv)

        self.assertTrue(np.linalg.norm(new_A_inv_ineff - new_A_inv) < 1e-5)
