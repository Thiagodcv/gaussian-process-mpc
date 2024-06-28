from unittest import TestCase
import time
import numpy as np
from src.gpr import GaussianProcessRegression
import torch
import scipy.linalg.blas as blas


class TestGaussianProcessRegression(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_matrix_inverse(self):
        # Results: 1000: 0.11s, 2000: 0.53s, 3000: 1.45s
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

    def test_torch_matrix_inverse(self):
        # Results:
        # 1000: 0.39s
        # 2000: 0.087s
        # 3000: 0.17s
        # 4000: 0.31s
        # 5000: 0.47s
        # 6000: 0.66s
        # 7000: 0.89s
        # 8000: 1.12s
        # 9000: 1.55s
        # 10_000: 2.01s
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        avg_time = []
        n_trials = 20
        for i in range(1, 10 + 1):
            print(i)
            size = (i * 1000, i * 1000)
            avg_time.append(0)
            for j in range(n_trials):
                start = time.time()
                A = torch.linalg.inv(torch.normal(mean=0., std=1., size=size, device=device))
                A_np = A.cpu().detach().numpy()
                end = time.time()
                avg_time[i - 1] += end - start
            avg_time[i - 1] = avg_time[i - 1] / n_trials
        print(avg_time)

    def test_pytorch_mat_mult(self):
        # For multiplying two (10_000, 10_000) matrices, cuda+torch on average takes 0.8s (after first multiply)
        # numpy on average takes 8.2s, and blas on average takes 6s. Tiny bit of numerical discrepancy between numpy
        # and blas.
        A = torch.normal(mean=0, std=1, size=(10_000, 10_000), device='cuda')
        B = torch.normal(mean=0, std=1, size=(10_000, 10_000), device='cuda')

        for i in range(5):
            torch.cuda.synchronize()
            a = time.perf_counter()
            C_torch = torch.mm(A, B)
            torch.cuda.synchronize()  # wait for mm to finish
            b = time.perf_counter()
            print('Torch with CUDA {:.02e}s'.format(b - a))

        A_np = A.cpu().detach().numpy()
        B_np = B.cpu().detach().numpy()

        for i in range(3):
            a = time.time()
            C_np = A_np @ B_np
            b = time.time()
            print('Numpy on CPU {:.02e}s'.format(b - a))

        A_blas = np.array(A_np, order='F')
        B_blas = np.array(B_np, order='F')

        for i in range(3):
            a = time.time()
            C_blas = blas.sgemm(alpha=1., a=A_blas, b=B_blas)
            b = time.time()
            print('SciPy blas on CPU {:.02e}s'.format(b - a))

        self.assertTrue(np.linalg.norm(C_np - C_blas) < 1e-5)

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

    def test_update_A_inv_mat(self):
        num_train = 100
        sigma_e = 1
        sigma_f = 1
        X_train = np.random.standard_normal(size=(num_train, 2))
        y_train = np.random.standard_normal(size=(num_train,))
        x = np.random.standard_normal(size=(2,))
        y = np.random.standard_normal(size=(1,))

        def gauss_kern(x1, x2):
            return sigma_f ** 2 * np.exp(-1 / 2 * (x1 - x2).T @ (x1 - x2))

        K = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K[i, j] = gauss_kern(X_train[i, :], X_train[j, :])

        k_new = np.array([gauss_kern(x, X_train[i, :]) for i in range(X_train.shape[0])])
        k_new = np.reshape(k_new, (num_train, 1))

        A = K + (sigma_e ** 2) * np.identity(num_train)
        A_inv = np.linalg.inv(A)
        B = k_new
        C = k_new.T
        D = np.array([[sigma_e ** 2 + sigma_f ** 2]])
        Q = np.linalg.inv(D - C @ A_inv @ B)

        new_K_inv_top_left = A_inv + A_inv @ B @ Q @ C @ A_inv
        new_K_inv_top_right = -A_inv @ B @ Q
        new_K_inv_bottom_left = -Q @ C @ A_inv
        new_K_inv_bottom_right = Q

        new_K_inv_top = np.concatenate((new_K_inv_top_left, new_K_inv_top_right), axis=1)
        new_K_inv_bottom = np.concatenate((new_K_inv_bottom_left, new_K_inv_bottom_right), axis=1)
        new_K_inv = np.concatenate((new_K_inv_top, new_K_inv_bottom), axis=0)

        # Now test to see if GPR leads to same result
        gpr = GaussianProcessRegression(x_dim=2)

        for i in range(num_train):
            gpr.append_train_data(X_train[i, :], y_train[i])
        gpr.append_train_data(x, y)
        gpr_A_inv = gpr.A_inv.cpu().detach().numpy()

        self.assertTrue(np.max(new_K_inv - gpr_A_inv) < 1e-5)

    def test_update_ML_estimate(self):
        """
        Ensure program runs without crashing
        """
        num_train = 100
        X_train = np.random.standard_normal(size=(num_train, 2))
        y_train = np.random.standard_normal(size=(num_train,))

        gpr = GaussianProcessRegression(x_dim=2)
        for i in range(num_train):
            gpr.append_train_data(X_train[i, :], y_train[i])

        gpr.update_ML_estimate()

        print("sigma_e: ", gpr.sigma_e)
        print("sigma_f: ", gpr.sigma_f)
        print("Lambda: ", gpr.Lambda)

    def test_lambda_gradient_calculation(self):
        """
        Test to see if the gradient calculation of lambda parameters I derived is correct.
        NOTE: The K kernel matrix is K_f, not K_y in the following tests.
        """
        num_train = 3
        sigma_f = 1.
        X_train = np.random.standard_normal(size=(num_train, 2))
        lambdas = np.array([1., 2.])
        epsilon = 1e-7

        def gauss_kern(x1, x2, e1, e2):
            ers = np.array([e1, e2])
            Lambda_inv = np.diag(1 / (lambdas + ers))
            return sigma_f ** 2 * np.exp(-1 / 2 * (x1 - x2).T @ Lambda_inv @ (x1 - x2))

        K = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K[i, j] = gauss_kern(X_train[i, :], X_train[j, :], 0, 0)

        A = np.zeros((num_train, num_train, 2))
        for i in range(num_train):
            for j in range(num_train):
                for k in range(2):
                    A[i, j, k] = (1/(2*lambdas[k]**2)) * (X_train[i, k] - X_train[j, k])**2

        dKdL1 = np.multiply(K, A[:, :, 0])
        dKdL2 = np.multiply(K, A[:, :, 1])

        # Estimate matrix derivatives using finite difference
        K_L1_step = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K_L1_step[i, j] = gauss_kern(X_train[i, :], X_train[j, :], epsilon, 0)

        K_L2_step = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K_L2_step[i, j] = gauss_kern(X_train[i, :], X_train[j, :], 0, epsilon)

        dKdL1_finite_diff = (K_L1_step - K) / epsilon
        dKdL2_finite_diff = (K_L2_step - K) / epsilon

        print(np.linalg.norm(dKdL1 - dKdL1_finite_diff))
        print(np.linalg.norm(dKdL2 - dKdL2_finite_diff))

    def test_sigma_gradient_calculation(self):
        """
        Test to see if the gradient calculation of sigma parameters I derived is correct.
        NOTE: The K kernel matrix is K_y in the following tests.
        """
        num_train = 3
        sigma_f = 1.
        sigma_e = 0.5
        X_train = np.random.standard_normal(size=(num_train, 2))
        lambdas = np.array([1., 2.])
        epsilon = 1e-7

        def gauss_kern(x1, x2, e_f):
            Lambda_inv = np.diag(1 / lambdas)
            return (sigma_f + e_f) ** 2 * np.exp(-1 / 2 * (x1 - x2).T @ Lambda_inv @ (x1 - x2))

        K = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K[i, j] = gauss_kern(X_train[i, :], X_train[j, :], 0)

        K_y = K + sigma_e**2 * np.identity(num_train)

        K_y_noise_step = K + (sigma_e + epsilon)**2 * np.identity(num_train)

        # Estimate matrix derivatives using finite difference
        K_y_function_step = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K_y_function_step[i, j] = gauss_kern(X_train[i, :], X_train[j, :], epsilon)
        K_y_function_step += sigma_e**2 * np.identity(num_train)

        dKdsigma_e_finite_diff = (K_y_noise_step - K_y) / epsilon
        dKdsigma_f_finite_diff = (K_y_function_step - K_y) / epsilon

        dKdsigma_e = 2 * sigma_e * np.identity(num_train)
        dKdsigma_f = 2/sigma_f * K

        print(np.linalg.norm(dKdsigma_e_finite_diff - dKdsigma_e))
        print(np.linalg.norm(dKdsigma_f_finite_diff - dKdsigma_f))

    def test_gradient_calculation_scalar(self):
        sigma_f = 1.
        x1 = np.array([5., 6.])
        x2 = np.array([3., 4.])
        lambdas = np.array([1., 2.])
        epsilon = 1e-7

        def gauss_kern(x1, x2, e1, e2):
            eps = np.array([e1, e2])
            Lambda_inv = np.diag(1 / (lambdas + eps))
            return (sigma_f ** 2) * np.exp(-1 / 2 * (x1 - x2).T @ Lambda_inv @ (x1 - x2))

        k = gauss_kern(x1, x2, 0, 0)
        k_l1_step = gauss_kern(x1, x2, epsilon, 0)
        k_l2_step = gauss_kern(x1, x2, 0, epsilon)
        dkdl1_fin_diff = (k_l1_step - k)/epsilon
        dkdl2_fin_diff = (k_l2_step - k)/epsilon

        # Derivative is clearly wrong.
        dkdl1 = k * (1/(2 * lambdas[0]**2)) * (x1[0] - x2[0])**2
        dkdl2 = k * (1/(2 * lambdas[1]**2)) * (x1[1] - x2[1])**2

        print(dkdl1 - dkdl1_fin_diff)
        print(dkdl2 - dkdl2_fin_diff)

    def test_kernel_matrix_gradient(self):
        x_dim = 2
        num_train = 3
        gpr = GaussianProcessRegression(x_dim=x_dim)
        X_train = np.random.standard_normal(size=(num_train, x_dim))
        y_train = np.random.standard_normal(size=(num_train,))

        for i in range(num_train):
            gpr.append_train_data(X_train[i, :], y_train[i])

        grad_dict = gpr.kernel_matrix_gradient()

        # Now compute all gradients using finite difference and compare
        lambdas = gpr.Lambda.cpu().detach().numpy()
        sigma_f = gpr.sigma_f.cpu().detach().numpy()
        sigma_e = gpr.sigma_e.cpu().detach().numpy()
        epsilon = 1e-7

        def gauss_kern(x1, x2, e1, e2, e3):
            ers = np.array([e1, e2])
            Lambda_inv = np.diag(1 / (lambdas + ers))
            return (sigma_f + e3) ** 2 * np.exp(-1 / 2 * (x1 - x2).T @ Lambda_inv @ (x1 - x2))

        K = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K[i, j] = gauss_kern(X_train[i, :], X_train[j, :], 0, 0, 0)
        K_y = K + sigma_e**2 * np.identity(num_train)

        # LAMBDA PARAMETERS
        K_y_L1_step = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K_y_L1_step[i, j] = gauss_kern(X_train[i, :], X_train[j, :], epsilon, 0, 0)
        K_y_L1_step += sigma_e**2 * np.identity(num_train)

        K_y_L2_step = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K_y_L2_step[i, j] = gauss_kern(X_train[i, :], X_train[j, :], 0, epsilon, 0)
        K_y_L2_step += sigma_e**2 * np.identity(num_train)

        dKdL1_finite_diff = (K_y_L1_step - K_y) / epsilon
        dKdL2_finite_diff = (K_y_L2_step - K_y) / epsilon

        self.assertTrue(np.linalg.norm(grad_dict['lambda'][:, :, 0].cpu().detach().numpy() - dKdL1_finite_diff) < 1e-5)
        self.assertTrue(np.linalg.norm(grad_dict['lambda'][:, :, 1].cpu().detach().numpy() - dKdL2_finite_diff) < 1e-5)

        # SIGMA_F PARAMETER
        K_y_sigma_f_step = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K_y_sigma_f_step[i, j] = gauss_kern(X_train[i, :], X_train[j, :], 0, 0, epsilon)
        K_y_sigma_f_step += sigma_e**2 * np.identity(num_train)

        dKdsigma_f_finite_diff = (K_y_sigma_f_step - K_y) / epsilon
        self.assertTrue(np.linalg.norm(grad_dict['sigma_f'].cpu().detach().numpy() - dKdsigma_f_finite_diff) < 1e-5)

        # SIGMA_E PARAMETER
        K_y_sigma_e_step = K + (sigma_e + epsilon)**2 * np.identity(num_train)
        dKdsigma_e_finite_diff = (K_y_sigma_e_step - K_y) / epsilon
        self.assertTrue(np.linalg.norm(grad_dict['sigma_n'].cpu().detach().numpy() - dKdsigma_e_finite_diff) < 1e-5)

    def test_marginal_likelihood_grad(self):
        x_dim = 2
        num_train = 3
        gpr = GaussianProcessRegression(x_dim=x_dim)
        X_train = np.random.standard_normal(size=(num_train, x_dim))
        y_train = np.random.standard_normal(size=(num_train,))

        for i in range(num_train):
            gpr.append_train_data(X_train[i, :], y_train[i])

        grad_dict = gpr.kernel_matrix_gradient()
        dml_dict = gpr.marginal_likelihood_grad(grad_dict)
        print(dml_dict)

        # Now compute all gradients using finite difference and compare
        lambdas = gpr.Lambda.cpu().detach().numpy()
        sigma_f = gpr.sigma_f.cpu().detach().numpy()
        sigma_e = gpr.sigma_e.cpu().detach().numpy()
        epsilon = 1e-10

        def gauss_kern(x1, x2, e1, e2, e3):
            ers = np.array([e1, e2])
            Lambda_inv = np.diag(1 / (lambdas + ers))
            return (sigma_f + e3) ** 2 * np.exp(-1 / 2 * (x1 - x2).T @ Lambda_inv @ (x1 - x2))

        K = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K[i, j] = gauss_kern(X_train[i, :], X_train[j, :], 0, 0, 0)
        K_y = K + sigma_e ** 2 * np.identity(num_train)

        # LAMBDA PARAMETERS
        K_y_L1_step = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K_y_L1_step[i, j] = gauss_kern(X_train[i, :], X_train[j, :], epsilon, 0, 0)
        K_y_L1_step += sigma_e ** 2 * np.identity(num_train)

        K_y_L2_step = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K_y_L2_step[i, j] = gauss_kern(X_train[i, :], X_train[j, :], 0, epsilon, 0)
        K_y_L2_step += sigma_e ** 2 * np.identity(num_train)

        # SIGMA_F PARAMETER
        K_y_sigma_f_step = np.zeros((num_train, num_train))
        for i in range(num_train):
            for j in range(num_train):
                K_y_sigma_f_step[i, j] = gauss_kern(X_train[i, :], X_train[j, :], 0, 0, epsilon)
        K_y_sigma_f_step += sigma_e ** 2 * np.identity(num_train)

        # SIGMA_E PARAMETER
        K_y_sigma_e_step = K + (sigma_e + epsilon) ** 2 * np.identity(num_train)

        ml_K_y = (-1/2 * y_train.T @ np.linalg.inv(K_y) @ y_train -
                  1/2 * np.log(np.linalg.det(K_y)) - num_train/2 * np.log(2*np.pi))
        ml_L1_step = (-1/2 * y_train.T @ np.linalg.inv(K_y_L1_step) @ y_train -
                      1/2 * np.log(np.linalg.det(K_y_L1_step)) - num_train/2 * np.log(2*np.pi))
        ml_L2_step = (-1/2 * y_train.T @ np.linalg.inv(K_y_L2_step) @ y_train -
                      1/2 * np.log(np.linalg.det(K_y_L2_step)) - num_train/2 * np.log(2*np.pi))
        ml_sigma_f_step = (-1/2 * y_train.T @ np.linalg.inv(K_y_sigma_f_step) @ y_train -
                           1/2 * np.log(np.linalg.det(K_y_sigma_f_step)) - num_train/2 * np.log(2*np.pi))
        ml_sigma_e_step = (-1/2 * y_train.T @ np.linalg.inv(K_y_sigma_e_step) @ y_train -
                           1/2 * np.log(np.linalg.det(K_y_sigma_e_step)) - num_train/2 * np.log(2*np.pi))

        L1_finite_diff = (ml_L1_step - ml_K_y)/epsilon
        L2_finite_diff = (ml_L2_step - ml_K_y)/epsilon
        sigma_f_finite_diff = (ml_sigma_f_step - ml_K_y)/epsilon
        sigma_e_finite_diff = (ml_sigma_e_step - ml_K_y) / epsilon

        print(L1_finite_diff)
        print(L2_finite_diff)
        print(sigma_f_finite_diff)
        print(sigma_e_finite_diff)

        dml_dlambda = dml_dict['lambda'].cpu().detach().numpy()
        dml_dsigma_f = dml_dict['sigma_f'].cpu().detach().numpy()
        dml_dsigma_n = dml_dict['sigma_n'].cpu().detach().numpy()
        self.assertTrue(np.linalg.norm(dml_dlambda[0] - L1_finite_diff) < 1e-3)
        self.assertTrue(np.linalg.norm(dml_dlambda[1] - L2_finite_diff) < 1e-3)
        self.assertTrue(np.linalg.norm(dml_dsigma_f - sigma_f_finite_diff) < 1e-3)
        self.assertTrue(np.linalg.norm(dml_dsigma_n - sigma_e_finite_diff) < 1e-3)

    def test_update_hyperparams(self):
        """
        TODO: Major issue, sigma_n becomes negative
        """
        x_dim = 2
        num_train = 100
        gpr = GaussianProcessRegression(x_dim=x_dim)

        def f(x):
            return x.T @ x

        X_train = np.random.standard_normal(size=(num_train, x_dim))
        y_train = np.array([f(X_train[i, :]) for i in range(num_train)])

        for i in range(num_train):
            gpr.append_train_data(X_train[i, :], y_train[i])

        print("Done accumulating data")
        gpr.update_hyperparams()
