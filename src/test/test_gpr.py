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
        """
        Test how long it takes to inverse a matrix inverse using NumPy.
        Results: 1000: 0.11s, 2000: 0.53s, 3000: 1.45s
        """
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
        """
        Test how long it takes to inverse a matrix using Torch + GPU.

        Results:
        -------
        1000: 0.39s
        2000: 0.087s
        3000: 0.17s
        4000: 0.31s
        5000: 0.47s
        6000: 0.66s
        7000: 0.89s
        8000: 1.12s
        9000: 1.55s
        10_000: 2.01s
        """
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
        """
        Test how long matrix multiplication takes using Torch + GPU, NumPy, and SciPy.blas.

        For multiplying two (10_000, 10_000) matrices, cuda+torch on average takes 0.8s (after first multiply)
        numpy on average takes 8.2s, and blas on average takes 6s. Tiny bit of numerical discrepancy between numpy
        and blas.
        """

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
        """
        Test to see if inverse formula for 2x2 block matrices holds in the context of Gaussian Process Regression.
        """
        num_train = 2000
        sigma_e = 1
        sigma_f = 1.5
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
        """
        Test to see if inverse formula for 2x2 block matrices holds.
        """
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

    def test_lambda_gradient_calculation(self):
        """
        Test to see if the gradient calculation of lambda parameters I derived is correct.
        NOTE: The K kernel matrix is K_f, not K_y in the following tests.
        """
        num_train = 3
        sigma_f = 1.5
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
                    A[i, j, k] = (1 / (2 * lambdas[k] ** 2)) * (X_train[i, k] - X_train[j, k]) ** 2

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

        self.assertTrue(np.linalg.norm(dKdL1 - dKdL1_finite_diff) < 1e-5)
        self.assertTrue(np.linalg.norm(dKdL2 - dKdL2_finite_diff) < 1e-5)

    def test_sigma_gradient_calculation(self):
        """
        Test to see if the gradient calculation of sigma parameters I derived is correct.
        NOTE: The K kernel matrix is K_y in the following tests.
        """
        num_train = 3
        sigma_f = 1.5
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

        self.assertTrue(np.linalg.norm(dKdsigma_e_finite_diff - dKdsigma_e) < 1e-5)
        self.assertTrue(np.linalg.norm(dKdsigma_f_finite_diff - dKdsigma_f) < 1e-5)

    def test_lambda_gradient_calculation_scalar(self):
        """
        Test to see if gradient calculation of lambda parameters I derived is correct.
        """
        sigma_f = 1.5
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

        dkdl1 = k * (1/(2 * lambdas[0]**2)) * (x1[0] - x2[0])**2
        dkdl2 = k * (1/(2 * lambdas[1]**2)) * (x1[1] - x2[1])**2

        print(dkdl1 - dkdl1_fin_diff)
        print(dkdl2 - dkdl2_fin_diff)

    def test_covariance_computation_time(self):
        """
        Test how long it takes to compute the covariance matrix using NumPy
        vs Torch + GPU.

        When num_train = 2000, for-loop takes 90s, torch takes 3.6s.
        When num_train = 5000, torch takes 3.6s still.
        """
        x_dim = 2
        num_train = 1000
        sigma_e = 1
        sigma_f = 1.5
        lambdas = np.array([1., 0.5])  # Has to have x_dim elements
        X_train = np.random.standard_normal(size=(num_train, x_dim))

        def gauss_kern(x1, x2):
            Lambda_inv = np.diag(1 / lambdas)
            return sigma_f ** 2 * np.exp(-1 / 2 * (x1 - x2).T @ Lambda_inv @ (x1 - x2))

        start = time.time()
        # K = np.zeros((num_train, num_train))
        # for i in range(num_train):
        #     for j in range(num_train):
        #         K[i, j] = gauss_kern(X_train[i, :], X_train[j, :])
        K = np.array([[gauss_kern(X_train[i, :], X_train[j, :]) for i in range(num_train)] for j in range(num_train)])
        end = time.time()

        print("For-loop method: ", end - start)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X_train = torch.tensor(X_train, device=device)
        lambdas = torch.tensor(lambdas, device=device)

        torch_start = time.time()
        X_train_mod = X_train * torch.sqrt(1 / lambdas)
        dist_mat = torch.cdist(X_train_mod, X_train_mod, p=2)
        torch_K = (sigma_f ** 2) * torch.exp(-1 / 2 * torch.square(dist_mat))
        torch_end = time.time()

        print("Torch method: ", torch_end - torch_start)

        self.assertTrue(np.linalg.norm(torch_K.cpu().detach().numpy() - K) < 1e-5)

    def test_A_matrix_computation_time(self):
        """
        Test how long it takes to compute the A matrix found in the kernel_matrix_gradient() method using NumPy
        vs Torch + GPU.
        """
        # For x_dim = 1,...,10, torch takes around 3.6s.
        # For x_dim = 4, num_train = 2000, for-loop takes 26s.
        x_dim = 4
        num_train = 2000
        lambdas = np.array([1., 2., 3., 4.])  # Has to have x_dim elements
        X_train = np.random.standard_normal(size=(num_train, x_dim))

        # Compute using for-loops
        A = np.zeros(shape=(num_train, num_train, x_dim))
        start = time.time()
        for i in range(num_train):
            for j in range(num_train):
                for p in range(x_dim):
                    A[i, j, p] = (1/(2 * lambdas[p]**2)) * (X_train[i, p] - X_train[j, p]) ** 2
        end = time.time()
        print("For-loop method: ", end-start)

        # Compute using Torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X_train = torch.tensor(X_train, device=device)
        lambdas = torch.tensor(lambdas, device=device)

        torch_start = time.time()
        A_torch = torch.zeros(size=(num_train, num_train, x_dim), device=device)

        for p in range(x_dim):
            v = torch.reshape(X_train[:, p], (num_train, 1))
            A_torch[:, :, p] = (1/(2 * lambdas[p]**2)) * torch.square(torch.cdist(v, v, p=2))

        torch_end = time.time()
        print("Torch method: ", torch_end-torch_start)

        # Some numerical error definitely builds up, but they're roughly the same. 1e-3 = 0.001
        self.assertTrue(np.linalg.norm(A_torch.cpu().detach().numpy() - A) < 1e-3)

    def test_covariance_vector(self):
        """
        Test how long it takes to compute a vector of covariances between a new test data and training data.
        With num_train=5000, x_dim=4, for-loop takes 0.11s, torch takes 2.09s. Perhaps just for this one, use loop?
        """
        x_dim = 4
        num_train = 5000
        lambdas = np.array([1., 2., 3., 4.])  # Has to have x_dim elements
        sigma_f = 1.5
        X_train = np.random.standard_normal(size=(num_train, x_dim))
        x_new = np.ones(x_dim)

        # Compute covariance vector k_new using numpy on CPU
        def gauss_kern(x1, x2):
            Lambda_inv = np.diag(1 / lambdas)
            return sigma_f ** 2 * np.exp(-1 / 2 * (x1 - x2).T @ Lambda_inv @ (x1 - x2))

        start = time.time()
        k_new = np.array([gauss_kern(X_train[i, :], x_new) for i in range(num_train)])
        end = time.time()

        print("For-loop method: ", end - start)

        # Compute covariance vector k_new using Torch on GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", device)
        X_train = torch.tensor(X_train, device=device)
        lambdas = torch.tensor(lambdas, device=device)
        x_new = torch.tensor(x_new, device=device)

        torch_start = time.time()
        X_train_mod = X_train * torch.sqrt(1/lambdas)
        x_new_mod = x_new * torch.sqrt(1/lambdas)
        k_new_torch = (sigma_f**2) * torch.exp(-1/2 * torch.sum(torch.square(X_train_mod - x_new_mod), dim=1))
        torch_end =time.time()

        print("Torch method: ", torch_end - torch_start)

        self.assertTrue(np.linalg.norm(k_new_torch.cpu().detach().numpy() - k_new) < 1e-7)

    def test_build_covariance_from_scratch(self):
        """
        Test which is faster: using the 2x2 block inverse formula to update the covariance matrix, or simply
        building the covariance from scratch. PyTorch used in both cases.

        Time to add 1 observation to preexisting covariance matrix: 4.77s.
        Time to recompute preexisting covariance matrix from scratch: 0.39s.
        Both using Torch with num_train = 5000.
        --------------------------------------------------------------------
        Time to add 1 observation to preexisting covariance matrix: 0.1s.
        Time to recompute preexisting covariance matrix from scratch: 0.003s.
        Both using Torch with num_train = 100.
        """
        x_dim = 4
        num_train = 5000
        X_train = np.random.standard_normal(size=(num_train, x_dim))
        y_train = np.random.standard_normal(size=(num_train, 1))
        x_new = np.array([[1., 2., 3., 4.]])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X_train_torch = torch.tensor(X_train, device=device).type(torch.float)
        y_train_torch = torch.tensor(y_train, device=device).type(torch.float)
        x_new_torch = torch.tensor(x_new, device=device).type(torch.float)
        X_train_combined = torch.cat((X_train_torch, x_new_torch), dim=0)

        gpr = GaussianProcessRegression(x_dim=x_dim)
        gpr.X_train = X_train_torch
        gpr.y_train = y_train_torch
        gpr.num_train = num_train
        gpr.build_Ky_inv_mat()

        # Time how long it takes to add 1 datapoint
        add1_list = []
        for i in range(5):
            start_add1 = time.time()
            gpr.append_train_data(x=x_new, y=0)
            end_add1 = time.time()
            add1_list.append(end_add1 - start_add1)

        print("Time to add 1: ", np.mean(add1_list))

        # Time out long it takes to just build from scratch
        scratch_list = []
        for i in range(5):
            start_scratch = time.time()
            gpr.X_train = X_train_combined
            gpr.num_train = num_train + 1
            gpr.build_Ky_inv_mat()
            end_scratch = time.time()
            scratch_list.append(end_scratch - start_scratch)

        print("Time to recompute from scratch: ", np.mean(scratch_list))

    # FUNCTIONS THAT ACTUALLY TEST GPR CLASS DIRECTLY
    # -----------------------------------------------

    def test_build_Ky_inv_mat(self):
        """
        Test to see if build_Ky_inv_mat() method correctly computes the covariance matrix.
        """
        gpr = GaussianProcessRegression(x_dim=2)
        num_train = 100
        sigma_e = torch.exp(gpr.log_sigma_n).item()
        sigma_f = torch.exp(gpr.log_sigma_f).item()
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

        new_Ky_inv_top_left = A_inv + A_inv @ B @ Q @ C @ A_inv
        new_Ky_inv_top_right = -A_inv @ B @ Q
        new_Ky_inv_bottom_left = -Q @ C @ A_inv
        new_Ky_inv_bottom_right = Q

        new_Ky_inv_top = np.concatenate((new_Ky_inv_top_left, new_Ky_inv_top_right), axis=1)
        new_Ky_inv_bottom = np.concatenate((new_Ky_inv_bottom_left, new_Ky_inv_bottom_right), axis=1)
        new_Ky_inv = np.concatenate((new_Ky_inv_top, new_Ky_inv_bottom), axis=0)

        # Now test to see if GPR leads to same result

        for i in range(num_train):
            gpr.append_train_data(X_train[i, :], y_train[i])
        gpr.append_train_data(x, y)
        gpr_Ky_inv = gpr.Ky_inv.cpu().detach().numpy()

        self.assertTrue(np.max(new_Ky_inv - gpr_Ky_inv) < 1e-5)

    def test_kernel_matrix_gradient(self):
        """
        Test to see if the kernel_matrix_gradient() method of the GPR class works correctly.
        """
        x_dim = 2
        num_train = 3
        gpr = GaussianProcessRegression(x_dim=x_dim)
        X_train = np.random.standard_normal(size=(num_train, x_dim))
        y_train = np.random.standard_normal(size=(num_train,))

        for i in range(num_train):
            gpr.append_train_data(X_train[i, :], y_train[i])

        # Test build_A_inv_mat calculation of K using Torch
        K1 = gpr.Kf.cpu().detach().numpy()
        gpr.build_Ky_inv_mat()
        self.assertTrue(np.linalg.norm(K1 - gpr.Kf.cpu().detach().numpy()) < 1e-5)
        grad_dict = gpr.kernel_matrix_gradient()

        # Now compute all gradients using finite difference and compare
        lambdas = torch.exp(gpr.log_lambdas).cpu().detach().numpy()
        sigma_f = torch.exp(gpr.log_sigma_f).cpu().detach().numpy()
        sigma_e = torch.exp(gpr.log_sigma_n).cpu().detach().numpy()
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
        """
        Test to see if the marginal_likelihood_grad() method of the GPR class works correctly.
        """
        x_dim = 2
        num_train = 3
        gpr = GaussianProcessRegression(x_dim=x_dim)
        X_train = np.random.standard_normal(size=(num_train, x_dim))
        y_train = np.random.standard_normal(size=(num_train,))

        for i in range(num_train):
            gpr.append_train_data(X_train[i, :], y_train[i])

        gpr.build_Ky_inv_mat()  # Test build_A_inv_mat()
        grad_dict = gpr.kernel_matrix_gradient()
        dml_dict = gpr.marginal_likelihood_grad(grad_dict)
        print(dml_dict)

        # Now compute all gradients using finite difference and compare
        lambdas = torch.exp(gpr.log_lambdas).cpu().detach().numpy()
        sigma_f = torch.exp(gpr.log_sigma_f).cpu().detach().numpy()
        sigma_e = torch.exp(gpr.log_sigma_n).cpu().detach().numpy()
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

    def test_setters_and_getters(self):
        x_dim = 2
        lambdas = np.array([4.3, 6.5])
        sigma_f = 5.3
        sigma_n = 2.0

        gpr = GaussianProcessRegression(x_dim)
        gpr.set_lambdas(lambdas)
        gpr.set_sigma_f(sigma_f)
        gpr.set_sigma_n(sigma_n)

        self.assertTrue(np.linalg.norm(lambdas - gpr.get_lambdas()) < 1e-5)
        self.assertTrue(np.linalg.norm(sigma_f - gpr.get_sigma_f()) < 1e-5)
        self.assertTrue(np.linalg.norm(sigma_n - gpr.get_sigma_n()) < 1e-5)

    def test_append_train_data(self):
        # Hyperparameters
        lambdas = np.array([1., 2.])
        sigma_f = 1.2
        sigma_n = 1.5

        # Functions to input
        x_dim = 2
        x1 = np.array([1., 1.])
        x234 = np.array([[2., 2.],
                         [3., 3.],
                         [4., 4.]])
        y1 = np.array([1.])
        y234 = np.array([2., 3., 4.])

        def gauss_kern(x1, x2):
            Lambda_inv = np.diag(1 / lambdas)
            return (sigma_f ** 2) * np.exp(-1 / 2 * (x1 - x2).T @ Lambda_inv @ (x1 - x2))

        gpr = GaussianProcessRegression(x_dim=x_dim)

        # Test 1x1 covariance matrix
        gpr.append_train_data(x1, y1)
        Ky1 = gauss_kern(x1, x1) + sigma_n**2

        self.assertTrue(np.linalg.norm(Ky1 - gpr.Ky.cpu().detach().norm()) < 1e-5)

    def test_update_hyperparams(self):
        """
        TODO: Major issue, sigma_n becomes negative. This is WIP for now.
        """
        x_dim = 2
        num_train = 100
        gpr = GaussianProcessRegression(x_dim=x_dim)

        def f(x):
            return x.T @ x

        X_train = np.random.multivariate_normal(mean=np.zeros(x_dim),
                                                cov=1*np.identity(x_dim),
                                                size=num_train)
        y_train = np.array([f(X_train[i, :]) for i in range(num_train)])

        for i in range(num_train):
            print(i)
            gpr.append_train_data(X_train[i, :], y_train[i])

        print("Done accumulating data")
        gpr.update_hyperparams()
