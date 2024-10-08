from unittest import TestCase
import numpy as np
from src.mpc import RiskSensitiveMPC
import torch


class TestRiskSensitiveMPC(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cost(self):
        """
        Ensure method runs without failing and returns the correct output.
        """
        N_c = 1
        state_dim = 2
        input_dim = 2
        x_ref = np.array([0.5, 0.5])
        u_ref = np.array([0.6, 0.6])

        x_traj = np.zeros((N_c+1, state_dim))
        u_traj = np.zeros((N_c, input_dim))
        sig_traj = np.zeros((N_c+1, state_dim, state_dim))

        x_traj[0, :] = np.array([1, 1])
        x_traj[1, :] = np.array([3, 3])
        u_traj[0, :] = np.array([2, 2])
        sig_traj[0, :, :] = np.array([[1, 2],
                                      [3, 4]])
        sig_traj[1, :, :] = np.array([[5, 6],
                                      [7, 8]])
        gamma = 1

        Q = np.array([[2, 0],
                      [0, 2]])
        R = np.array([[1, 1],
                      [1, 1]])

        cost = 0
        cost += 1/gamma * np.log(np.linalg.det(np.identity(state_dim) + gamma * Q @ sig_traj[0, :, :]))
        cost += 1 / gamma * np.log(np.linalg.det(np.identity(state_dim) + gamma * Q @ sig_traj[1, :, :]))
        cost += ((x_traj[0, :] - x_ref).T @ np.linalg.inv(np.linalg.inv(Q) + gamma * sig_traj[0, :, :])
                 @ (x_traj[0, :] - x_ref))
        cost += ((x_traj[1, :] - x_ref).T @ np.linalg.inv(np.linalg.inv(Q) + gamma * sig_traj[1, :, :])
                 @ (x_traj[1, :] - x_ref))
        cost += (u_traj[0, :] - u_ref).T @ R @ (u_traj[0, :] - u_ref)

        mpc = RiskSensitiveMPC(gamma, N_c, state_dim, input_dim, Q, R)
        mpc_cost = mpc.cost(x_traj, u_traj, sig_traj, x_ref, u_ref)

        print(cost)
        print(mpc_cost)
        self.assertTrue(np.linalg.norm(mpc_cost - cost) < 1e-6)

    def test_cost_torch(self):
        """
        Ensure method runs without failing and returns the same output as NumPy method.
        """
        N_c = 1
        state_dim = 2
        input_dim = 2
        x_ref = np.array([0.5, 0.5])
        u_ref = np.array([0.6, 0.6])

        x_traj = np.zeros((N_c + 1, state_dim))
        u_traj = np.zeros((N_c, input_dim))
        sig_traj = np.zeros((N_c + 1, state_dim, state_dim))

        x_traj[0, :] = np.array([1, 1])
        x_traj[1, :] = np.array([3, 3])
        u_traj[0, :] = np.array([2, 2])
        sig_traj[0, :, :] = np.array([[1, 2],
                                      [3, 4]])
        sig_traj[1, :, :] = np.array([[5, 6],
                                      [7, 8]])
        gamma = 1

        Q = np.array([[2, 0],
                      [0, 2]])
        R = np.array([[1, 1],
                      [1, 1]])

        mpc = RiskSensitiveMPC(gamma, N_c, state_dim, input_dim, Q, R)

        # Compute cost with NumPy method
        np_cost = mpc.cost(x_traj, u_traj, sig_traj, x_ref, u_ref)

        # Compute cost with PyTorch method
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x_traj = torch.tensor(x_traj, device=device).type(torch.float64)
        u_traj = torch.tensor(u_traj, device=device).type(torch.float64)
        sig_traj = torch.tensor(sig_traj, device=device).type(torch.float64)
        x_ref = torch.tensor(x_ref, device=device).type(torch.float64)
        u_ref = torch.tensor(u_ref, device=device).type(torch.float64)

        torch_cost = mpc.cost_torch(x_traj, u_traj, sig_traj, x_ref, u_ref)

        print(np_cost)
        print(torch_cost)
        self.assertTrue(np.linalg.norm(np_cost - torch_cost.item()) < 1e-5)

    def test_get_optimal_trajectory(self):
        """
        Ensure method runs without failing and returns the same output as NumPy method.
        """
        horizon = 2
        state_dim = 2
        input_dim = 2
        x_ref = np.array([0.5, 0.5])
        u_ref = np.array([0.6, 0.6])
        gamma = 1

        Q = np.array([[2, 0],
                      [0, 2]])
        R = np.array([[1, 0],
                      [0, 1]])

        mpc = RiskSensitiveMPC(gamma, horizon, state_dim, input_dim, Q, R)

        # Feed some data into dynamics object
        # state = np.array([[0., 1.],
        #                   [1., 2.]])
        # action = np.array([[0., 0.],
        #                    [0., 0.]])
        # next_state = np.array([[1., 2.],
        #                        [2., 3.]])
        state = np.random.standard_normal(size=(2, 2))
        action = np.random.standard_normal(size=(2, 2))
        next_state = np.random.standard_normal(size=(2, 2))
        mpc.dynamics.append_train_data(state, action, next_state)

        curr_state = np.array([0., 0.])
        torch.autograd.set_detect_anomaly(True)
        opt_traj = mpc.get_optimal_trajectory(curr_state)
        self.assertTrue(opt_traj.shape == (horizon, input_dim))

    def test_shape_transform(self):
        """
        Just making sure np.reshape works the way I think it does.
        """
        ls = [1, 2, 3, 4, 1, 2, 3, 4]
        ls_reshaped = np.reshape(ls, newshape=(2, 4))
        ls_unshaped = np.reshape(ls_reshaped, newshape=(8,))

        print(ls)
        print(ls_reshaped)
        print(ls_unshaped)

        self.assertTrue(([[1, 2, 3, 4],
                          [1, 2, 3, 4]] == ls_reshaped).all())
        self.assertTrue((ls == ls_unshaped).all())

    def test_lag_input_variable(self):
        """
        Just figuring out how to lag an input variable to calculate delta_u.
        """
        last_u = torch.tensor([1., 1.])[None, :]
        u = torch.tensor([[2., 2.],
                          [4., 4.],
                          [7., 7.]])
        u = torch.concatenate((last_u, u), dim=0)

        print(torch.diff(u, dim=0))

    def test_cost_torch_with_R_delta(self):
        """
        Ensure method runs without failing and returns the correct output.
        R_delta is used in this test.
        """
        N_c = 2
        state_dim = 2
        input_dim = 2
        x_ref = np.array([0.5, 0.5])
        u_ref = np.array([0.6, 0.6])

        x_traj = np.zeros((N_c+1, state_dim))
        u_traj = np.zeros((N_c, input_dim))
        sig_traj = np.zeros((N_c+1, state_dim, state_dim))

        x_traj[0, :] = np.array([1, 1])
        x_traj[1, :] = np.array([2, 2])
        x_traj[2, :] = np.array([3, 3])
        u_traj[0, :] = np.array([2, 2])
        u_traj[1, :] = np.array([4, 4])
        sig_traj[0, :, :] = np.array([[1, 2],
                                      [3, 4]])
        sig_traj[1, :, :] = np.array([[5, 6],
                                      [7, 8]])
        sig_traj[2, :, :] = np.array([[9, 10],
                                      [11, 12]])
        gamma = 1.1

        Q = np.array([[2, 0],
                      [0, 2]])
        R = np.array([[1, 1],
                      [1, 1]])
        R_delta = np.array([[0.5, 0], [0, 1.5]])

        # Manually compute cost
        cost = 0
        cost += 1 / gamma * np.log(np.linalg.det(np.identity(state_dim) + gamma * Q @ sig_traj[0, :, :]))
        cost += 1 / gamma * np.log(np.linalg.det(np.identity(state_dim) + gamma * Q @ sig_traj[1, :, :]))
        cost += 1 / gamma * np.log(np.linalg.det(np.identity(state_dim) + gamma * Q @ sig_traj[2, :, :]))

        cost += ((x_traj[0, :] - x_ref).T @ np.linalg.inv(np.linalg.inv(Q) + gamma * sig_traj[0, :, :])
                 @ (x_traj[0, :] - x_ref))
        cost += ((x_traj[1, :] - x_ref).T @ np.linalg.inv(np.linalg.inv(Q) + gamma * sig_traj[1, :, :])
                 @ (x_traj[1, :] - x_ref))
        cost += ((x_traj[2, :] - x_ref).T @ np.linalg.inv(np.linalg.inv(Q) + gamma * sig_traj[2, :, :])
                 @ (x_traj[2, :] - x_ref))

        cost += (u_traj[0, :] - u_ref).T @ R @ (u_traj[0, :] - u_ref)
        cost += (u_traj[1, :] - u_ref).T @ R @ (u_traj[1, :] - u_ref)

        cost += u_traj[0, :].T @ R_delta @ u_traj[0, :]
        cost += (u_traj[1, :] - u_traj[0, :]).T @ R_delta @ (u_traj[1, :] - u_traj[0, :])

        mpc = RiskSensitiveMPC(gamma, N_c, state_dim, input_dim, Q, R, R_delta)

        # Compute cost using torch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x_traj = torch.tensor(x_traj, device=device).to(torch.float64)
        x_traj = [x_traj[0, :], x_traj[1, :], x_traj[2, :]]

        u_traj = torch.tensor(u_traj, device=device).to(torch.float64)

        sig_traj = torch.tensor(sig_traj, device=device).to(torch.float64)
        sig_traj = [sig_traj[0, :, :], sig_traj[1, :, :], sig_traj[2, :, :]]

        x_ref = torch.tensor(x_ref, device=device).to(torch.float64)
        u_ref = torch.tensor(u_ref, device=device).to(torch.float64)

        # Now mpc.last_traj is generated randomly, so set to zeros for test to still work
        mpc.last_traj = [0 for _ in range(N_c * input_dim)]
        cost_torch = mpc.cost_torch(x_traj, u_traj, sig_traj, x_ref, u_ref)

        print(cost)
        print(cost_torch)
        self.assertTrue(np.linalg.norm(cost_torch.tolist() - cost) < 1e-6)

    def test_state_cost(self):
        """
        Test RiskSensitiveMPC cost_torch() method when R = delta_R = 0.
        """
        horizon = 5
        x_traj = np.array([5, 4, 3, 2, 1, 0])
        sig_traj = np.array([1/6, 1/7, 1/8, 1/9, 1/10, 1/11])
        cost_np = 0
        for i in range(horizon + 1):
            cost_np += -np.log(1-2*sig_traj[i]) + x_traj[i]**2 / (1/2 - sig_traj[i])
        print(cost_np)

        Q = 2 * np.identity(1)
        R = np.array([[0]])
        R_delta = np.array([[0]])
        gamma = -1  # Negative gamma is risk-averse, positive gamma is risk-seeking
        state_dim = 1
        action_dim = 1
        mpc = RiskSensitiveMPC(gamma, horizon, state_dim, action_dim, Q, R, R_delta)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x_traj = torch.tensor(x_traj, device=device).reshape(horizon+1, 1).type(torch.float64)
        u_traj = torch.zeros(size=(horizon, 1), device=device).type(torch.float64)
        sig_traj = torch.tensor(sig_traj, device=device).reshape(horizon+1, 1, 1).type(torch.float64)
        x_ref = torch.zeros(1, device=device).type(torch.float64)
        u_ref = torch.zeros(1, device=device).type(torch.float64)

        cost = mpc.cost_torch(x_traj, u_traj, sig_traj, x_ref, u_ref)
        print(cost.item())
        self.assertTrue(np.abs(cost_np - cost.item()) < 1e-7)

    def test_mpc_autograd(self):
        """
        Test if MPC autograd is functioning properly. It appears that it is, although when testing gradients
        against finite-difference approximations, you can only set epsilon to be so small before machine
        error begins to destroy the quality of the finite difference approx, and the results begin to diverge.
        epsilon = 1e-2 leads to fd approx that is closest to pytorch gradients. We set -10 <= s <= 10 and -1 <= a <= 1.
        """
        num_train = 1000
        s_min = -10
        s_max = 10
        a_min = -1
        a_max = 1

        def f(s, a):
            return s + a

        state = np.random.uniform(s_min, s_max, num_train)[:, None]
        action = np.random.uniform(a_min, a_max, num_train)[:, None]

        next_state = f(state, action)

        Q = 2 * np.identity(1)
        R = np.array([[1]])
        R_delta = np.array([[0]])
        gamma = 1e-5  # Negative gamma is risk-averse, positive gamma is risk-seeking
        horizon = 3
        state_dim = 1
        action_dim = 1
        mpc = RiskSensitiveMPC(gamma, horizon, state_dim, action_dim, Q, R, R_delta)

        mpc.dynamics.gpr_err[0].set_sigma_n(1e-5)  # Recall method doesn't automatically make Ky get rebuilt
        mpc.dynamics.gpr_err[0].set_lambdas([2., 2.])
        mpc.dynamics.append_train_data(state, action, next_state)
        mpc.set_ub([a_max])
        mpc.set_lb([a_min])
        mpc.set_xref(np.array([0.]))
        mpc.set_uref(np.array([0.]))

        curr_state = np.array([5.])
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mpc.curr_state = torch.tensor(curr_state, device=device).type(torch.float64)
        torch.autograd.set_detect_anomaly(True)

        # It seems like when epsilon is made small, gradients & finite differences diverge due to machine error
        epsilon = 1e-2
        traj = np.array([-1., -1., -1.])  # FD & PyTorch agree for many different trajectories.
        cost = mpc.objective(traj)
        grad = mpc.gradient(traj)

        traj_pert0 = traj.copy()
        traj_pert0[0] += epsilon
        cost_pert0 = mpc.objective(traj_pert0)
        grad_finite_diff_0 = (cost_pert0 - cost)/epsilon

        traj_pert1 = traj.copy()
        traj_pert1[1] += epsilon
        cost_pert1 = mpc.objective(traj_pert1)
        grad_finite_diff_1 = (cost_pert1 - cost)/epsilon

        traj_pert2 = traj.copy()
        traj_pert2[2] += epsilon
        cost_pert2 = mpc.objective(traj_pert2)
        grad_finite_diff_2 = (cost_pert2 - cost)/epsilon

        print("Finite difference gradient: ", [grad_finite_diff_0, grad_finite_diff_1, grad_finite_diff_2])
        print("Autograd gradient: ", grad)
