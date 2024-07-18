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
        