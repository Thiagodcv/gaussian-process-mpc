import numpy as np
import torch
from src.dynamics import Dynamics
import cyipopt


class RiskSensitiveMPC:
    """
    A class implementation of MPC which uses a risk-sensitive cost.
    """

    def __init__(self, gamma, horizon, state_dim, input_dim, Q, R, R_delta=None):
        """
        Parameters
        ----------
        gamma: float
            Risk sensitivity
        horizon: int
            Number of steps optimized over by the MPC controller
        state_dim: int
            State dimension
        input_dim: int
            Input dimension
        Q: numpy array
            State cost matrix
        R: numpy array
            Input cost matrix
        R_delta: numpy array
            Input rate cost matrix
        """
        self.gamma = gamma
        self.horizon = horizon
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.Q = Q
        self.R = R
        self.R_delta = R_delta
        self.dynamics = Dynamics(self.state_dim, self.input_dim, nominal_models=False)

        # Torch stuff. Perhaps will put everything in tensor format afterwards.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Q_tor = torch.tensor(self.Q, device=self.device).type(torch.float64)
        self.R_tor = torch.tensor(self.R, device=self.device).type(torch.float64)

        if self.R_delta is not None:
            self.R_delta_tor = torch.tensor(self.R_delta, device=self.device).type(torch.float64)
        else:
            self.R_delta_tor = None

        # Reference variables to guide states and actions towards
        self.x_ref = torch.zeros(self.state_dim)  # TODO: Make it so that these are settable
        self.u_ref = torch.zeros(self.input_dim)

        # Optimization variables in current iteration of IPOPT
        self.curr_cost = None
        self.curr_grad = None
        self.curr_state = None

        # Save last optimal action trajectory to warm-start optimization in new timestep
        self.last_traj = None

        # Upper- and lower-bounds on control input
        self.ub = [1e16 for _ in range(self.input_dim)]  # TODO: Make it so that these are settable
        self.lb = [-1e16 for _ in range(self.input_dim)]

    def cost(self, x, u, sig, x_ref, u_ref):
        """
        Computes the risk-sensitive cost.

        TODO: Add input rate cost to the cost returned

        Parameters
        ----------
        x: (horizon+1, state_dim) numpy array
            A trajectory of state means; Assume x[0, :] is the current state
        u: (horizon, input_dim) numpy array
            A trajectory of inputs
        sig: (horizon+1, state_dim, state_dim) numpy array
            A trajectory of state covariance matrices
        x_ref: (state_dim,) numpy array
            The reference (or setpoint) state
        u_ref: (input_dim,) numpy array
            The reference (or setpoint) input

        Returns
        -------
        scalar float
        """

        # Precompute inverse
        Q_inv = np.linalg.inv(self.Q)

        cost = 0
        for i in range(self.horizon + 1):
            cost += (1/self.gamma *
                     np.log(np.linalg.det(np.identity(self.state_dim) + self.gamma * self.Q @ sig[i, :, :])))
            cost += (x[i, :] - x_ref).T @ np.linalg.inv(Q_inv + self.gamma * sig[i, :, :]) @ (x[i, :] - x_ref)

        for j in range(self.horizon):
            cost += (u[j, :] - u_ref).T @ self.R @ (u[j, :] - u_ref)

        return cost

    def cost_torch(self, x, u, sig, x_ref, u_ref):
        """
        Computes the risk-sensitive cost using PyTorch.

        TODO: Add input rate cost to the cost returned

        Parameters
        ----------
        x: (horizon+1, state_dim) torch tensor
            A trajectory of state means; Assume x[0, :] is the current state
        u: (horizon, input_dim) torch tensor
            A trajectory of inputs
        sig: (horizon+1, state_dim, state_dim) torch tensor
            A trajectory of state covariance matrices
        x_ref: (state_dim,) torch tensor
            The reference (or setpoint) state
        u_ref: (input_dim,) torch tensor
            The reference (or setpoint) input

        Returns
        -------
        scalar tensor
        """

        # Precompute inverse
        Q_inv = torch.linalg.inv(self.Q_tor)

        cost = 0
        for i in range(self.horizon + 1):
            cost += (1/self.gamma * torch.log(torch.linalg.det(torch.eye(self.state_dim, device=self.device) +
                                                                 self.gamma * self.Q_tor @ sig[i, :, :])))
            cost += (x[i, :] - x_ref) @ torch.linalg.inv(Q_inv + self.gamma * sig[i, :, :]) @ (x[i, :] - x_ref)

        for j in range(self.horizon):
            cost += (u[j, :] - u_ref) @ self.R_tor @ (u[j, :] - u_ref)

        return cost

    def objective(self, x):
        """
        TODO: Need to test this method.
        Callback for calculating the objective given an action trajectory. Used for IPOPT.

        Parameters:
        ----------
        x: (horizon * input_dim,) np.array

        Returns:
        -------
        scalar
        """
        u = torch.as_tensor(x.reshape(self.horizon, self.input_dim), device=self.device).type(torch.float64)
        u.requires_grad_(True)
        x_init = self.curr_state

        state_means, state_covars = self.dynamics.forward_propagate_torch(self.horizon, x_init, u)
        cost = self.cost_torch(state_means, u, state_covars, self.x_ref, self.u_ref)
        cost.backward()

        self.curr_cost = cost
        self.curr_grad = u.grad.cpu().detach().numpy()

        return cost.item()

    def gradient(self, x):
        """
        TODO: Need to test this method.
        Callback for calculating the gradient of the objective w.r.t an action trajectory. Used for IPOPT.

        Parameters:
        ----------
        x: (horizon * input_dim,) np.array

        Returns:
        -------
        scalar
        """
        if self.curr_cost is None:
            self.objective(x)

        return self.curr_grad

    def constraints(self, x):
        """
        No complex constraints need to be applied to control inputs.
        """
        return 0

    def jacobian(self, x):
        """
        No complex constraints need to be applied to control inputs.
        """
        return np.zeros(x.shape)

    def get_optimal_trajectory(self, curr_state):
        """
        TODO: Need to test this method.
        Use IPOPT to solve for optimal action trajectory given the current state.

        Parameters:
        ----------
        curr_state: (state_dim,) np.array
            The current state

        Return:
        ------
        (horizon, input_dim) np.array
            The optimal action trajectory
        """
        self.curr_state = curr_state

        if self.last_traj is None:
            x0 = [0 for _ in range(self.horizon * self.input_dim)]
        else:
            x0 = self.last_traj

        expanded_lb = self.horizon * self.lb
        expanded_ub = self.horizon * self.ub

        nlp = cyipopt.Problem(
            n=len(x0),
            m=0,
            problem_obj=self,
            lb=expanded_lb,
            ub=expanded_ub,
            cl=[0],
            cu=[0]
        )

        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('tol', 1e-7)

        x, info = nlp.solve(x0)
        print(x)
