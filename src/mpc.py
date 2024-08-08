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
        self.dynamics = Dynamics(self.state_dim, self.input_dim, nominal_models=None)

        # Torch stuff. Perhaps will put everything in tensor format afterwards.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.Q_tor = torch.tensor(self.Q, device=self.device).type(torch.float64)
        self.R_tor = torch.tensor(self.R, device=self.device).type(torch.float64)

        if self.R_delta is not None:
            self.R_delta_tor = torch.tensor(self.R_delta, device=self.device).type(torch.float64)
        else:
            self.R_delta_tor = None

        # Reference variables to guide states and actions towards
        self.x_ref = torch.zeros(self.state_dim, device=self.device)
        self.u_ref = torch.zeros(self.input_dim, device=self.device)

        # Optimization variables in current iteration of IPOPT
        self.curr_cost = None
        self.curr_grad = None
        self.curr_state = None
        self.backward_taken = False
        self.curr_u = None

        # Save last optimal action trajectory to warm-start optimization in new timestep
        self.last_traj = np.random.standard_normal(size=(self.horizon * self.input_dim,))
        # [0 for _ in range(self.horizon * self.input_dim)]

        # Upper- and lower-bounds on control input
        self.ub = [1e16 for _ in range(self.input_dim)]
        self.lb = [-1e16 for _ in range(self.input_dim)]

        # Set this variable to False once training data isn't empty
        self.train_empty = True

    def set_ub(self, ub):
        """
        Set the upper bound on the control input.

        Parameters:
        ----------
        ub: list of length input_dim
            ub[i] is the upper bound of the ith dimension of the control input.
        """
        assert len(ub) == self.input_dim
        self.ub = ub

    def set_lb(self, lb):
        """
        Set the lower bound on the control input.

        Parameters:
        ----------
        lb: list of length input_dim
            lb[i] is the lower bound of the ith dimension of the control input.
        """
        assert len(lb) == self.input_dim
        self.lb = lb

    def set_xref(self, x_ref):
        """
        Set the set point of the state variable.

        Parameters:
        ----------
        x_ref: numpy array of size (state_dim,)
        """
        assert len(x_ref) == self.state_dim
        self.x_ref = torch.tensor(x_ref, device=self.device).type(torch.float64)

    def set_uref(self, u_ref):
        """
        Set the set point of the input variable.

        Parameters:
        ----------
        u_ref: numpy array of size (input_dim,)
        """
        assert len(u_ref) == self.input_dim
        self.u_ref = torch.tensor(u_ref, device=self.device).type(torch.float64)

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

        Parameters
        ----------
        x: list of length horizon+1 with (state_dim,) torch tensors
            A trajectory of state means; Assume x[0] is the current state
        u: (horizon, input_dim) torch tensor
            A trajectory of inputs
        sig: list of length horizon+1 with (state_dim, state_dim) torch tensor
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
            cost = cost + (1/self.gamma * torch.log(torch.linalg.det(torch.eye(self.state_dim, device=self.device) +
                                                                 self.gamma * self.Q_tor @ sig[i])))
            cost = cost + (x[i] - x_ref) @ torch.linalg.inv(Q_inv + self.gamma * sig[i]) @ (x[i] - x_ref)
            # cost += (x[i, :] - x_ref) @ self.Q_tor @ (x[i, :] - x_ref)  # for testing purposes only

        for j in range(self.horizon):
            cost = cost + (u[j, :] - u_ref) @ self.R_tor @ (u[j, :] - u_ref)

        if self.R_delta_tor is not None:
            last_u = torch.tensor(self.last_traj[0:self.input_dim], device=self.device).type(torch.float64)
            last_u = last_u[None, :]
            u_expanded = torch.concatenate((last_u, u), dim=0)
            delta_u = torch.diff(u_expanded, dim=0)

            for j in range(self.horizon):
                cost = cost + delta_u[j, :] @ self.R_delta_tor @ delta_u[j, :]

        return cost

    def objective(self, x):
        """
        TODO: This method isn't directly tested.
        Callback for calculating the objective given an action trajectory. Used for IPOPT.

        Parameters:
        ----------
        x: (horizon * input_dim,) np.array

        Returns:
        -------
        scalar
        """
        # print("objective")
        # print("curr_x: ", x)
        u = torch.as_tensor(x.reshape(self.horizon, self.input_dim), device=self.device).type(torch.float64)
        u.requires_grad_(True)
        x_init = self.curr_state

        state_means, state_covars = self.dynamics.forward_propagate_torch(self.horizon, x_init, u)
        cost = self.cost_torch(state_means, u, state_covars, self.x_ref, self.u_ref)
        # print("mean states: ", state_means)

        self.curr_cost = cost
        self.curr_u = u
        self.backward_taken = False

        return cost.item()

    def gradient(self, x):
        """
        TODO: This method isn't directly tested.
        Callback for calculating the gradient of the objective w.r.t an action trajectory. Used for IPOPT.

        Parameters:
        ----------
        x: (horizon * input_dim,) np.array

        Returns:
        -------
        scalar
        """
        # print("gradient")
        if self.curr_cost is None:
            self.objective(x)

        if self.backward_taken:
            return self.curr_grad
        else:
            self.curr_cost.backward(retain_graph=True)
            self.backward_taken = True
            self.curr_grad = self.curr_u.grad.cpu().detach().numpy()
            # print("curr_grad: ", self.curr_grad)
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
        TODO: This method isn't directly tested.
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
        # If no training data, just return zero vector for optimal trajectory.
        if self.train_empty:
            if self.dynamics.gpr_err[0].num_train > 0:
                self.train_empty = False
            else:
                return np.zeros((self.horizon, self.input_dim))

        self.curr_state = torch.tensor(curr_state, device=self.device).type(torch.float64)
        x0 = np.zeros(shape=len(self.last_traj))
        # x0 = self.last_traj  Setting x0 to be last optimal trajectory can cause local minima issues

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
        nlp.add_option('accept_every_trial_step', 'yes')  # Disable line search
        # nlp.add_option('limited_memory_update_type', 'bfgs')

        # Trying to loosen convergence constraints to converge earlier... doesn't really seem to help
        # nlp.add_option('max_iter', 10)  # default 3000
        nlp.add_option('tol', 1e-4)  # default 1e-8
        nlp.add_option('acceptable_tol', 1e-4)  # default 1e-6
        nlp.add_option('constr_viol_tol', 1e-4)  # default 1e-8
        nlp.add_option('compl_inf_tol', 1e-4)  # default 1e-8
        nlp.add_option('dual_inf_tol', 1e-4)  # default 1e-8
        nlp.add_option('mu_target', 1e-4)  # default 1e-6
        nlp.add_option('acceptable_iter', 3)  # default 15

        # Hide banner and other output to STDOUT
        nlp.add_option('sb', 'yes')
        nlp.add_option('print_level', 0)

        x, info = nlp.solve(x0)
        # print('optimal solution:', x)

        self.last_traj = x
        return np.reshape(x, (self.horizon, self.input_dim))
