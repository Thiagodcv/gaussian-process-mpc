import numpy as np


class RiskSensitiveMPC:
    """
    A class implementation of MPC which uses a risk-sensitive cost.
    """

    def __init__(self, gamma, horizon, state_dim, input_dim, Q, R, R_delta):
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

    def cost(self, x, u, sig, x_ref, u_ref):
        """
        Computes the risk-sensitive cost.

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
        for i in range(self.horizon):
            cost += (1/self.gamma *
                     np.log(np.linalg.det(np.identity(self.state_dim) + self.gamma * self.Q @ sig[i, :, :])))
            cost += (x[i, :] - x_ref).T @ np.linalg.inv(Q_inv + self.gamma * sig[i, :, :]) @ (x[i, :] - x_ref)





