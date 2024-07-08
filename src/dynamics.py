from src.gpr import GaussianProcessRegression
import numpy as np


class Dynamics(object):

    def __init__(self, state_dim, action_dim, nominal_model):
        """
        Parameters:
        ----------
        state_dim: int
            The dimension of the state space
        action_dim: int
            The dimension of the action space
        nominal_model: function
            The nominal model of the system
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nominal_model = nominal_model
        self.gpr_err = [GaussianProcessRegression(x_dim=self.state_dim + self.action_dim) for _ in range(state_dim)]

    def append_train_data(self, state, action, next_state):
        """
        TODO: Think this method through and write tests.
        Updates bundle of GPR models given a state, action, and next state tuple.

        Parameters:
        ----------
        state: np.array
        action: np.array
        next_state: np.array
        """
        x = np.concatenate((state, action))
        self.gpr_err[0].append_train_data(x, next_state[0])

        if self.state_dim > 1:
            for i in range(1, self.state_dim):
                # requires_grad is False for X_train, so can pass in
                self.gpr_err[i].X_train = self.gpr_err[0].X_train
                self.gpr_err[i].num_train = self.gpr_err[0].num_train

                # Because each GP has its own lambda parameters, can't just pass in K matrices from one to the other
                self.gpr_err[i].build_Ky_inv_mat()
