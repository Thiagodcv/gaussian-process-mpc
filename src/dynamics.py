from src.gpr import GaussianProcessRegression
from src.tools.uncertainty_prop import mean_prop, variance_prop, covariance_prop
import numpy as np


class Dynamics(object):

    def __init__(self, state_dim, action_dim, nominal_models):
        """
        Parameters:
        ----------
        state_dim: int
            The dimension of the state space
        action_dim: int
            The dimension of the action space
        nominal_models: list of functions
            The nominal model of the system; Each function in the list represents a dimension

        Returns:
        -------
        np.array
            State expectations
        np.array
            State covariances
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nominal_models = nominal_models
        self.gpr_err = [GaussianProcessRegression(self.state_dim + self.action_dim,
                                                  nominal_models[i]) for i in range(state_dim)]

    def append_train_data(self, state, action, next_state):
        """
        TODO: Write tests
        Updates bundle of GPR models given a state, action, and next state tuple.

        Parameters:
        ----------
        state: np.array
        action: np.array
        next_state: np.array
        """
        x = np.concatenate((state, action))

        for i in range(self.state_dim):
            self.gpr_err[i].append_train_data(x, next_state[i])

    def forward_propagate(self, horizon, curr_state, actions):
        """
        TODO: Something is wrong. How do actions get incorporated?
        Given `horizon` number of actions, compute the expected states and state covariances.
        Note that because this method only takes actions as arguments (and not states), this method
        corresponds to a shooting method.

        Parameters:
        ----------
        horizon: int
        curr_state: np.array with dimensions (state_dim,)
        actions: np.array with dimensions (horizon, action_dim)
        """
        state_means = np.zeros((horizon+1, self.state_dim))
        state_means[0, :] = curr_state
        state_covars = np.zeros((horizon+1, self.state_dim, self.state_dim))
        state_covars[0, :, :] = 1e-3 * np.identity(self.state_dim)

        X_train = self.gpr_err[0].X_train
        y_train = self.gpr_err[0].y_train

        for t in range(1, horizon + 1):
            mean = np.concatenate(state_means[t - 1, :], actions[t - 1, :])

            # Compute covariance matrix
            covar_top = np.concatenate((state_covars[t - 1, :, :], np.zeros((self.action_dim, self.action_dim))),
                                       axis=1)
            covar_bottom = np.concatenate((np.zeros((self.action_dim, self.action_dim)),
                                           1e-5*np.identity(self.action_dim)), axis=1)
            covar = np.concatenate((covar_top, covar_bottom), axis=0)

            for s_dim in range(self.state_dim):
                # compute means for each state dimension
                K = self.gpr_err[s_dim].Kf
                lambda_mat = np.diag(self.gpr_err[s_dim].get_lambdas())

                state_means[t, s_dim] = mean_prop(K, lambda_mat, mean, covar, X_train, y_train)

                # compute variances for each state dimension
                state_covars[t, s_dim, s_dim] = variance_prop(K, lambda_mat, mean, covar, X_train, y_train)

            # Find lower diagonal covariances
            for i in range(1, self.state_dim):
                for j in range(i - 1):
                    K_i = self.gpr_err[i].Kf
                    lambda_mat_i = np.diag(self.gpr_err[i].get_lambdas())

                    K_j = self.gpr_err[j].Kf
                    lambda_mat_j = np.diag(self.gpr_err[j].get_lambdas())

                    # compute covariances between different state dimensions
                    state_covars[t, i, j] = covariance_prop(K_i, K_j,
                                                            lambda_mat_i, lambda_mat_j, mean, covar, X_train, y_train)

            # Copy values from lower diagonal to upper diagonal
            state_covars[t, :, :] = state_covars[t, :, :].T - np.diag(np.diag(state_covars[t, :, :]))

        return state_means, state_covars
