from src.gpr import GaussianProcessRegression
from src.tools.uncertainty_prop import mean_prop, variance_prop, covariance_prop
import numpy as np


class Dynamics(object):

    def __init__(self, state_dim, action_dim, nominal_models=None):
        """
        Parameters:
        ----------
        state_dim: int
            The dimension of the state space
        action_dim: int
            The dimension of the action space
        nominal_models: list of functions or None
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

        if self.nominal_models is None:
            self.gpr_err = [GaussianProcessRegression(self.state_dim + self.action_dim) for _ in range(state_dim)]
        else:
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
        # If only one observation
        if len(state.shape) == 1:
            x = np.concatenate((state, action))
            for i in range(self.state_dim):
                self.gpr_err[i].append_train_data(x, next_state[i])
        # If multiple observations
        else:
            if len(action.shape) == 1:
                action = action[:, None]
            x = np.concatenate((state, action), axis=1)
            for i in range(self.state_dim):
                self.gpr_err[i].append_train_data(x, next_state[:, i])

    def forward_propagate(self, horizon, curr_state, actions):
        """
        TODO: Write tests
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

        X_train = self.gpr_err[0].X_train.cpu().detach().numpy()
        y_train = self.gpr_err[0].y_train.cpu().detach().numpy()

        for t in range(1, horizon + 1):
            mean = np.concatenate((state_means[t - 1, :], actions[t - 1, :]))

            # Compute covariance matrix
            covar_top = np.concatenate((state_covars[t - 1, :, :], np.zeros((self.state_dim, self.action_dim))),
                                       axis=1)
            covar_bottom = np.concatenate((np.zeros((self.action_dim, self.state_dim)),
                                           1e-3*np.identity(self.action_dim)), axis=1)
            covar = np.concatenate((covar_top, covar_bottom), axis=0)

            for s_dim in range(self.state_dim):
                # compute means for each state dimension
                K = self.gpr_err[s_dim].Kf.cpu().detach().numpy()
                lambda_mat = np.diag(self.gpr_err[s_dim].get_lambdas())

                state_means[t, s_dim] = mean_prop(K, lambda_mat, mean, covar, X_train, y_train.squeeze())[0]

                # compute variances for each state dimension
                state_covars[t, s_dim, s_dim] = variance_prop(K, lambda_mat, mean, covar, X_train, y_train.squeeze())

            # Find lower diagonal covariances
            for i in range(1, self.state_dim):
                for j in range(i - 1):
                    K_i = self.gpr_err[i].Kf.cpu().detach().numpy()
                    lambda_mat_i = np.diag(self.gpr_err[i].get_lambdas())

                    K_j = self.gpr_err[j].Kf.cpu().detach().numpy()
                    lambda_mat_j = np.diag(self.gpr_err[j].get_lambdas())

                    # compute covariances between different state dimensions
                    state_covars[t, i, j] = covariance_prop(K_i, K_j,
                                                            lambda_mat_i, lambda_mat_j,
                                                            mean, covar, X_train, y_train.squeeze())

            # Copy values from lower diagonal to upper diagonal
            state_covars[t, :, :] += state_covars[t, :, :].T - np.diag(np.diag(state_covars[t, :, :]))

        return state_means, state_covars
