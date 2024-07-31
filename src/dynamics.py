from src.gpr import GaussianProcessRegression
from src.tools.uncertainty_prop import (mean_prop, variance_prop, covariance_prop,
                                        mean_prop_torch, variance_prop_torch, covariance_prop_torch)
import numpy as np
import torch


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
        self.device = self.gpr_err[0].device

    def append_train_data(self, state, action, next_state):
        """
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
        TODO: nominal models aren't taken into account here. Try to see if this can be remedied.
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
            print("Timestep: ", t)
            mean = np.concatenate((state_means[t - 1, :], actions[t - 1, :]))

            # Compute covariance matrix
            covar_top = np.concatenate((state_covars[t - 1, :, :], np.zeros((self.state_dim, self.action_dim))),
                                       axis=1)
            covar_bottom = np.concatenate((np.zeros((self.action_dim, self.state_dim)),
                                           1e-3*np.identity(self.action_dim)), axis=1)
            covar = np.concatenate((covar_top, covar_bottom), axis=0)

            for s_dim in range(self.state_dim):
                print("s_dim: ", s_dim)
                # compute means for each state dimension
                Ky = self.gpr_err[s_dim].Ky.cpu().detach().numpy()
                lambda_mat = np.diag(self.gpr_err[s_dim].get_lambdas())

                state_means[t, s_dim] = mean_prop(Ky, lambda_mat, mean, covar, X_train, y_train.squeeze())[0]

                # compute variances for each state dimension
                state_covars[t, s_dim, s_dim] = variance_prop(Ky, lambda_mat, mean, covar, X_train, y_train.squeeze())

            # Find lower diagonal covariances
            for i in range(1, self.state_dim):
                for j in range(i):  # i not i-1
                    Ky_i = self.gpr_err[i].Ky.cpu().detach().numpy()
                    lambda_mat_i = np.diag(self.gpr_err[i].get_lambdas())

                    Ky_j = self.gpr_err[j].Ky.cpu().detach().numpy()
                    lambda_mat_j = np.diag(self.gpr_err[j].get_lambdas())

                    # compute covariances between different state dimensions
                    state_covars[t, i, j] = covariance_prop(Ky_i, Ky_j,
                                                            lambda_mat_i, lambda_mat_j,
                                                            mean, covar, X_train, y_train.squeeze())

            # Copy values from lower diagonal to upper diagonal
            state_covars[t, :, :] += state_covars[t, :, :].T - np.diag(np.diag(state_covars[t, :, :]))

        return state_means, state_covars

    def forward_propagate_torch(self, horizon, curr_state, actions):
        """
        TODO: nominal models aren't taken into account here. Try to see if this can be remedied.
        TODO: Figure out why CUDA runs out of memory for large num_train.
        TODO: Figure out why covariance_prop_torch is much slower than for variance_prop_torch.
        TODO: (mean/cov/var)_prop_torch all assume sigma_f = 1. Modify for general sigma_f.
        Given `horizon` number of actions, compute the expected states and state covariances.
        Note that because this method only takes actions as arguments (and not states), this method
        corresponds to a shooting method. This is implemented using Torch instead of NumPy.

        Parameters:
        ----------
        horizon: int
        curr_state: torch.tensor with dimensions (state_dim,)
        actions: torch.tensor with dimensions (horizon, action_dim)

        Returns:
        -------
        state_means: list of tensors
        state_covars: list of tensors
        """
        state_means = list()
        state_means.append(curr_state)
        state_covars = list()
        state_covars.append(1e-3 * torch.eye(self.state_dim, device=self.device).type(torch.float64))

        X_train = self.gpr_err[0].X_train

        for t in range(1, horizon + 1):
            # print("Timestep: ", t)
            mean = torch.concatenate((state_means[t - 1], actions[t - 1, :]))
            new_mean = []
            new_var = []

            # Compute covariance matrix
            covar_top = torch.concatenate((state_covars[t - 1],
                                           torch.zeros((self.state_dim, self.action_dim), device=self.device)), dim=1)
            covar_bottom = torch.concatenate((torch.zeros((self.action_dim, self.state_dim), device=self.device),
                                              1e-3 * torch.eye(self.action_dim, device=self.device)), dim=1)
            covar = torch.concatenate((covar_top, covar_bottom), dim=0)

            betas = []
            for s_dim in range(self.state_dim):
                # print("s_dim: ", s_dim)

                # compute means for each state dimension
                Ky_inv = self.gpr_err[s_dim].Ky_inv.detach()
                lambdas = torch.exp(self.gpr_err[s_dim].log_lambdas).detach()
                y_train = self.gpr_err[s_dim].y_train  # Each GPR has different y_train

                mean_elem, mp_dict = mean_prop_torch(Ky_inv, lambdas, mean, covar, X_train, y_train.squeeze())
                new_mean.append(mean_elem)
                betas.append(mp_dict['beta'])

                # compute variances for each state dimension
                var_elem = variance_prop_torch(Ky_inv, lambdas, mean, covar, X_train, new_mean[s_dim], betas[s_dim])
                new_var.append(var_elem)

            # TODO: So far, only paying attention to mean and variance. Implement covariance.
            mean_tensor = torch.stack(new_mean)
            state_means.append(mean_tensor)

            covar_tensor = torch.diag(torch.stack(new_var))
            state_covars.append(covar_tensor)

        return state_means, state_covars
