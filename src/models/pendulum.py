import numpy as np
import torch


state_dim = 2
action_dim = 1

m = 1  # mass
l = 1  # length of pendulum
b = 0.5  # coefficient of friction
g = 9.8
delta_t = 0.1


# Nominal models (given to the MPC)
def nom_model_th(x, u):
    """
    Given x_k = [theta_k, omega_k] and u_k return theta_{k+1}.
    """
    if len(x.shape) == 2:
        return x[:, 0] + x[:, 1] * delta_t
    else:
        return x[0] + x[1] * delta_t


def nom_model_om(x, u):
    """
    Given x_k = [theta_k, omega_k] and u_k return omega_{k+1}.
    """
    if len(x.shape) == 2:
        return -g/l * torch.sin(x[:, 0]) * delta_t + x[:, 1] + 1 / (m*l**2) * u.squeeze() * delta_t
    else:
        return -g/l * torch.sin(x[0]) * delta_t + x[1] + 1 / (m*l**2) * u * delta_t


# True model (used in simulation)
def true_model_th(x, u):
    """
    Given x_k = [theta_k, omega_k] and u_k return theta_{k+1}.
    """
    if len(x.shape) == 2:
        return x[:, 0] + x[:, 1] * delta_t
    else:
        return x[0] + x[1] * delta_t


def true_model_om(x, u):
    """
    Given x_k = [theta_k, omega_k] and u_k return omega_{k+1}.
    """
    if len(x.shape) == 2:
        return -b/m * x[:, 1] * delta_t - g/l * np.sin(x[:, 0]) * delta_t + x[:, 1] + 1/(m*l**2) * u.squeeze() * delta_t
    else:
        return -b/m * x[1] * delta_t - g/l * np.sin(x[0]) * delta_t + x[1] + 1 / (m*l**2) * u * delta_t
