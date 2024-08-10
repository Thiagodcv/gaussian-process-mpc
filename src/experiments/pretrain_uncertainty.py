from src.environments.adjustable_pendulum import AdjustablePendulumEnv
import numpy as np
from src.mpc import RiskSensitiveMPC
from src.simulator import Simulator
import os
import cProfile
import torch
import matplotlib.pyplot as plt


def uncertainty_experiment():
    """
    Suppose states and actions are both 2D, and [-1 -1] <= a <= [1 1].
    """
    a_min = -1
    a_max = 1

    def f(s, a):
        return s + a

    # Box interval 1
    num_train_1 = 200
    x_min_1 = 3.8
    x_max_1 = 4.2
    y_min_1 = -4.2
    y_max_1 = 0.2
    statex_1 = np.random.uniform(x_min_1, x_max_1, num_train_1)[:, None]
    statey_1 = np.random.uniform(y_min_1, y_max_1, num_train_1)[:, None]
    actionx_1 = np.random.uniform(a_min, a_max, num_train_1)[:, None]
    actiony_1 = np.random.uniform(a_min, a_max, num_train_1)[:, None]
    states_1 = np.concatenate((statex_1, statey_1), axis=1)
    actions_1 = np.concatenate((actionx_1, actiony_1), axis=1)

    # Box interval 2
    num_train_2 = 200
    x_min_2 = -0.2
    x_max_2 = 4.2
    y_min_2 = -0.2
    y_max_2 = 0.2
    statex_2 = np.random.uniform(x_min_2, x_max_2, num_train_2)[:, None]
    statey_2 = np.random.uniform(y_min_2, y_max_2, num_train_2)[:, None]
    actionx_2 = np.random.uniform(a_min, a_max, num_train_2)[:, None]
    actiony_2 = np.random.uniform(a_min, a_max, num_train_2)[:, None]
    states_2 = np.concatenate((statex_2, statey_2), axis=1)
    actions_2 = np.concatenate((actionx_2, actiony_2), axis=1)

    # Box interval 3
    # num_train_3 = 100
    # x_min_3 = -0.2
    # x_max_3 = 0.2
    # y_min_3 = -0.2
    # y_max_3 = 2.8
    # statex_3 = np.random.uniform(x_min_3, x_max_3, num_train_3)[:, None]
    # statey_3 = np.random.uniform(y_min_3, y_max_3, num_train_3)[:, None]
    # actionx_3 = np.random.uniform(a_min, a_max, num_train_3)[:, None]
    # actiony_3 = np.random.uniform(a_min, a_max, num_train_3)[:, None]
    # states_3 = np.concatenate((statex_3, statey_3), axis=1)
    # actions_3 = np.concatenate((actionx_3, actiony_3), axis=1)

    # Generate small number of datapoints everywhere else
    # num_train_else = 10
    # x_min_else = -0.5
    # x_max_else = 1.8
    # y_min_else = -2.2
    # y_max_else = -0.2
    # statex_else = np.random.uniform(x_min_else, x_max_else, num_train_else)[:, None]
    # statey_else = np.random.uniform(y_min_else, y_max_else, num_train_else)[:, None]
    # actionx_else = np.random.uniform(a_min, a_max, num_train_else)[:, None]
    # actiony_else = np.random.uniform(a_min, a_max, num_train_else)[:, None]
    # states_else = np.concatenate((statex_else, statey_else), axis=1)
    # actions_else = np.concatenate((actionx_else, actiony_else), axis=1)

    states = np.concatenate((states_1, states_2), axis=0)
    actions = np.concatenate((actions_1, actions_2), axis=0)
    next_states = f(states, actions)

    Q = 2 * np.identity(2)
    R = np.zeros(shape=(2, 2))
    R_delta = None
    gamma = 1e-5  # Negative gamma is risk-averse, positive gamma is risk-seeking
    horizon = 6
    state_dim = 2
    action_dim = 2
    mpc = RiskSensitiveMPC(gamma, horizon, state_dim, action_dim, Q, R, R_delta)

    for i in range(state_dim):
        mpc.dynamics.gpr_err[i].set_sigma_n(1e-5)  # Recall method doesn't automatically make Ky get rebuilt
        mpc.dynamics.gpr_err[i].set_lambdas([0.5, 0.5, 0.5, 0.5])
        mpc.dynamics.gpr_err[i].set_sigma_f(1.)

    mpc.dynamics.append_train_data(states, actions, next_states)
    mpc.set_ub([a_max, a_max])
    mpc.set_lb([a_min, a_min])
    mpc.set_xref(np.array([0., 0.]))
    mpc.set_uref(np.array([0., 0.]))
    curr_state = np.array([4, -4])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mpc.curr_state = torch.tensor(curr_state, device=device).type(torch.float64)
    mpc.gamma = gamma

    # Compute cost at minimizing trajectory when curr_state = [4, -4]
    opt_traj = mpc.get_optimal_trajectory(curr_state)
    print("Optimal input: ", opt_traj)

    # Get expected trajectory in state space
    u = torch.as_tensor(opt_traj.reshape(horizon, action_dim), device=mpc.device).type(torch.float64)
    state_means, state_covars = mpc.dynamics.forward_propagate_torch(horizon, mpc.curr_state, u)

    # Get actual trajectory in state space
    state_real = np.zeros(shape=(horizon+1, state_dim))
    state_real[0, :] = curr_state
    for i in range(horizon):
        state_real[i+1, :] = f(state_real[i, :], opt_traj[i, :])


    # Plot Trajectory on state space
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    plt.xlim(-1, 5)
    plt.ylim(-5, 1)
    ax.scatter(states[:, 0], states[:, 1])
    ax.scatter(0, 0, color='white', edgecolor='black', marker='*', s=400, linewidths=2)
    ax.scatter(curr_state[0], curr_state[1], color='white', edgecolor='black',
               marker='o', s=300, linewidths=2)
    for i in range(len(state_means)):
        ax.scatter(state_means[i][0].item(), state_means[i][1].item(), color='blue')
        ax.scatter(state_real[i, 0].item(), state_real[i, 1].item(), color='black')
    plt.text(0 - 0.4, 0 + 0.3, 'Set Point', fontsize=15)
    plt.text(4, -4 - 0.4, 'Initial State', fontsize=15)
    plt.show()


if __name__ == '__main__':
    uncertainty_experiment()
