from src.environments.adjustable_pendulum import AdjustablePendulumEnv
import numpy as np
from src.mpc import RiskSensitiveMPC
from src.simulator import Simulator
import os
import cProfile


def test_experiment():
    """
    Suppose -10 <= s <= 10 and -1 <= a <= 1.
    """
    num_train = 1000
    s_min = -10
    s_max = 10
    a_min = -1
    a_max = 1

    def f(s, a):
        return s + a

    state = np.random.uniform(s_min, s_max, num_train)[:, None]
    action = np.random.uniform(a_min, a_max, num_train)[:, None]

    next_state = f(state, action)

    Q = 2 * np.identity(1)
    R = np.array([[0]])
    R_delta = np.array([[0]])
    gamma = 1e-5  # Negative gamma is risk-averse, positive gamma is risk-seeking
    horizon = 5
    state_dim = 1
    action_dim = 1
    mpc = RiskSensitiveMPC(gamma, horizon, state_dim, action_dim, Q, R, R_delta)

    mpc.dynamics.gpr_err[0].set_sigma_n(1e-5)  # Recall method doesn't automatically make Ky get rebuilt
    mpc.dynamics.gpr_err[0].set_lambdas([2., 2.])
    mpc.dynamics.append_train_data(state, action, next_state)
    mpc.set_ub([a_max])
    mpc.set_lb([a_min])
    mpc.set_xref(np.array([0.]))
    mpc.set_uref(np.array([0.]))

    curr_state = np.array([5.])

    # Compute cost at minimizing trajectory when curr_state = [5]
    # mpc.last_traj = np.array([-1, -1, -1, -1, -1])  # set as initial starting point. Note this is not global min.
    mpc.get_optimal_trajectory(curr_state)
    # opt_u = mpc.last_traj  # Expect to get u = [-1, -1, -1, -1, -1]
    # min_cost = mpc.objective(opt_u)
    # min_grad = mpc.gradient(opt_u)
    # print("Cost of IPOPT trajectory: ", min_cost)
    # print("Gradient at IPOPT trajectory: ", min_grad)

    intuit_traj = np.array([-1, -1, -1, -1, -1])
    intuit_cost = mpc.objective(intuit_traj)
    intuit_grad = mpc.gradient(intuit_traj)
    print("Cost of intuitive trajectory: ", intuit_cost)
    print("Gradient at intuitive trajectory: ", intuit_grad)

    traj_d = np.array([-1, -1, -1, -1, -1.1])
    traj_d_cost = mpc.objective(traj_d)
    traj_d_grad = mpc.gradient(traj_d)
    print("Cost of traj_d trajectory: ", traj_d_cost)
    print("Gradient at traj_d trajectory: ", traj_d_grad)


if __name__ == '__main__':
    test_experiment()
