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

    print("Stop here")


if __name__ == '__main__':
    test_experiment()
