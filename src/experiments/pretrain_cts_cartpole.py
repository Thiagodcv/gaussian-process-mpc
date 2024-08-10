import numpy as np
from src.environments.continuous_cartpole import ContinuousCartPoleEnv
from src.mpc import RiskSensitiveMPC
from src.simulator import Simulator


def cts_cartpole_experiment():
    """
    State dimensions are x, x_dot, theta, theta_dot. Sample x in [-2.4, 2.4], xdot in [-5, 5],
    theta in [-pi/2, pi/2], thetadot in [-5, 5], and actions in [-1, 1].
    """

    # Create training data
    num_train = 300
    x = np.random.uniform(-2.4, 2.4, size=num_train)
    xdot = np.random.uniform(-2, 2, size=num_train)
    theta = np.random.uniform(-np.pi/4, np.pi/4, size=num_train)  # Instead of doing [-pi/2, pi/2]
    thetadot = np.random.uniform(-2, 2, size=num_train)
    states = np.concatenate((x[:, None], xdot[:, None], theta[:, None], thetadot[:, None]), axis=1)
    actions = np.random.uniform(-1, 1, size=num_train)[:, None]
    next_states = np.zeros(shape=states.shape)

    env = ContinuousCartPoleEnv(render_mode='human')
    for i in range(num_train):
        x, xdot, theta, thetadot = env.stepPhysics(actions[i], states[i, :])
        next_states[i, :] = np.array([x, xdot.item(), theta, thetadot.item()], dtype=np.float32)

    # Define MPC
    Q = 2 * np.identity(4)
    R = 0.01 * np.array([[1]])
    R_delta = None
    gamma = -1
    horizon = 5
    state_dim = 4
    action_dim = 1
    mpc = RiskSensitiveMPC(gamma, horizon, state_dim, action_dim, Q, R, R_delta)
    mpc.ub = [1]
    mpc.lb = [-1]

    # Set hyperparameters of GPRs & pretrain MPC
    for i in range(state_dim):
        mpc.dynamics.gpr_err[i].set_sigma_n(1e-5)  # Recall method doesn't automatically make Ky get rebuilt
        mpc.dynamics.gpr_err[i].set_lambdas([2., 2., 2., 2., 2.])
    mpc.dynamics.append_train_data(states, actions, next_states)

    sim = Simulator(mpc, env, num_iters=50)
    sim.run()


if __name__ == '__main__':
    cts_cartpole_experiment()
