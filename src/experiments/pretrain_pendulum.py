from src.environments.adjustable_pendulum import AdjustablePendulumEnv
import numpy as np
from src.mpc import RiskSensitiveMPC
from src.simulator import Simulator
import os
import cProfile


def pendulum_experiment():
    num_train = 300
    options = {'g': 10.0,
               'dt': 0.05,
               'm': 1.0,
               'l': 1.0,
               'max_speed': 8.0,
               'max_torque': 5.0}

    th = np.random.uniform(0, np.pi, size=num_train)
    th_dot = np.random.uniform(-options['max_speed'], options['max_speed'], size=num_train)
    actions = np.random.uniform(-options['max_torque'], options['max_torque'], size=num_train)[:, None]
    states = np.concatenate((th[:, None], th_dot[:, None]), axis=1)
    next_states = np.zeros(shape=states.shape)

    for i in range(num_train):
        next_states[i, :] = AdjustablePendulumEnv.step_static(states[i, :], actions[i, :], options)

    # Define environment
    g = options['g']
    max_speed = options['max_speed']
    max_torque = options['max_torque']
    init_state = {'th_init': 0, 'thdot_init': -1}
    seed = None  # Note that if init_state is given, seed is overriden.

    env = AdjustablePendulumEnv(render_mode='human',  # None,
                                g=g,
                                max_speed=max_speed,
                                max_torque=max_torque,
                                init_state=init_state)

    # Define MPC
    Q = 2 * np.identity(2)
    R = 2 * np.array([[1]])
    R_delta = np.array([[1]])
    gamma = -1
    horizon = 10
    state_dim = 2
    action_dim = 1
    mpc = RiskSensitiveMPC(gamma, horizon, state_dim, action_dim, Q, R, R_delta)
    mpc.ub = [options['max_torque']]
    mpc.lb = [-options['max_torque']]

    # Pretrain dynamics of MPC
    for i in range(state_dim):
        mpc.dynamics.gpr_err[i].set_sigma_n(1e-5)  # Recall method doesn't automatically make Ky get rebuilt
        mpc.dynamics.gpr_err[i].set_lambdas([2., 2., 2.])
    mpc.dynamics.append_train_data(states, actions, next_states)

    sim = Simulator(mpc, env, num_iters=10)
    sim.run()

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # To show real time spent on .time()
    # run_str = 'sim.run()'
    # cProfile.runctx(run_str, globals(), locals())


if __name__ == '__main__':
    pendulum_experiment()
