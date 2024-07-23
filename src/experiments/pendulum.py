from src.environments.adjustable_pendulum import AdjustablePendulumEnv
import numpy as np
from src.mpc import RiskSensitiveMPC
from src.simulator import Simulator
import os
import cProfile


def pendulum_experiment():
    # Define environment
    g = 10
    max_speed = 8
    max_torque = 2
    init_state = {'th_init': 1.1 * np.pi, 'thdot_init': 2}
    seed = None  # Note that if init_state is given, seed is overriden.

    env = AdjustablePendulumEnv(render_mode=None,  # 'human',
                                g=g,
                                max_speed=max_speed,
                                max_torque=max_torque,
                                init_state=init_state)

    # Define MPC
    Q = 2 * np.identity(2)
    R = 2 * np.array([[1]])
    R_delta = np.array([[1]])
    gamma = 1
    horizon = 10
    state_dim = 2
    action_dim = 1
    mpc = RiskSensitiveMPC(gamma, horizon, state_dim, action_dim, Q, R, R_delta)

    sim = Simulator(mpc, env, num_iters=10)
    sim.run()

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # To show real time spent on .time()
    # run_str = 'sim.run()'
    # cProfile.runctx(run_str, globals(), locals())


if __name__ == '__main__':
    pendulum_experiment()
