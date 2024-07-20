from src.environments.adjustable_pendulum import AdjustablePendulumEnv
import numpy as np
from src.mpc import RiskSensitiveMPC
from src.simulator import Simulator


def pendulum_experiment():
    # Define environment
    g = 10
    max_speed = 8
    max_torque = 2
    init_state = {'th_init': 1.1 * np.pi, 'thdot_init': 2}
    seed = None  # Note that if init_state is given, seed is overriden.

    env = AdjustablePendulumEnv(render_mode='human',
                                g=g,
                                max_speed=max_speed,
                                max_torque=max_torque,
                                init_state=init_state)

    # Define MPC
    Q = 2 * np.identity(2)
    R = 2 * np.identity(2)
    R_delta = np.array([[1]])
    gamma = 1
    horizon = 10
    state_dim = 2
    action_dim = 1
    mpc = RiskSensitiveMPC(gamma, horizon, state_dim, action_dim, Q, R, R_delta)

    sim = Simulator(mpc, env, num_iters=500)
    sim.run()


if __name__ == '__main__':
    pendulum_experiment()
