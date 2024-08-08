import numpy as np
from src.environments.continuous_cartpole import ContinuousCartPoleEnv


def cts_cartpole_experiment():
    """State dimensions are x, x_dot, theta, theta_dot"""
    env = ContinuousCartPoleEnv(render_mode="human")
    env.reset()

    ep_len = 200
    for t in range(ep_len):
        print(t)
        # env.step(env.action_space.sample())
        env.step(np.array([0.0], dtype=np.float32))
        env.render()

    env.close()


if __name__ == '__main__':
    cts_cartpole_experiment()
