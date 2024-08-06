import numpy as np


class Simulator(object):
    """
    Class with methods for simulating a GP-MPC setup in the given environment.
    """

    def __init__(self, mpc, env, num_iters=500):
        """
        Parameters:
        ----------
        mpc: RiskSensitiveMPC object
        env: Gym environment
        num_iters: int
            Number of iterations to run the environment for
        """
        self.mpc = mpc
        self.env = env
        self.num_iters = num_iters

    def run(self):
        """
        Simulates running a GP-MPC setup in the given environment.
        """
        obs, info = self.env.reset()
        for t in range(self.num_iters):
            action = self.mpc.get_optimal_trajectory(obs)[0, :]
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            # print("state: ", obs)
            # print("Timestep {}: Reward: {:.2f}".format(t, reward))

            if terminated or truncated:
                break

            self.mpc.dynamics.append_train_data(obs, action, next_obs)
            obs = next_obs
