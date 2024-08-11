import gym.wrappers
import numpy as np


class Simulator(object):
    """
    Class with methods for simulating a GP-MPC setup in the given environment. Comes with recording functionality for
    classic gym environments.
    """

    def __init__(self, mpc, env, num_iters=500, record=False, video_folder=None, name_prefix=None):
        """
        Parameters:
        ----------
        mpc: RiskSensitiveMPC object
        env: Gym environment
        num_iters: int
            Number of iterations to run the environment for
        record: bool
            Set to true to record the episode
        video_folder: None or String
            Path to folder where the video should be saved if record=True
        name_prefix: None or String
            Prefix to give to the video file name if record=True
        """
        self.mpc = mpc
        self.num_iters = num_iters

        self.record = record
        self.video_folder = video_folder
        self.name_prefix = name_prefix
        if record:
            self.env = gym.wrappers.RecordVideo(env=env, video_folder=video_folder, name_prefix=name_prefix)
        else:
            self.env = env

    def run(self):
        """
        Simulates running a GP-MPC setup in the given environment.
        """
        obs, info = self.env.reset()
        if self.record:
            assert self.env.render_mode == 'rgb_array'
            self.env.start_video_recorder()

        for t in range(self.num_iters):
            action = self.mpc.get_optimal_trajectory(obs)[0, :]
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            # print("state: ", obs)
            # print("Timestep {}: Reward: {:.2f}".format(t, reward))

            if terminated or truncated:
                break

            self.mpc.dynamics.append_train_data(obs, action, next_obs)
            obs = next_obs

        if self.record:
            self.env.close_video_recorder()
        self.env.close()
