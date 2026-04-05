from collections import deque
import numpy as np


class MetricTracker:
    def __init__(self, window_size=100):
        # the size of the history we use to track stats
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.current_episode_reward = 0

    def add_step_reward(self, reward):
        # add received reward to the current reward
        self.current_episode_reward += reward

    def end_episode(self):
        # add reward for episode to history
        self.rewards.append(self.current_episode_reward)
        # reset metrics
        self.current_episode_reward = 0

    @property
    def avg_reward(self):
        return np.mean(self.rewards) if self.rewards else 0
