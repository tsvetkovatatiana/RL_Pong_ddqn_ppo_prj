import numpy as np
import torch
from tqdm import tqdm
from environment import make_env


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self._buffer = np.zeros(capacity, dtype=object)
        self._position = 0
        self._size = 0
        self.device = device

    def store(self, experience):
        idx = self._position % self.capacity
        self._buffer[idx] = experience
        self._position += 1
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self._size, size=batch_size)
        batch = self._buffer[indices]

        return (
            self.transform(batch, 0, dtype=torch.float32),  # obs
            self.transform(batch, 1, dtype=torch.int64),    # actions
            self.transform(batch, 2, dtype=torch.float32),  # rewards
            self.transform(batch, 3, dtype=torch.float32),  # next_obs
            self.transform(batch, 4, dtype=torch.bool),     # dones
        )

    def transform(self, batch, index, dtype):
        values = np.array([val[index] for val in batch])

        # Just in case an extra singleton channel appears
        if values.ndim == 5 and values.shape[-1] == 1:
            values = np.squeeze(values, axis=-1)

        return torch.as_tensor(values, dtype=dtype, device=self.device)

    def __len__(self):
        return self._size


def load_buffer(preload, capacity, game, device):
    env = make_env(game)
    buffer = ReplayBuffer(capacity, device)

    obs, _ = env.reset()

    if obs.ndim == 4 and obs.shape[-1] == 1:
        obs = np.squeeze(obs, axis=-1)

    for _ in tqdm(range(preload), desc="Preloading replay buffer"):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if next_obs.ndim == 4 and next_obs.shape[-1] == 1:
            next_obs = np.squeeze(next_obs, axis=-1)

        buffer.store((obs, action, reward, next_obs, done))
        obs = next_obs

        if done:
            obs, _ = env.reset()
            if obs.ndim == 4 and obs.shape[-1] == 1:
                obs = np.squeeze(obs, axis=-1)

    return buffer, env