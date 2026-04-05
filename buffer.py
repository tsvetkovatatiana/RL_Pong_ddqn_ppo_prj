import numpy as np
import torch
from tqdm import tqdm
from environment import make_env


device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("using device", device)

class ReplayBuffer:
    def __init__(self, capacity, devc):
        self.capacity = capacity
        self._buffer = np.zeros(capacity, dtype=object)
        self._position = 0
        self._size = 0
        self.device = devc

    def store(self, experience):
        idx = self._position % self.capacity
        self._buffer[idx] = experience
        self._position += 1
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self._size, size=batch_size)
        batch = self._buffer[indices]

        return (
            self.transform(batch, 0, dtype=torch.float32),
            self.transform(batch, 1, dtype=torch.int64),
            self.transform(batch, 2, dtype=torch.float32),
            self.transform(batch, 3, dtype=torch.float32),
            self.transform(batch, 4, dtype=torch.bool),
        )


    def transform(self, batch, index, dtype):
        values = np.array([val[index] for val in batch])
        if values.ndim == 5 and values.shape[-1] == 1:
            values = np.squeeze(values, axis=-1)

        return torch.as_tensor(values, dtype=dtype, device=self.device)


    def __len__(self):
        return self._size


def load_buffer(preload, capacity, game, *, devc=device):
    env = make_env(game)
    buffer = ReplayBuffer(capacity, devc)

    obs, _ = env.reset()

    for _ in tqdm(range(preload)):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.store((obs, action, reward, next_state, done))
        obs = next_state

        if done:
            obs, _ = env.reset()

    return buffer, env