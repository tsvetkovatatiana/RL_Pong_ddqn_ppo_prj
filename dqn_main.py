import os
import torch

from train_agent import train
from utils import MetricTracker
from buffer import load_buffer
from dqn import DQN

LOG_DIR = "ppo_logs"
RUN_NAME = "pong_dqn"

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

TIMESTEPS = 6_000_000
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 128
TARGET_UPDATE_C = 25_000
GAMMA = 0.99
TRAIN_FREQ = 4

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 300_000


def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    buffer, env = load_buffer(50_000, 200_000, "ALE/Pong-v5", device=device)

    q_network = DQN(
        environment=env,
        device=device,
        in_channels=4,
        hidden_filters=[32, 64],
        start_epsilon=EPSILON_START,
        max_decay=EPSILON_END,
        decay_steps=EPSILON_DECAY_STEPS,
    )

    target_network = DQN(
        environment=env,
        device=device,
        in_channels=4,
        hidden_filters=[32, 64],
        start_epsilon=EPSILON_START,
        max_decay=EPSILON_END,
        decay_steps=EPSILON_DECAY_STEPS,
    )

    target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
    metrics = MetricTracker()

    try:
        train(
            environment=env,
            name=RUN_NAME,
            q_net=q_network,
            target_net=target_network,
            optimizer=optimizer,
            timesteps=TIMESTEPS,
            replay=buffer,
            metric=metrics,
            train_freq=TRAIN_FREQ,
            batch=BATCH_SIZE,
            g=GAMMA,
            C=TARGET_UPDATE_C,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()