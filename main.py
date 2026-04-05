from train_agent import *
import torch
from utils import MetricTracker
from buffer import *
import ale_py

timesteps = 1_000_000
lr = 1e-4
batch_size = 32
target_update_C = 10_000
gamma = 0.99
train_freq = 4

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 300_000

buffer, env = load_buffer(50_000, 200_000, "ALE/Pong-v5")

q_network = DQN(
    env,
    start_epsilon=epsilon_start,
    max_decay=epsilon_end,
    decay_steps=epsilon_decay_steps,
    hidden_filters=[32, 64],
)

target_network = DQN(
    env,
    start_epsilon=epsilon_start,
    max_decay=epsilon_end,
    decay_steps=epsilon_decay_steps,
    hidden_filters=[32, 64],
)

target_network.load_state_dict(q_network.state_dict())

optimizer = torch.optim.Adam(q_network.parameters(), lr=lr)
metrics = MetricTracker()

def main():
    buffer, env = load_buffer(50_000, 200_000, game="ALE/Pong-v5")
    train(
        env,
        "pong",
        q_network,
        target_network,
        optimizer,
        timesteps,
        buffer,
        metrics,
        train_freq,
        batch_size,
        gamma,
        target_update_C,
    )

if __name__ == "__main__":
    main()