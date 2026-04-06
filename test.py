import numpy as np
import torch

from dqn import DQN
from eniroment import make_env


def test(game, model, num_eps=2):
    env_test = make_env(game, render_mode="human")

    q_network_trained = DQN(env_test)
    checkpoint = torch.load(model, map_location="cpu")
    q_network_trained.load_state_dict(checkpoint["model_state_dict"])
    q_network_trained.eval()
    q_network_trained.epsilon = 0.05

    rewards_list = []

    for episode in range(num_eps):
        print(f"Episode {episode + 1}/{num_eps}", end="\r", flush=True)
        obs, _ = env_test.reset()
        done = False
        total_reward = 0

        while not done:
            batched_obs = np.expand_dims(obs, axis=0)  # [1, 4, 84, 84]

            action = q_network_trained.epsilon_greedy(
                torch.as_tensor(batched_obs, dtype=torch.float32)
            ).cpu().item()

            next_observation, reward, terminated, truncated, _ = env_test.step(action)
            total_reward += reward
            obs = next_observation
            done = terminated or truncated

        rewards_list.append(total_reward)

    env_test.close()
    print(f"\nAverage episode reward achieved: {np.mean(rewards_list):.2f}")


test("ALE/Pong-v5", "models/pong_dqn_best_85000.pth")