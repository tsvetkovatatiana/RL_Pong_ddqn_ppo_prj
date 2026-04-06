import os
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecVideoRecorder

ENV_ID = "ALE/Pong-v5"
MODEL_PATH = "ppo_models/best_model/best_model.zip"
VIDEO_DIR = "ppo_videos"
N_EPISODES = 3
SEED = 123


def main():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=SEED,
        env_kwargs={"render_mode": "rgb_array"},
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    env = VecVideoRecorder(
        env,
        video_folder=VIDEO_DIR,
        record_video_trigger=lambda step: step == 0,
        video_length=10_000,
        name_prefix="pong_ppo_eval",
    )

    model = PPO.load(MODEL_PATH)

    obs = env.reset()
    episode_rewards = []
    current_reward = 0.0
    finished_episodes = 0

    while finished_episodes < N_EPISODES:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        current_reward += float(rewards[0])

        if dones[0]:
            finished_episodes += 1
            episode_rewards.append(current_reward)
            print(f"Episode {finished_episodes}: reward = {current_reward:.2f}")
            current_reward = 0.0

    env.close()

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"Average reward over {N_EPISODES} episodes: {mean_reward:.2f}")
    print(f"Videos saved in: {VIDEO_DIR}")


if __name__ == "__main__":
    main()