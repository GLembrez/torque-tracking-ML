import KinovaRL
import gymnasium as gym
from stable_baselines3 import PPO
import os
import numpy as np

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.policies import MlpPolicy


def evaluate(model, num_episodes=2, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    vec_env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = vec_env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = vec_env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
    return mean_episode_reward


save_path = "/home/gabinlembrez/trained_nets/RL"
os.makedirs(save_path, exist_ok=True)

env = gym.make("Kinova_RL/KinovaEnv", path="/home/gabinlembrez/GitHub/torque-tracking-ML/python_controller/xml/gen3_7dof_mujoco.xml")
model = PPO(MlpPolicy, env)
episode_length = 2500
n_episode = 100

print(model.policy)

reward = evaluate(model,num_episodes=10)

model.learn(total_timesteps=n_episode*episode_length, log_interval=5_000, progress_bar=True)

reward = evaluate(model, num_episodes=10)

model.save(f"{save_path}/trained")



