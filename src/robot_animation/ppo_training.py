from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import gymnasium as gym

env = gym.make("RobotAnimationEnv-v0")
env = make_vec_env(lambda: env, n_envs=1)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
