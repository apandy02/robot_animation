import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from robot_animation.env import RobotAnimationEnv

def main():
    env = gym.make("RobotAnimationEnv-kuka")
    env = make_vec_env(lambda: env, n_envs=1)

    model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=5)

if __name__ == "__main__":
    main()