import argparse
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from robot_animation.env import RobotAnimationEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Train a PPO agent for robot animation')
    parser.add_argument('--env', type=str, default="RobotAnimationEnv-kuka", 
                        help='Environment ID')
    parser.add_argument('--n_envs', type=int, default=1, 
                        help='Number of environments to run in parallel')
    parser.add_argument('--timesteps', type=int, default=100000, 
                        help='Total timesteps to train for')
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(args.env)
    env = make_vec_env(lambda: env, n_envs=args.n_envs)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    return 1


if __name__ == "__main__":
    sys.exit(main())