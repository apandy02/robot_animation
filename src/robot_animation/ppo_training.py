import argparse
import sys
import numpy as np
import mediapy as media

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from robot_animation.env import RobotAnimationEnv  # noqa: F401


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a PPO agent for robot animation')
    
    parser.add_argument(
        '--env', type=str, default="RobotAnimationEnv-kuka", help='Environment ID'
    )
    parser.add_argument(
        '--n_envs', type=int, default=1, help='Number of environments to run in parallel'
    )
    parser.add_argument(
        '--timesteps', type=int, default=100000, help='Total timesteps to train for'
    )
    
    return parser.parse_args()

def evaluate_policy(model, env, num_episodes=1):
    """
    Evaluate the trained policy and return frames for visualization.
    
    Args:
        model: Trained PPO model
        env: Environment to evaluate in
        num_episodes: Number of episodes to run
        
    Returns:
        List of frames from all episodes
    """
    all_frames = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_frames = []
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Render and store frame
            frame = env.render()
            if frame is not None:  # Some environments might return None
                episode_frames.append(frame)
        
        all_frames.extend(episode_frames)
    
    return all_frames

def main():
    try:
        args = parse_args()
        env = gym.make(args.env)
        env = make_vec_env(lambda: env, n_envs=args.n_envs)

        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=args.timesteps)
        
        # Evaluate the model and collect frames
        eval_env = gym.make(args.env)  # Create a separate environment for evaluation
        frames = evaluate_policy(model, eval_env, num_episodes=5)
        media.show_video(frames, fps=30)
        
        # Save the model
        model.save("ppo_robot_animation")

        env.close()
        eval_env.close()
        
        # Here you can return the frames to be used with show_video
        # For example, you could save them to a file or pass them to another function
        return frames, 0

    except Exception as e:
        print(f"Error: {e}")
        return None, 1


if __name__ == "__main__":
    frames, exit_code = main()
    sys.exit(exit_code)