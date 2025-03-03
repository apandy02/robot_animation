import argparse
import os
import sys
from typing import Callable

import gymnasium as gym
import mediapy as media
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from wandb.sdk.wandb_run import Run

import wandb
from robot_animation.data_processing import (
    process_raw_robot_data,
    robot_data_to_qpos_qvel,
)

DEFAULT_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/kuka_2.csv"))
MODEL_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))

print(f"DEFAULT_CSV_PATH: {DEFAULT_CSV_PATH}")

def main() -> tuple[list[np.ndarray], int]:
    """
    Main function to train a PPO agent that transfers robot behaviors generated in blender to real robot.
    Target behavior is specified by a CSV file containing the target qpos and qvel.
    The PPO agent is trained using the Stable Baselines3 library using a custom mujoco environment.
    We use Weights and Biases for experiment tracking.
    """
    try:
        args = parse_args()
        
        run, wandb_callback = setup_wandb(args.env, args.n_envs, args.timesteps)
        animation_df = process_raw_robot_data(args.csv_path)
        target_qpos, target_qvel = robot_data_to_qpos_qvel(animation_df, num_q=7)
        env = make_vec_env(make_env(args.env, target_qpos, target_qvel),n_envs=args.n_envs)
        
        model = PPO(
            "MlpPolicy", 
            env, 
            batch_size=64, 
            verbose=1, 
            device="cpu",
            tensorboard_log=f"runs/{run.id}" if run is not None else None
        )
        model.learn(total_timesteps=args.timesteps, callback=wandb_callback)
        
        eval_env = gym.make(
            args.env,
            animation_frame_rate=460,
            target_qpos=target_qpos,
            target_qvel=target_qvel,
            num_q=7,
            reset_noise_scale=0.1,
            render_mode="rgb_array"
        )
        frames = evaluate_policy(model, eval_env, num_episodes=5)
        
        media.show_video(frames, fps=30)
        model.save("ppo_robot_animation")
        env.close()
        eval_env.close()
        
        if run is not None:
            run.finish()
        
        return frames, 0

    except Exception as e:
        print(f"Error: {e}")
        if 'run' in locals() and run is not None:
            run.finish()
        raise e


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train a PPO agent for robot animation')
    parser.add_argument('--env', type=str, default="RobotAnimationEnv-kuka", help='Environment ID')
    parser.add_argument('--n_envs', type=int, default=2, help='Number of environments to run in parallel')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps to train for')
    parser.add_argument(
        '--csv_path',
        type=str,
        default=DEFAULT_CSV_PATH,
        help='Path to the CSV file containing the animation data'
    )
    
    return parser.parse_args()


def setup_wandb(env: str, n_envs: int, timesteps: int) -> tuple[Run, WandbCallback]:
    """
    Setup Weights and Biases for experiment tracking.
    """
    run = wandb.init(
        project="robot-animation",
        config={
            "algorithm": "PPO",
            "env_id": env,
            "n_envs": n_envs,
            "total_timesteps": timesteps,
            "batch_size": 64,
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    wandb_callback = WandbCallback(
        model_save_path=MODEL_SAVE_PATH,verbose=2, gradient_save_freq=100,model_save_freq=10000
    )
    return run, wandb_callback


def make_env(env_id: str, target_qpos: np.ndarray, target_qvel: np.ndarray) -> Callable[[], gym.Env]:
    """
    Create a function that will create and return a fresh instance of the environment.
    Enables parallel environment creation.
    Args:
        env_id: The id of the environment to make
        target_qpos: The target qpos of the environment
        target_qvel: The target qvel of the environment
    Returns:
        A function that will create and return a fresh instance of the environment
    """
    def _init():
        env = gym.make(
            env_id,
            animation_frame_rate=460,
            target_qpos=target_qpos,
            target_qvel=target_qvel,
            num_q=7,
            reset_noise_scale=0.1,
            render_mode="rgb_array"
        )
        env = Monitor(env)
        return env
    return _init


def evaluate_policy(model: PPO, env: gym.Env, num_episodes: int = 1) -> list[np.ndarray]:
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
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_frames = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)
        
        all_frames.extend(episode_frames)
    
    return all_frames


if __name__ == "__main__":
    frames, exit_code = main()
    sys.exit(exit_code)
