import argparse
import os
import sys
from typing import Callable

import gymnasium as gym
import mediapy as media
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from wandb.sdk.wandb_run import Run
from carbs import CARBS, Param, LogSpace, LinearSpace, CARBSParams, WandbLoggingParams # noqa - not using carbs for now
from wandb_carbs import WandbCarbs, create_sweep # noqa - not using carbs for now

from robot_animation.data_processing import (
    process_raw_robot_data,
    robot_data_to_qpos_qvel,
)

DEFAULT_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/kuka_2.csv"))
MODEL_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))


def main() -> int:
    """
    Main function to train a PPO agent that transfers robot behaviors generated in blender to real robot.
    Target behavior is specified by a CSV file containing the target qpos and qvel. TODO: fix 
    The PPO agent is trained using the Stable Baselines3 library using a custom mujoco environment.
    We use Weights and Biases for experiment tracking and CARBS for hyperparameter optimization.
    """
    try:
        args = parse_args()
        run = None
        wandb_callback = None

        # Define CARBS parameter spaces
        param_spaces = [
            Param(name='learning_rate', space=LogSpace(min=1e-5, max=1e-3), search_center=1e-4),
            Param(name='batch_size', space=LinearSpace(min=8, max=64, is_integer=True), search_center=32),
            Param(name='n_epochs', space=LinearSpace(min=3, max=10, is_integer=True), search_center=6),
            Param(name='n_steps', space=LinearSpace(min=512, max=2048, is_integer=True), search_center=1024)
        ]

        if args.track:
            # Create CARBS sweep
            """
            # carbs stuff (frozen for now)
            sweep_id = create_sweep(
                sweep_name='PPO Robot Animation',
                wandb_entity='aryaman-pandya-99',
                wandb_project='robot-animation',
                carb_params=param_spaces
            )
            print(f"Sweep ID: {sweep_id}")"""
            run, wandb_callback = setup_wandb(args.env, args.n_envs, args.timesteps)
            print(f"Run ID: {run.id}")
            
            # Create plots directory if it doesn't exist
            plots_dir = os.path.join(os.path.dirname(__file__), "../../plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            """
            carbs stuff (frozen for now)
            carbs_params = CARBSParams(
                # better_direction_sign=1,
                is_wandb_logging_enabled=False,
                # wandb_params=WandbLoggingParams(
                #     run_id=run.id,
                #     run_name=run.name,
                #     group_name=run.group,
                #     project_name=run.project,
                #     root_dir=plots_dir
                # )
            )
            carbs = CARBS(config=carbs_params, params=param_spaces)
            wandb_carbs = WandbCarbs(carbs=carbs)
            suggestion = wandb_carbs.suggest()"""
        
        animation_df = process_raw_robot_data(args.csv_path)
        target_qpos, _ = robot_data_to_qpos_qvel(animation_df, num_q=7)

        target_qvel = np.zeros_like(target_qpos)
        target_qvel[1:] = (target_qpos[1:] - target_qpos[:-1]) * args.animation_fps # TODO: shift this upstream
        target_qvel[0] = np.zeros(target_qpos.shape[1])  
        
        target_qpos[:, 3], target_qvel[:, 3] = -target_qpos[:, 3], -target_qvel[:, 3]
        
        if args.multi_proc:
            env = make_vec_env(
                make_env(args.env, target_qpos, target_qvel, args.animation_fps),
                n_envs=args.n_envs,
                vec_env_cls=SubprocVecEnv
            )
        else:
            env = make_vec_env(
                make_env(args.env, target_qpos, target_qvel, args.animation_fps),
                n_envs=args.n_envs,
                vec_env_cls=DummyVecEnv
            )
        # Use CARBS suggestions if tracking
        model_kwargs = {
            "policy": "MlpPolicy",
            "env": env,
            "verbose": 1,
            "device": "cpu",
            "tensorboard_log": f"runs/{run.id}" if run is not None else None,
            "learning_rate": 3e-4,
            "batch_size": 8,
            "n_epochs": 5,
            "n_steps": 1024
        }


        model = PPO(**model_kwargs)
        if args.track:
            model.learn(total_timesteps=args.timesteps, callback=wandb_callback)
        else:
            model.learn(total_timesteps=args.timesteps)
        
        eval_env = gym.make(
            args.env,
            animation_frame_rate=args.animation_fps,
            target_qpos=target_qpos,
            target_qvel=target_qvel,
            num_q=7,
            reset_noise_scale=0.1,
            # render_mode="rgb_array"
        )
        frames = evaluate_policy(model, eval_env, num_episodes=5)
        
        # Save frames as video
        video_path = os.path.join(os.path.dirname(__file__), "../../videos")
        os.makedirs(video_path, exist_ok=True)
        media.write_video(os.path.join(video_path, f"eval_video_{run.id if run else 'no_wandb'}.mp4"), frames, fps=args.animation_fps)
        
        # media.show_video(frames, fps=args.animation_fps)
        model.save("ppo_robot_animation")
        env.close()
        eval_env.close()
        
        if run is not None:
            # Record final performance metrics
            eval_reward = np.mean([evaluate_episode_reward(model, eval_env) for _ in range(5)])
            """
            # carbs stuff (frozen for now)
            wandb_carbs.record_observation(
                objective=eval_reward,
                cost=args.timesteps # Using total timesteps as cost metric
            )
            """
            run.finish()
        
        return 0

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
    parser.add_argument('--animation_fps', type=int, default=153, help='Frame rate of the animation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--track', action='store_true', help='Whether to track with wandb')
    parser.add_argument('--multi_proc', action='store_true', help='Whether to use multiple processes for training')
    
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
            "batch_size": 8,
        },
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    wandb_callback = WandbCallback(
        model_save_path=MODEL_SAVE_PATH,verbose=2, gradient_save_freq=100,model_save_freq=10000
    )
    return run, wandb_callback


def make_env(
    env_id: str,
    target_qpos: np.ndarray,
    target_qvel: np.ndarray, 
    animation_fps: int,
    obs_norm: bool = True,
    gamma: float = 0.99
) -> Callable[[], gym.Env]:
    """
    Create a function that will create and return a fresh instance of the environment.
    Enables parallel environment creation.
    Args:
        env_id: The id of the environment to make
        target_qpos: The target qpos of the environment
        target_qvel: The target qvel of the environment
        animation_fps: The frame rate of the animation
    Returns:
        A function that will create and return a fresh instance of the environment
    """
    def _init():
        env = gym.make(
            env_id,
            animation_frame_rate=animation_fps,
            target_qpos=target_qpos,
            target_qvel=target_qvel,
            num_q=7,
            reset_noise_scale=0.1,
            render_mode="rgb_array"
        )
        env = Monitor(env)
        """
        # TODO: normalize obs and reward
        if obs_norm:
            env = gym.wrappers.NormalizeObservation(env)
            # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

        env = gym.wrappers.NormalizeReward(env, gamma=gamma)"""
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


def evaluate_episode_reward(model: PPO, env: gym.Env) -> float:
    """Helper function to evaluate total episode reward"""
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
    return total_reward


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
