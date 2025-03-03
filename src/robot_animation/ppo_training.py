import argparse
import logging
import os
import sys

import gymnasium as gym
import mediapy as media
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from robot_animation.data_processing import robot_data_to_qpos_qvel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ppo_training")


DEFAULT_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/kuka_formatted2.csv"))


def parse_args():
    """
    Parse command line arguments.
    """
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

def make_env(env_id, target_qpos, target_qvel):
    """
    Create a function that will create and return a fresh instance of the environment.
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
        return env
    return _init

def main():
    try:
        args = parse_args()
        
        target_qpos, target_qvel = robot_data_to_qpos_qvel(
            csv_path=args.csv_path,
            num_q=7
        )
        
        env = make_vec_env(
            make_env(args.env, target_qpos, target_qvel),
            n_envs=args.n_envs
        )

        model = PPO("MlpPolicy", env, batch_size=64, verbose=1, device="cpu")
        model.learn(total_timesteps=args.timesteps, callback=WandbCallback())
        
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
        
        return frames, 0

    except Exception as e:
        print(f"Error: {e}")
        return None, 1


if __name__ == "__main__":
    frames, exit_code = main()
    sys.exit(exit_code)