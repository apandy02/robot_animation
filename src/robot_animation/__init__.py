
import os

import gymnasium as gym
from data_processing import robot_data_to_qpos_qvel
from gymnasium.envs.registration import register

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../robot_models/kuka_iiwa/scene.xml"))
target_qpos, target_qvel = robot_data_to_qpos_qvel(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/kuka_formatted.csv")))

register(
    id='RobotAnimationEnv-kuka-v0',
    entry_point='robot_animation.env:RobotAnimationEnv',
    max_episode_steps=1000,
    kwargs={
        'model_path': model_path,
        'animation_frame_rate': 30,
        'target_qpos': target_qpos,
        'target_qvel': target_qvel,
        'num_q': 7
    }
)

gym.pprint_registry()