
import os

import numpy as np
from gymnasium.envs.registration import register

from robot_animation.data_processing import (
    process_raw_robot_data,
    robot_data_to_qpos_qvel,
)

animation_fps = 153

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, "../../robot_models/kuka_iiwa/scene.xml"))
data_path = os.path.abspath(os.path.join(current_dir, "../../data/kuka_2.csv"))

animation_df = process_raw_robot_data(data_path)
target_qpos, _ = robot_data_to_qpos_qvel(animation_df, num_q=7)

target_qvel = np.zeros_like(target_qpos)
target_qvel[1:] = (target_qpos[1:] - target_qpos[:-1]) * animation_fps # TODO: shift this upstream
target_qvel[0] = np.zeros(target_qpos.shape[1])  


register(
    id='RobotAnimationEnv-kuka',
    entry_point='robot_animation.environments.kuka_iiwa4:KukaIIWA4Env',
    max_episode_steps=1000,
    kwargs={
        'model_path': model_path,
        'target_qpos': target_qpos,
        'target_qvel': target_qvel,
        'num_q': 7,
        'base_reward_coeff': 0.3,
        'imitation_reward_coeffs': (0.65, 0.35)
    }
)
