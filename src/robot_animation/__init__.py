
import os

from gymnasium.envs.registration import register

from robot_animation.data_processing import (
    process_raw_robot_data,
    robot_data_to_qpos_qvel,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, "../../robot_models/kuka_iiwa/scene.xml"))
data_path = os.path.abspath(os.path.join(current_dir, "../../data/kuka_2.csv"))

animation_df = process_raw_robot_data(data_path)
target_qpos, target_qvel = robot_data_to_qpos_qvel(animation_df, num_q=7)


register(
    id='RobotAnimationEnv-kuka',
    entry_point='robot_animation.env:RobotAnimationEnv',
    max_episode_steps=1000,
    kwargs={
        'model_path': model_path,
        'animation_frame_rate': 153,
        'target_qpos': target_qpos,
        'target_qvel': target_qvel,
        'num_q': 7
    }
)