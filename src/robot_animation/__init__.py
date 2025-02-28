
import os

from data_processing import robot_data_to_qpos_qvel
from gymnasium.envs.registration import register

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, "../../robot_models/kuka_iiwa/scene.xml"))
data_path = os.path.abspath(os.path.join(current_dir, "../../data/kuka_formatted.csv"))

target_qpos, target_qvel = robot_data_to_qpos_qvel(data_path, num_q=7)


register(
    id='RobotAnimationEnv-kuka',
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

print(f"target_qpos.shape: {target_qpos.shape}, target_qvel.shape: {target_qvel.shape}")