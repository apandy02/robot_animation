from gymnasium.envs.registration import register

import os

import numpy as np

animation_fps = 153

import numpy as np
import pandas as pd


def robot_data_to_qpos_qvel(animation_df: pd.DataFrame, num_q: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw robot data from animation to qpos and qvel

    Args:
        animation_df: dataframe containing the robot data
        num_q: number of qpos/qvel variables

    Returns:
        qpos: numpy array of shape (num_frames, num_q)
        qvel: numpy array of shape (num_frames, num_q)
    """
    num_frames = len(animation_df)
    qpos = np.zeros((num_frames, num_q))
    qvel = np.zeros((num_frames, num_q))
    
    link_names = [f'LinkN{i}' for i in range(1, num_q + 1)] 
    
    for i, link in enumerate(link_names):
        qpos_col = f'qpos_{link}'
        qvel_col = f'qvel_{link}'
        qpos[:, i] = animation_df[qpos_col].values
        qvel[:, i] = animation_df[qvel_col].values
        
    return qpos, qvel

def process_raw_robot_data(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    cleaned_df = clean_robot_data(df)
    link_dfs = process_link_data(cleaned_df)
    merged_df = merge_link_data(link_dfs)
    return merged_df


def clean_robot_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw robot data by filtering out unwanted links and handling special cases.

    Args:
        df: Raw dataframe containing robot data

    Returns:
        Cleaned dataframe with only relevant link data
    """
    filtered_df = df[~df['Link'].str.contains('Axis')]
    links = ["LinkN0", "LinkN1", "LinkN2", "LinkN3", "LinkN4", "LinkN5", "LinkN6", "LinkN7"]
    cleaned_df = filtered_df[filtered_df['Link'].isin(links)]
    return cleaned_df


def process_link_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Process cleaned data into separate dataframes for each link.

    Args:
        df: Cleaned dataframe containing robot data

    Returns:
        Dictionary mapping link names to their respective dataframes
    """
    links = ["LinkN0", "LinkN1", "LinkN2", "LinkN3", "LinkN4", "LinkN5", "LinkN6", "LinkN7"]
    link_dfs = {}

    for link in links:
        link_dfs[link] = df[df['Link'] == link].reset_index(drop=True)
        if link != "LinkN7":
            link_dfs[link]['qpos'] = link_dfs[link]['Y_Rotation']
            link_dfs[link]['qvel'] = link_dfs[link]['Y_Velocity']
        else:
            link_dfs[link]['qpos'] = link_dfs[link]['Z_Rotation']
            link_dfs[link]['qvel'] = link_dfs[link]['Z_Velocity']

    return link_dfs


def merge_link_data(link_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all link data into a single dataframe.

    Args:
        link_dfs: Dictionary of dataframes for each link

    Returns:
        Merged dataframe containing all link data
    """
    links = list(link_dfs.keys())
    merged_df = link_dfs[links[0]][['Frame', 'qpos', 'qvel']].copy()
    merged_df.rename(columns={
        'qpos': f'qpos_{links[0]}',
        'qvel': f'qvel_{links[0]}'
    }, inplace=True)

    for link in links[1:]:
        link_data = link_dfs[link][['Frame', 'qpos', 'qvel']].copy()
        link_data.rename(columns={
            'qpos': f'qpos_{link}',
            'qvel': f'qvel_{link}'
        }, inplace=True)
        merged_df = pd.merge(merged_df, link_data, on='Frame')

    return merged_df

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, "../../robot_models/kuka_iiwa/scene.xml"))
data_path = os.path.abspath(os.path.join(current_dir, "../../data/kuka_2.csv"))

animation_df = process_raw_robot_data(data_path)
target_qpos, target_qvel = robot_data_to_qpos_qvel(animation_df, num_q=7)

target_qvel[1:] = (target_qpos[1:] - target_qpos[:-1]) * animation_fps # TODO: shift this upstream
target_qvel[0] = np.zeros(target_qpos.shape[1])  
target_qpos[:, 3], target_qvel[:, 3] = -target_qpos[:, 3], -target_qvel[:, 3]



register(
    id='RobotAnimationEnv-kuka',
    entry_point='robot_animation.optimized.updated_envs.robot_animation:RobotAnimationEnv',
    max_episode_steps=1000,
    kwargs={
        'animation_frame_rate': animation_fps,
        'target_qpos': target_qpos,
        'target_qvel': target_qvel,
        'num_q': 7,
        'model_path': model_path
    }
)
register(
    id="Ant-v5",
    entry_point="robot_animation.optimized.updated_envs.ant_v5:AntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="Humanoid-v5",
    entry_point="robot_animation.optimized.updated_envs.humanoid_v5:HumanoidEnv",
    max_episode_steps=1000,
)

