import numpy as np
import pandas as pd


def robot_data_to_qpos_qvel(csv_path: str, num_q: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw robot data from animation to qpos and qvel
    """
    df = pd.read_csv(csv_path)
    num_frames = len(df)
    
    qpos = np.zeros((num_frames, num_q))
    qvel = np.zeros((num_frames, num_q))
    
    link_names = [f'LinkN{i}' for i in range(1, num_q + 1)] # 1-indexed to ignore base link
    
    for i, link in enumerate(link_names):
        qpos_col = f'qpos_{link}'
        qvel_col = f'qvel_{link}'
        
        qpos[:, i] = df[qpos_col].values
        qvel[:, i] = df[qvel_col].values
        
    return qpos, qvel
