import numpy as np
import pandas as pd

def robot_data_to_qpos_qvel(csv_path):
    df = pd.read_csv(csv_path)
    num_frames = len(df)
    
    qpos = np.zeros((num_frames, 8))
    qvel = np.zeros((num_frames, 8))
    
    link_names = ['LinkN0', 'LinkN1', 'LinkN2', 'LinkN3', 'LinkN4', 'LinkN5', 'LinkN6', 'LinkN7']
    
    for i, link in enumerate(link_names):
        qpos_col = f'qpos_{link}'
        qvel_col = f'qvel_{link}'
        
        qpos[:, i] = df[qpos_col].values
        qvel[:, i] = df[qvel_col].values
        
    return qpos, qvel
