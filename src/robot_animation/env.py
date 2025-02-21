import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class RobotAnimationEnv(MujocoEnv, utils.EzPickle):
    """
    Custom environment definition for transferring robot behavior 
    from animation to simulation using RL. 
    """
    metadata = {
        "render_fps": 30,
        "render_modes": ["human"],
    }
    def __init__(
            self,
            model_path: str,
            frame_skip: int,
            target_qpos: np.ndarray,
            target_qvel: np.ndarray,
            num_links: int,
            **kwargs,
        ):
        # observation space is the position and velocity of the links
        observation_space = Box(low=-np.inf, high=np.inf, shape=(num_links * 2,), dtype=np.float64)
        self.target_qpos, self.target_qvel = target_qpos, target_qvel
        self.max_frames = len(target_qpos)
        self.frame_number = 1
        
        utils.EzPickle.__init__(self)
        
        MujocoEnv.__init__(
            self, 
            model_path, 
            frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Step function that applies an action to the environment and gathers 
        observations, rewards, and termination signals.

        Args:
            action: The action to apply to the environment.

        Returns:
            observation: The observation from the environment.
            reward: The reward from the environment.
            terminated: Whether the episode is terminated.
        """
        observation = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        reward = self._imitation_reward()
        self.frame_number += 1
        terminated = self.frame_number >= self.max_frames
        
        if terminated:
            self.frame_number = 1
        
        info = {}
        return observation, reward, terminated, False, info
    
    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        observation = np.concatenate((position, velocity)).ravel()
        return observation
    
    def _imitation_reward(self):
        qpos_diff = np.linalg.norm(self.data.qpos - self.target_qpos[self.frame_number], axis=1)
        qvel_diff = np.linalg.norm(self.data.qvel - self.target_qvel[self.frame_number], axis=1)
        return -np.sum(qpos_diff) - np.sum(qvel_diff)