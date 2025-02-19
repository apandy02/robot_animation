from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
from gymnasium import utils
DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class RobotAnimationEnv(MujocoEnv, utils.EzPickle):
    """
    Our environment definition for transferring robot behavior 
    from animation to simulation using RL. 
    """
    metadata = {
        "render_fps": 20,
    }
    max_timesteps = 10
    def __init__(
            self,
            model_path: str,
            frame_skip: int,
            **kwargs,
        ):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        self.time = 0 # note, we also have data.time
        
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
        reward = 0
        self.time += 1
        terminated = self.timestep >= self.max_timesteps
        info = {}
        return observation, reward, terminated, False, info
    
    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        observation = np.concatenate((position, velocity)).ravel()
        return observation