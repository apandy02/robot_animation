import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class RobotAnimationEnv(MujocoEnv, utils.EzPickle):
    """
    Custom environment definition for transferring robot behavior from animation to simulation using RL.
    Defines the robot model, the action space, the observation space, and the imitation reward function.
    """
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 500
    }
    def __init__(
        self,
        model_path: str,
        animation_frame_rate: int,
        target_qpos: np.ndarray,
        target_qvel: np.ndarray,
        num_q: int,
        reset_noise_scale: float = 0.1,
        render_mode: str = "human",
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            model_path,
            animation_frame_rate,
            target_qpos,
            target_qvel,
            num_q,
            reset_noise_scale,
            **kwargs
        )
        
        self.target_qpos, self.target_qvel = target_qpos, target_qvel
        self._num_q = num_q
        self._reset_noise_scale = reset_noise_scale
        self.max_frames = len(target_qpos)
        self.frame_number = 1

        observation_space = Box(low=-np.inf, high=np.inf, shape=(num_q * 2 + 1,), dtype=np.float64)
        dummy_frame_skip = 1

        MujocoEnv.__init__(
            self,
            model_path,
            dummy_frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            **kwargs,
        )

        self.frame_skip = int(1 / (animation_frame_rate * self.model.opt.timestep))
    
    @property
    def num_q(self):
        """
        Get the number of q in the robot.
        """
        return self._num_q

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

        self.do_simulation(action, self.frame_skip)
        reward = self._imitation_reward()
        self.frame_number += 1
        terminated = self.frame_number >= self.max_frames
        new_observation = self._get_obs()
        
        if terminated:
            # TODO: add a termination condition for falling
            self.frame_number = 1
        
        info = {}
        return new_observation, reward, terminated, False, info
    
    def reset_model(self) -> np.ndarray:
        """
        Reset robot model to the initial state.

        Returns:
            observation: the env observation post reset 
        """
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)

        self.set_state(qpos, qvel)
        observation = self._get_obs()

        return observation
    
    def _set_action_space(self):
        bounds = self.model.jnt_range.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = Box(low=low, high=high, dtype=np.float32)
        return self.action_space
    
    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        phase_signal = self.frame_number / self.max_frames
        observation = np.concatenate((position, velocity, [phase_signal])).ravel()
        return observation
    
    def _imitation_reward(self):
        """
        Reward function that penalizes the agent for deviating from the target qpos and qvel.
        """
        qpos_diff = np.linalg.norm(self.data.qpos - self.target_qpos[self.frame_number], axis=0)
        qvel_diff = np.linalg.norm(self.data.qvel - self.target_qvel[self.frame_number], axis=0)
        return -0.65*np.sum(qpos_diff) - 0.35*np.sum(qvel_diff)
