import numpy as np
from gymnasium.spaces import Box

from .base import RobotAnimationEnv


class KukaIIWA4Env(RobotAnimationEnv):
    """
    Environment for the Kuka IIWA4 robot that implements the base RobotAnimationEnv.
    This environment is designed for transferring animation behavior to the robot using RL.
    """
    
    def __init__(
        self,
        target_qpos: np.ndarray,
        target_qvel: np.ndarray,
        model_path,
        reset_noise_scale: float = 0.1,
        render_mode: str = "human",
        base_reward_coeff: float = 0.3,
        num_q: int = 7,
        imitation_reward_coeffs: tuple[float, float] = (0.65, 0.35),
        animation_frame_rate: int = 153,
        **kwargs
    ):
        self.base_reward_coeff = base_reward_coeff
        observation_space = Box(low=-np.inf, high=np.inf, shape=(num_q * 2 + 2,), dtype=np.float64) # joint pos, joint vel, phase signal, target base rotation TODO: move downstream

        
        super().__init__(
            model_path=model_path,
            animation_frame_rate=animation_frame_rate,
            target_qpos=target_qpos,
            target_qvel=target_qvel,
            num_q=num_q,
            reset_noise_scale=reset_noise_scale,
            render_mode=render_mode,
            observation_space=observation_space,
            imitation_reward_coeffs=imitation_reward_coeffs,
            **kwargs
        )

        self.base_ctrl_range = self.model.actuator_ctrlrange[0]
        self.target_base_rotation = np.random.uniform(self.base_ctrl_range[0], self.base_ctrl_range[1])
    
    def reset_model(self):
        self.target_base_rotation = np.random.uniform(self.base_ctrl_range[0], self.base_ctrl_range[1])
        return super().reset_model()
    
    def total_reward(self) -> float:
        """
        Calculate the total reward for the current state.
        This method needs to be implemented as required by the abstract base class.
        
        Returns:
            float: The total reward value
        """
        return self._imitation_reward() + self._base_rotation_reward()

    def _get_obs(self):
        obs = super()._get_obs()
        obs = np.concatenate((obs, [self.target_base_rotation]))
        return obs
    
    def _base_rotation_reward(self):
        return self.base_reward_coeff * np.abs(self.target_base_rotation - self.data.qpos[0])

    def _imitation_reward(self):
        qpos_diff = np.linalg.norm(self.data.qpos[1:] - self.target_qpos[self.frame_number][1:], axis=0)
        qvel_diff = np.linalg.norm(self.data.qvel[1:] - self.target_qvel[self.frame_number][1:], axis=0)
        return -self.imitation_reward_coeffs[0] * np.sum(qpos_diff) - self.imitation_reward_coeffs[1] * np.sum(qvel_diff) # TODO: can generralize so we can have arbitrary characteristics being imitated
