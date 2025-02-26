
from gymnasium.envs.registration import register

register(
    id='RobotAnimationEnv-v0',
    entry_point='robot_animation.env:RobotAnimationEnv',
    max_episode_steps=1000,
)