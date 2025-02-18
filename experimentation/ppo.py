"""
A reimplementation of the PPO algorithm.
This file is more of an exercise/memory refresher.
A production version of this implementation will eventually
be introduced.

Implementation recipe: 
Env:
- We need to modify the env to compute our reward
- make it so that we can vectorize environments
- normalize observations 
- scale rewards
- return "done" when some termination criteria is met

Algorithm related:
- step through N envs M times to collect data
    - data: [observation, reward, action, done, etc.]
- Generalized Advantage Estimation implementation
- policy network (actor)
- value network (critic)
    - L2 Reg 
- minibatch updates (of each network)




"""